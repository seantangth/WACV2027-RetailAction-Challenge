#!/usr/bin/env python3
"""RAC — B3 segment-level spatio-temporal contact model: one candidate segment (2 views x K frames, cropped 320 around one anchor) per forward pass ->
  (a) one segment heat map per view (learned frame weighting + FiLM)  (b) boundary refinement (start/end each -2..+2 frames, 5 classes)  (c) segment confidence (regressing tIoU)
Backbone = the v4 HeatNet (ConvNeXt-S + FPN, 11 input channels: t-1, t, t+1 RGB + 2 hand Gaussian maps), initialised from v4 epoch-2 weights;
frame token = stride-32 global pooling (768) + frame-level extra vector (10) -> 2-layer Transformer over the 2K tokens.
Training: GT segments of train+validation (boundary jitter 0/+-1/+-2 frames with probability 50%/35%/15%) + 20% negatives (random segments with tIoU < 0.3 to every GT).
Evaluation: test split, GT segments (fixed-seed jitter); reports the <=5 cm hit rate (min over views) of the segment heat map / per-frame oracle / per-frame mean, tIoU before/after boundary refinement, and confidence error.
Usage:
  python train_b3.py --meta_dir META --raw RAW --out_dir b3_v1 --init spatial_v4/model_ep2.pt --epochs 4 --bs 4 --workers 16
  python train_b3.py ... --eval_only --ckpt b3_v1/model.pt
"""
import argparse, os, json, pickle, math, time, random, copy, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from metric import JIDX, factors_from_meta
from train_spatial import HeatNet, decode_frames, crop_at, gauss_map, soft_argmax, second_peak, SPLIT_DIRS, MEAN, STD, W

HANDS = [JIDX["right_hand"], JIDX["left_hand"], JIDX["right_wrist"], JIDX["left_wrist"]]
EXTRA_DIM = 10; OFF = 2; NOFF = 2 * OFF + 1
JIT = [(0, 0.50), (1, 0.175), (-1, 0.175), (2, 0.075), (-2, 0.075)]

def grid_N(r):
    n = max(r["views"][0]["n_video"], r["views"][1]["n_video"])
    if n <= 0: n = max(len(r["views"][0]["poses"]), len(r["views"][1]["poses"]))
    return max(int(n), 1)

def seg_frames(lo, hi, K, n):
    idx = np.linspace(lo, hi, K).round().astype(int)
    return [int(min(max(i, 0), max(n - 1, 0))) for i in idx]

def tiou(a, b, c, d):
    inter = max(0, min(b, d) - max(a, c)); union = max(b, d) - min(a, c)
    return inter / union if union > 0 else 0.0

def draw_jit():
    x = random.random(); acc = 0.0
    for off, p in JIT:
        acc += p
        if x < acc: return off
    return 0

def hand_xy(P, fi):
    out = []
    for j in (JIDX["right_hand"], JIDX["left_hand"]):
        out.append((float(P[fi, j, 1]) * W, float(P[fi, j, 0]) * W) if (fi < P.shape[0] and not np.isnan(P[fi, j, 0])) else None)
    return out

def nearest_hand(P, fis, gx, gy):
    for fi in [fis[len(fis) // 2]] + list(fis):
        if fi < P.shape[0]:
            c = [(float(P[fi, j, 1]) * W, float(P[fi, j, 0]) * W) for j in HANDS if not np.isnan(P[fi, j, 0])]
            if c:
                d = [math.hypot(cx - gx, cy - gy) for cx, cy in c]; return c[int(np.argmin(d))]
    return None

def frame_w_target(P, fis, gx, gy, sigma=20.0):
    """Contact-frame prior: per frame, minimum distance d_k (px) of the hand joints to the GT -> softmax(-d_k/sigma); frames without pose use the mean distance of valid frames."""
    d = []
    for fi in fis:
        c = [math.hypot(float(P[fi, j, 1]) * W - gx, float(P[fi, j, 0]) * W - gy) for j in HANDS if fi < P.shape[0] and not np.isnan(P[fi, j, 0])]
        d.append(min(c) if c else np.nan)
    d = np.array(d, np.float64)
    if np.isnan(d).all(): return np.full(len(fis), 1.0 / len(fis), np.float32)
    d = np.where(np.isnan(d), np.nanmean(d), d); z = -d / sigma; z -= z.max(); w = np.exp(z); return (w / w.sum()).astype(np.float32)

def mean_hand(P, fis, faces):
    pts = [(float(P[fi, j, 1]) * W, float(P[fi, j, 0]) * W) for fi in fis if fi < P.shape[0] for j in HANDS if not np.isnan(P[fi, j, 0])]
    if pts: return (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))
    for fi in fis:
        if fi < faces.shape[0] and not np.isnan(faces[fi, 0]): return (float(faces[fi, 0]) * W, float(faces[fi, 1]) * W)
    return (W / 2, W / 2)

def make_view(frames, n, fis, ax, ay, C, P, lo, hi, a2, b2, v, flip=False, cj=None):
    """K frames of one view: x [K,11,C,C] float16, extra [K,10], (x0,y0)"""
    xs = []; ex = []; x0 = y0 = 0
    for k, fi in enumerate(fis):
        chans = []
        for d in (-1, 0, 1):
            f = min(max(fi + d, 0), max(n - 1, 0)); im = frames.get(f)
            if im is None: im = frames.get(fi)
            if im is None: im = np.zeros((W, W, 3), np.uint8)
            c_, x0, y0 = crop_at(im, ax, ay, C)
            if flip: c_ = c_[:, ::-1]
            x = c_.astype(np.float32) / 255.0
            if cj is not None: x = np.clip(np.clip(x * cj[0] + cj[1], 0, 1) * cj[2], 0, 1)
            chans.append(((x - MEAN) / STD).transpose(2, 0, 1))
        hm = np.zeros((2, C, C), np.float32); hr = [0.0, 0.0, 0.0, 0.0]
        for j, h in enumerate(hand_xy(P, fi)):
            if h is not None:
                hx, hy = h[0] - x0, h[1] - y0
                if flip: hx = C - 1 - hx
                if -C < hx < 2 * C and -C < hy < 2 * C: hm[j] = gauss_map(C, C, hx, hy, 12.0)
                hr[2 * j] = float(np.clip((hx - C / 2) / C, -2, 2)); hr[2 * j + 1] = float(np.clip((hy - C / 2) / C, -2, 2))
        if flip: hm = hm[::-1].copy(); hr = [hr[2], hr[3], hr[0], hr[1]]
        xs.append(np.concatenate(chans + [hm], 0).astype(np.float16))
        ex.append([fi / max(n, 1), (fi - lo) / max(hi - lo, 1), 1.0 if a2 <= fi < b2 else 0.0, float(v), *hr, (b2 - a2) / 32.0, k / max(len(fis) - 1, 1)])
    return np.stack(xs), np.array(ex, np.float32), x0, y0

class SegDS(Dataset):
    """item = (rec, label_idx or -1 = negative). Returns the K-frame crops of both views, extra vectors, heat-map targets, boundary offset classes and the tIoU target."""
    def __init__(self, recs, raw, C, K=8, train=True, jitter=30.0, color=True, neg_frac=0.2, seed=0):
        self.raw = raw; self.C = C; self.K = K; self.train = train; self.jitter = jitter; self.color = color; self.seed = seed
        self.items = [(r, li) for r in recs for li in range(len(r["labels"]))]
        rng = random.Random(seed)
        if neg_frac > 0 and recs:
            self.items += [(rng.choice(recs), -1) for _ in range(int(len(self.items) * neg_frac))]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        r, li = self.items[i]
        if not self.train: random.seed(self.seed * 1000003 + i)   # evaluation: fixed jitter
        N = grid_N(r); K = self.K; C = self.C; Hh = C // 4
        if li >= 0:
            s, e, gx0, gy0, gx1, gy1, _ = r["labels"][li]
            a = max(0, min(N - 1, int(round(s * N)))); b = max(a + 1, min(N, int(round(e * N))))
            ja, jb = draw_jit(), draw_jit()
            a2 = max(0, min(N - 1, a + ja)); b2 = max(a2 + 1, min(N, b + jb))
            off_s = max(-OFF, min(OFF, a - a2)) + OFF; off_e = max(-OFF, min(OFF, b - b2)) + OFF
            tt = tiou(a, b, a2, b2); bmask = 1.0; gts = [(gx0 * W, gy0 * W), (gx1 * W, gy1 * W)]
        else:
            L = random.randint(3, 10); a2 = random.randint(0, max(0, N - L)); b2 = min(N, a2 + L)
            for _ in range(12):
                if all(tiou(int(round(s * N)), max(int(round(s * N)) + 1, int(round(e * N))), a2, b2) < 0.3 for (s, e, *_r) in r["labels"]): break
                L = random.randint(3, 10); a2 = random.randint(0, max(0, N - L)); b2 = min(N, a2 + L)
            tt = max([tiou(int(round(s * N)), max(int(round(s * N)) + 1, int(round(e * N))), a2, b2) for (s, e, *_r) in r["labels"]] + [0.0])
            off_s = off_e = OFF; bmask = 0.0; gts = [None, None]
        lo, hi = a2 - 1, b2
        xs = []; exs = []; tgs = []; ins = []; metas = []; wts = []
        cj = (np.random.uniform(0.75, 1.25), np.random.uniform(-0.08, 0.08), np.random.uniform(0.9, 1.1, size=(1, 1, 3)).astype(np.float32)) if (self.train and self.color) else None
        for v in (0, 1):
            vw = r["views"][v]; n = vw["n_video"]; P = vw["poses"]
            if n <= 0: n = max(P.shape[0], 1)
            fis = seg_frames(lo, hi, K, n)
            path = os.path.join(self.raw, SPLIT_DIRS[r["split"]], r["id"], f"rank{v}_video.mp4")
            need = sorted(set(min(max(f + d, 0), n - 1) for f in fis for d in (-1, 0, 1)))
            frames = decode_frames(path, need)
            g = gts[v]
            anc = nearest_hand(P, fis, g[0], g[1]) if g is not None else None
            if anc is None: anc = (g[0] + np.random.randn() * 40, g[1] + np.random.randn() * 40) if g is not None else mean_hand(P, fis, vw["faces"])
            ax, ay = anc
            if self.train: ax += np.random.randn() * self.jitter; ay += np.random.randn() * self.jitter
            flip = bool(self.train and random.random() < 0.5)
            x, ex, x0, y0 = make_view(frames, n, fis, ax, ay, C, P, lo, hi, a2, b2, v, flip=flip, cj=cj)
            if g is not None:
                tx, ty = g[0] - x0, g[1] - y0
                if flip: tx = C - 1 - tx
                inside = 1.0 if (0 <= tx < C and 0 <= ty < C) else 0.0
                tg = gauss_map(Hh, Hh, tx / 4, ty / 4, 1.5) if inside else np.zeros((Hh, Hh), np.float32)
            else:
                inside = 0.0; tg = np.zeros((Hh, Hh), np.float32); tx = ty = -1.0
            xs.append(x); exs.append(ex); tgs.append(tg); ins.append(inside); metas.append((x0, y0, flip, tx, ty, fis))
            wts.append(frame_w_target(P, fis, g[0], g[1]) if g is not None else np.full(K, 1.0 / K, np.float32))
        return dict(x=torch.from_numpy(np.stack(xs)), extra=torch.from_numpy(np.stack(exs)), target=torch.from_numpy(np.stack(tgs)), wt=torch.from_numpy(np.stack(wts)),
                    inside=torch.tensor(ins), off_s=torch.tensor(off_s), off_e=torch.tensor(off_e), bmask=torch.tensor(bmask), tiou=torch.tensor(tt, dtype=torch.float32),
                    seg=torch.tensor([a2, b2, N]), pos=torch.tensor(1.0 if li >= 0 else 0.0), idx=i)

class B3Net(nn.Module):
    def __init__(self, backbone="convnext_small.fb_in22k_ft_in1k", in_chans=11, K=8, d=256, layers=2, drop_path=0.2, extra_dim=EXTRA_DIM):
        super().__init__()
        self.heat = HeatNet(backbone, in_chans=in_chans, drop_path=drop_path)
        self.K = K; self.d = d
        self.tok = nn.Linear(768, d); self.tok_extra = nn.Linear(extra_dim, d)
        self.view_emb = nn.Parameter(torch.zeros(2, d)); self.pos_emb = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.pos_emb, std=0.02); nn.init.normal_(self.view_emb, std=0.02)
        enc = nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers); self.norm = nn.LayerNorm(d)
        self.w_head = nn.Linear(d, 1); self.film = nn.Linear(d, 256)
        self.out_seg = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.GELU(), nn.Conv2d(128, 1, 1))
        self.bnd = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 2 * NOFF))
        self.conf = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.out_seg[-1].weight); nn.init.zeros_(self.out_seg[-1].bias)   # segment heat map = pretrained heat.out (fused map) + residual (initialised to 0)
    def forward(self, x, extra):
        B, V, K = x.shape[:3]
        xf = x.flatten(0, 2)
        fs = self.heat.bb(xf); g = fs[3].mean((2, 3))
        p = self.heat.lat[3](fs[3])
        for i in (2, 1, 0): p = F.interpolate(p, size=fs[i].shape[-2:], mode="bilinear", align_corners=False) + self.heat.lat[i](fs[i])
        p = self.heat.smooth(p)                                                    # [BVK,128,H,W]
        H, Wd = p.shape[-2:]
        hm_frame = self.heat.out(p)[:, 0].view(B, V, K, H, Wd)
        t = self.tok(g).view(B, V, K, -1) + self.tok_extra(extra) + self.view_emb[None, :, None, :] + self.pos_emb[None, None, :K, :]
        h = self.norm(self.enc(t.flatten(1, 2))).view(B, V, K, -1)
        w = torch.softmax(self.w_head(h)[..., 0].float(), dim=2)                 # [B,V,K]
        pv = p.view(B, V, K, -1, H, Wd)
        seg = (pv * w[..., None, None, None].to(pv.dtype)).sum(2)                 # [B,V,128,H,W]
        ctx = h.mean(2)                                                            # [B,V,d]
        gam, bet = self.film(ctx).chunk(2, -1)
        seg = seg * (1 + gam[..., None, None].to(seg.dtype)) + bet[..., None, None].to(seg.dtype)
        segf = seg.flatten(0, 1)
        hm_seg = (self.heat.out(segf) + self.out_seg(segf))[:, 0].view(B, V, H, Wd)
        pooled = torch.cat([ctx[:, 0], ctx[:, 1]], -1)
        return hm_seg, hm_frame, w, self.bnd(pooled).view(B, 2, NOFF), self.conf(pooled)[:, 0]

def _winit(_): torch.set_num_threads(1)

def losses(out, b, dev, w_frame=1.0):
    hm_seg, hm_frame, w, bnd, conf = out
    tg = b["target"].to(dev); wt = 1.0 + 20.0 * tg
    l_seg = (F.binary_cross_entropy_with_logits(hm_seg.float(), tg, reduction="none") * wt).mean() * 10
    tgf = tg[:, :, None].expand_as(hm_frame)
    l_frame = (F.binary_cross_entropy_with_logits(hm_frame.float(), tgf, reduction="none") * (1.0 + 20.0 * tgf)).mean() * 10 * 0.3
    bm = b["bmask"].to(dev)
    ce = F.cross_entropy(bnd[:, 0].float(), b["off_s"].to(dev), reduction="none") + F.cross_entropy(bnd[:, 1].float(), b["off_e"].to(dev), reduction="none")
    l_bnd = (ce * bm).sum() / bm.sum().clamp(min=1)
    l_conf = F.binary_cross_entropy_with_logits(conf.float(), b["tiou"].to(dev))
    wt = b["wt"].to(dev); pos = b["pos"].to(dev)                                  # frame-weight supervision (contact-frame prior), positives only
    l_w = (-(wt * torch.log(w.float() + 1e-6)).sum(-1).mean(-1) * pos).sum() / pos.sum().clamp(min=1) * w_frame
    return l_seg + l_frame + l_bnd + l_conf + l_w, dict(seg=l_seg.item(), frame=l_frame.item(), bnd=l_bnd.item(), conf=l_conf.item(), w=l_w.item())

def run_eval(model, ds, dev, fac, bs=4, workers=8, tag=""):
    model.eval(); dl = DataLoader(ds, batch_size=bs, num_workers=workers, timeout=900, worker_init_fn=_winit, multiprocessing_context="forkserver")
    per = collections.defaultdict(dict); rows = []
    with torch.no_grad():
        for b in dl:
            with torch.autocast("cuda", dtype=torch.float16):
                hm_seg, hm_frame, w, bnd, conf = model(b["x"].to(dev), b["extra"].to(dev))
            ps = torch.sigmoid(hm_seg.float()).cpu().numpy(); pf = torch.sigmoid(hm_frame.float()).cpu().numpy()
            bo = bnd.float().argmax(-1).cpu().numpy() - OFF; cf = torch.sigmoid(conf.float()).cpu().numpy(); wn = w.cpu().numpy()
            for i in range(len(ps)):
                idx = int(b["idx"][i]); r, li = ds.items[idx]
                if li < 0: rows.append(dict(pos=0, conf=float(cf[i]), tiou=float(b["tiou"][i]))); continue
                a2, b2, N = [int(v) for v in b["seg"][i]]
                s, e = r["labels"][li][:2]; a = max(0, min(N - 1, int(round(s * N)))); bb = max(a + 1, min(N, int(round(e * N))))
                a3 = max(0, min(N - 1, a2 + int(bo[i, 0]))); b3 = max(a3 + 1, min(N, b2 + int(bo[i, 1])))
                row = dict(pos=1, conf=float(cf[i]), tiou=float(b["tiou"][i]), tiou_ref=tiou(a, bb, a3, b3), bnd_ok=float(bo[i, 0] == (int(b["off_s"][i]) - OFF) and bo[i, 1] == (int(b["off_e"][i]) - OFF)),
                           w_ent=float(-(wn[i] * np.log(wn[i] + 1e-9)).sum(-1).mean()))
                ds_ = {k: [np.inf, np.inf] for k in ("seg", "oracle", "avg")}
                for v in (0, 1):
                    # target in crop coordinates: recovered from the argmax of the target heat map (when inside)
                    tg = b["target"][i, v].numpy()
                    if tg.max() <= 0: continue
                    k = int(np.argmax(tg)); ty_, tx_ = divmod(k, tg.shape[1])
                    f = fac[r["id"]][v]
                    def dist(m):
                        px, py, _ = soft_argmax(m); return math.hypot(px - tx_, py - ty_) * 4 / W * f
                    ds_["seg"][v] = dist(ps[i, v]); ds_["avg"][v] = dist(pf[i, v].mean(0)); ds_["oracle"][v] = min(dist(pf[i, v, k_]) for k_ in range(pf.shape[2]))
                for k_ in ds_: row[k_] = min(ds_[k_])
                rows.append(row)
    pos = [x for x in rows if x["pos"] == 1]; neg = [x for x in rows if x["pos"] == 0]
    res = dict(n_pos=len(pos), n_neg=len(neg))
    for k_ in ("seg", "oracle", "avg"):
        d = np.array([x[k_] for x in pos if np.isfinite(x[k_])]); res[f"{k_}_hit05"] = float((d <= 0.05).mean()) if len(d) else 0; res[f"{k_}_hit10"] = float((d <= 0.10).mean()) if len(d) else 0; res[f"{k_}_med_m"] = float(np.median(d)) if len(d) else 0
    res["tiou_in"] = float(np.mean([x["tiou"] for x in pos])); res["tiou_refined"] = float(np.mean([x["tiou_ref"] for x in pos])); res["bnd_acc"] = float(np.mean([x["bnd_ok"] for x in pos]))
    res["tiou05_in"] = float(np.mean([x["tiou"] >= 0.5 for x in pos])); res["tiou05_refined"] = float(np.mean([x["tiou_ref"] >= 0.5 for x in pos]))
    res["conf_pos"] = float(np.mean([x["conf"] for x in pos])); res["conf_neg"] = float(np.mean([x["conf"] for x in neg])) if neg else -1; res["w_ent"] = float(np.mean([x["w_ent"] for x in pos]))
    print(f"[eval{tag}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()), flush=True)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", required=True); ap.add_argument("--raw", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--init", default="", help="v4 HeatNet weights (model_ep2.pt) to initialise backbone + FPN"); ap.add_argument("--ckpt", default=""); ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--backbone", default="convnext_small.fb_in22k_ft_in1k"); ap.add_argument("--crop", type=int, default=320); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=2); ap.add_argument("--drop_path", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--lr_head", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--ema", type=float, default=0.999); ap.add_argument("--jitter", type=float, default=30.0); ap.add_argument("--neg_frac", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--train_splits", default="train,validation"); ap.add_argument("--eval_split", default="test")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--eval_limit", type=int, default=600); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w_frame", type=float, default=0.0, help="loss weight of the frame-weight supervision (contact-frame prior softmax(-d/20px)); 0 = off (v1)")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev = "cuda"
    load = lambda sp: [r for r in pickle.load(open(os.path.join(a.meta_dir, f"meta_{sp}.pkl"), "rb")) if "error" not in r]
    ev = [r for r in load(a.eval_split) if r["labels"]]
    if a.eval_limit: ev = ev[:a.eval_limit]
    fac = factors_from_meta(ev)
    model = B3Net(a.backbone, in_chans=11, K=a.K, d=a.d, layers=a.layers, drop_path=a.drop_path).to(dev)
    json.dump(dict(type="b3", backbone=a.backbone, crop=a.crop, nframes=3, K=a.K, d=a.d, layers=a.layers, stride=4), open(os.path.join(a.out_dir, "config.json"), "w"))
    if a.ckpt: model.load_state_dict(torch.load(a.ckpt, map_location=dev)); print("loaded", a.ckpt, flush=True)
    elif a.init: model.heat.load_state_dict(torch.load(a.init, map_location=dev)); print("init heat from", a.init, flush=True)
    ds_ev = SegDS(ev, a.raw, a.crop, K=a.K, train=False, neg_frac=0.2, seed=123)
    if a.eval_only:
        res = run_eval(model, ds_ev, dev, fac, bs=a.bs, workers=a.workers); json.dump(res, open(os.path.join(a.out_dir, "eval.json"), "w")); return
    tr = sum([load(s) for s in a.train_splits.split(",")], [])
    if a.limit: tr = tr[:a.limit]
    ds_tr = SegDS(tr, a.raw, a.crop, K=a.K, train=True, jitter=a.jitter, color=True, neg_frac=a.neg_frac, seed=a.seed)
    print(f"train clips {len(tr)} samples {len(ds_tr)} | eval clips {len(ev)} samples {len(ds_ev)} | crops/epoch {len(ds_tr) * 2 * a.K}", flush=True)
    dl = DataLoader(ds_tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, prefetch_factor=2, timeout=900, worker_init_fn=_winit, multiprocessing_context="forkserver", persistent_workers=False)
    heat_p = list(model.heat.parameters()); heat_ids = {id(p) for p in heat_p}; head_p = [p for p in model.parameters() if id(p) not in heat_ids]
    opt = torch.optim.AdamW([dict(params=heat_p, lr=a.lr), dict(params=head_p, lr=a.lr_head)], weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr_head], total_steps=a.epochs * len(dl), pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda"); ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters(): p_.requires_grad_(False)
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tl = 0; n = 0; parts = collections.defaultdict(float)
        for b in dl:
            x = b["x"].to(dev, non_blocking=True); ex = b["extra"].to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x, ex)
            loss, pd = losses(out, b, dev, w_frame=a.w_frame)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0); scaler.step(opt); scaler.update(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
                for be, bm_ in zip(ema.buffers(), model.buffers()): be.copy_(bm_)
            tl += loss.item(); n += 1
            for k_, v_ in pd.items(): parts[k_] += v_
            if n % 50 == 0: print(f"  ep{ep + 1} it{n}/{len(dl)} loss {tl / n:.4f} " + " ".join(f"{k_} {v_ / n:.3f}" for k_, v_ in parts.items()) + f" {time.time() - t0:.0f}s {n * a.bs / (time.time() - t0):.2f} samp/s", flush=True)
        print(f"ep {ep + 1}/{a.epochs} loss {tl / max(n, 1):.4f} time {time.time() - t0:.0f}s", flush=True)
        torch.save(ema.state_dict(), os.path.join(a.out_dir, "model.pt")); torch.save(ema.state_dict(), os.path.join(a.out_dir, f"model_ep{ep + 1}.pt"))
        res = run_eval(ema, ds_ev, dev, fac, bs=a.bs, workers=a.workers, tag=f" ep{ep + 1}"); json.dump(res, open(os.path.join(a.out_dir, f"eval_ep{ep + 1}.json"), "w"))
    print("B3_TRAIN_DONE", flush=True)

if __name__ == "__main__":
    main()
