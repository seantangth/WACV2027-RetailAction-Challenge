#!/usr/bin/env python3
"""RAC — spatial heat-map model: crops a native-resolution frame region (C x C) centred on a hand anchor and predicts the interaction-point heat map (stride 4).

Input channels = RGB + 2 Gaussian maps of the subject's hands (right_hand / left_hand, marking the subject of interest to exclude other people).
Training: frames inside action intervals; anchor = hand joint nearest to the GT + jitter (simulating anchor error at inference).
Inference: (sample, view, frames, anchor) given by the caller -> averaged heat map -> first/second peak -> full-frame coordinates + peak confidence.

Usage:
  python train_spatial.py --meta_dir ... --raw ... --out_dir OUT_DIR --epochs 8 --crop 288 --backbone convnext_tiny.fb_in22k_ft_in1k
  (evaluation: on the test split with GT intervals + hand anchors -> point-error distribution and hit rates per threshold; rerun with --eval_only --ckpt)
"""
import argparse, os, pickle, math, time, json, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, av, timm
from torch.utils.data import Dataset, DataLoader
from metric import JIDX

SPLIT_DIRS = {"train":"train/train","validation":"validation/validation","test":"test/test",
              "challenge":"challenge_test_set/vam-dataset-challenge_test_no_labels/data"}
MEAN = np.array([0.485,0.456,0.406],np.float32); STD = np.array([0.229,0.224,0.225],np.float32)
HAND_J = [JIDX["right_hand"], JIDX["left_hand"], JIDX["right_wrist"], JIDX["left_wrist"]]
W = 600

def decode_frames(path, idxs):
    """Returns dict idx -> uint8 [600,600,3]"""
    want = set(idxs); out = {}
    try:
        with av.open(path) as c:
            s = c.streams.video[0]; s.thread_type = "NONE"; s.thread_count = 1
            for i, fr in enumerate(c.decode(s)):
                if i in want: out[i] = fr.to_ndarray(format="rgb24")
                if i >= max(want): break
    except Exception: pass
    return out

def crop_at(img, cx, cy, C):
    """Crop C x C centred on pixel (cx, cy), out-of-bounds pixels 0; returns crop, (x0, y0) top-left corner"""
    x0 = int(round(cx - C/2)); y0 = int(round(cy - C/2))
    H, Wd = img.shape[:2]
    out = np.zeros((C, C, 3), np.uint8)
    xs0, ys0 = max(0,x0), max(0,y0); xs1, ys1 = min(Wd, x0+C), min(H, y0+C)
    if xs1 > xs0 and ys1 > ys0:
        out[ys0-y0:ys1-y0, xs0-x0:xs1-x0] = img[ys0:ys1, xs0:xs1]
    return out, x0, y0

def gauss_map(h, w, cx, cy, sigma):
    ys, xs = np.mgrid[0:h, 0:w]
    return np.exp(-((xs-cx)**2 + (ys-cy)**2) / (2*sigma**2)).astype(np.float32)

def build_train_items(recs, margin=0):
    items = []
    for r in recs:
        for (s,e,x0,y0,x1,y1,_) in r["labels"]:
            for v,(gx,gy) in enumerate([(x0,y0),(x1,y1)]):
                N = r["views"][v]["n_video"]
                if N <= 0: continue
                a = max(0, int(round(s*N)) - margin); b = min(N, int(round(e*N)) + margin)
                for fi in range(a, b):
                    items.append((r, v, fi, gx, gy))
    return items

class CropDS(Dataset):
    def __init__(self, items, raw, C, train=True, jitter=22.0, stride=4, sigma_cells=1.5, color=False, nframes=1):
        self.items = items; self.raw = raw; self.C = C; self.train = train; self.jitter = jitter; self.stride = stride; self.sigma = sigma_cells; self.color = color; self.nframes = nframes
        self.cache_key = None; self.cache = None
    def __len__(self): return len(self.items)
    def hands(self, r, v, fi):
        P = r["views"][v]["poses"]
        if fi >= P.shape[0]: return [None]*4
        out = []
        for j in HAND_J:
            yx = P[fi, j, :2]
            out.append(None if np.isnan(yx).any() else (float(yx[1])*W, float(yx[0])*W))  # (x,y) px
        return out
    def __getitem__(self, i):
        r, v, fi, gx, gy = self.items[i]
        path = os.path.join(self.raw, SPLIT_DIRS[r["split"]], r["id"], f"rank{v}_video.mp4")
        half = self.nframes // 2
        fis = [fi + d for d in range(-half, half+1)]
        frames = decode_frames(path, [max(0, f) for f in fis])
        img = frames.get(fi)
        if img is None: img = np.zeros((W,W,3), np.uint8)
        neigh = [frames.get(max(0, f), img) if f != fi else img for f in fis]  # out of range: nearest valid frame
        hands = self.hands(r, v, fi)
        gxp, gyp = gx*W, gy*W
        # anchor: hand joint nearest to the GT (training); otherwise GT + large jitter
        cands = [h for h in hands if h is not None]
        if cands:
            d = [math.hypot(h[0]-gxp, h[1]-gyp) for h in cands]; ax, ay = cands[int(np.argmin(d))]
        else:
            ax, ay = gxp + np.random.randn()*40, gyp + np.random.randn()*40
        if self.train:
            ax += np.random.randn()*self.jitter; ay += np.random.randn()*self.jitter
        crops = []
        for im in neigh:
            c_, x0, y0 = crop_at(im, ax, ay, self.C); crops.append(c_)
        flip = bool(self.train and random.random() < 0.5)  # horizontal flip (GT and hands flipped together, left/right hand channels swapped)
        if flip: crops = [c_[:, ::-1].copy() for c_ in crops]
        xs_ = []
        cj = (np.random.uniform(0.75, 1.25), np.random.uniform(-0.08, 0.08), np.random.uniform(0.9, 1.1, size=(1,1,3)).astype(np.float32)) if (self.train and self.color) else None
        for c_ in crops:
            x = c_.astype(np.float32)/255.0
            if cj is not None: x = np.clip(x * cj[0] + cj[1], 0, 1) * cj[2]; x = np.clip(x, 0, 1)
            xs_.append(((x - MEAN)/STD).transpose(2,0,1))
        x = np.concatenate(xs_, 0)  # [3*nframes, C, C]
        # hand channels (in crop coordinates)
        hm_h = np.zeros((2, self.C, self.C), np.float32)
        for k,(hj) in enumerate([0,1]):  # right_hand, left_hand
            h = hands[hj]
            if h is not None:
                hx, hy = h[0]-x0, h[1]-y0
                if flip: hx = self.C-1-hx
                if -self.C < hx < 2*self.C and -self.C < hy < 2*self.C:
                    hm_h[k] = gauss_map(self.C, self.C, hx, hy, 12.0)
        if flip: hm_h = hm_h[::-1].copy()  # swap left/right hands after flipping
        x = np.concatenate([x, hm_h], 0)
        # target heat map (at stride)
        tx, ty = gxp - x0, gyp - y0
        if flip: tx = self.C-1-tx
        Hh = self.C // self.stride
        inside = 1.0 if (0 <= tx < self.C and 0 <= ty < self.C) else 0.0
        target = gauss_map(Hh, Hh, tx/self.stride, ty/self.stride, self.sigma) if inside else np.zeros((Hh,Hh),np.float32)
        return dict(x=torch.from_numpy(x), target=torch.from_numpy(target), inside=torch.tensor(inside),
                    txy=torch.tensor([tx/self.stride, ty/self.stride], dtype=torch.float32), meta=(r["id"], v, fi, x0, y0, flip))

class HeatNet(nn.Module):
    def __init__(self, backbone="convnext_tiny.fb_in22k_ft_in1k", in_chans=5, drop_path=0.0):
        super().__init__()
        kw = dict(drop_path_rate=drop_path) if drop_path > 0 else {}
        # inference (Docker, HF_HUB_OFFLINE=1) sets RAC_NO_PRETRAINED=1: no ImageNet download (load_state_dict overwrites all weights; same architecture)
        self.bb = timm.create_model(backbone, pretrained=os.environ.get("RAC_NO_PRETRAINED", "0") != "1", features_only=True, in_chans=in_chans, out_indices=(0,1,2,3), **kw)
        chs = self.bb.feature_info.channels()  # strides 4,8,16,32
        self.lat = nn.ModuleList([nn.Conv2d(c, 128, 1) for c in chs])
        self.smooth = nn.Sequential(nn.Conv2d(128,128,3,padding=1), nn.GELU(), nn.Conv2d(128,128,3,padding=1), nn.GELU())
        self.out = nn.Conv2d(128, 1, 1)
        self.conf = nn.Linear(128, 1)
    def forward(self, x):
        fs = self.bb(x)
        p = self.lat[3](fs[3])
        for i in (2,1,0):
            p = F.interpolate(p, size=fs[i].shape[-2:], mode="bilinear", align_corners=False) + self.lat[i](fs[i])
        p = self.smooth(p)
        hm = self.out(p)[:,0]                     # [B,H,W] logits at stride 4
        conf = self.conf(p.mean((2,3)))[:,0]
        return hm, conf

def soft_argmax(hm_prob, win=3):
    """hm_prob [H,W] (numpy, >= 0) -> weighted centroid (x, y) around the first peak, peak value"""
    H, Wd = hm_prob.shape; k = int(np.argmax(hm_prob)); py, px = divmod(k, Wd)
    y0, y1 = max(0,py-win), min(H,py+win+1); x0, x1 = max(0,px-win), min(Wd,px+win+1)
    sub = hm_prob[y0:y1, x0:x1]; ys, xs = np.mgrid[y0:y1, x0:x1]; s = sub.sum() + 1e-9
    return float((xs*sub).sum()/s), float((ys*sub).sum()/s), float(hm_prob[py,px])

def second_peak(hm_prob, px, py, radius=6):
    h = hm_prob.copy(); H, Wd = h.shape
    y0, y1 = max(0,int(py)-radius), min(H,int(py)+radius+1); x0, x1 = max(0,int(px)-radius), min(Wd,int(px)+radius+1)
    h[y0:y1, x0:x1] = 0
    return soft_argmax(h)

def _winit(_):
    torch.set_num_threads(1)

def run_eval(model, ds, dev, C, stride, bs=64, workers=8):
    model.eval(); dl = DataLoader(ds, batch_size=bs, num_workers=workers, timeout=900, worker_init_fn=_winit, multiprocessing_context="forkserver")
    errs = []; confs = []; insides = []
    with torch.no_grad():
        for b in dl:
            hm, conf = model(b["x"].to(dev).half() if next(model.parameters()).dtype==torch.float16 else b["x"].to(dev))
            prob = torch.sigmoid(hm).float().cpu().numpy()
            for i in range(len(prob)):
                px, py, pk = soft_argmax(prob[i])
                tx, ty = b["txy"][i].tolist()
                errs.append(math.hypot(px-tx, py-ty)*stride); confs.append(pk); insides.append(float(b["inside"][i]))
    errs = np.array(errs); ins = np.array(insides) > 0
    res = dict(n=int(len(errs)), inside_rate=float(ins.mean()), med_px=float(np.median(errs[ins])),
               p75_px=float(np.percentile(errs[ins],75)), **{f"hit_{t}px": float((errs[ins] <= t).mean()) for t in (9,18,27,45,90)})
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", required=True); ap.add_argument("--raw", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--backbone", default="convnext_tiny.fb_in22k_ft_in1k"); ap.add_argument("--crop", type=int, default=288)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--bs", type=int, default=48); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=20); ap.add_argument("--train_splits", default="train,validation"); ap.add_argument("--eval_split", default="test")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--eval_only", action="store_true"); ap.add_argument("--ckpt", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ema", type=float, default=0.0); ap.add_argument("--jitter", type=float, default=22.0); ap.add_argument("--color", action="store_true"); ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--nframes", type=int, default=1); ap.add_argument("--drop_path", type=float, default=0.0)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda"
    load = lambda sp: [r for r in pickle.load(open(os.path.join(a.meta_dir, f"meta_{sp}.pkl"),"rb")) if "error" not in r]
    tr = sum([load(s) for s in a.train_splits.split(",")], []); ev = load(a.eval_split)
    if a.limit: tr = tr[:a.limit]; ev = ev[:max(200, a.limit//10)]
    it_tr = build_train_items(tr); it_ev = build_train_items(ev)
    print(f"train crops {len(it_tr)} eval crops {len(it_ev)}", flush=True)
    ds_tr = CropDS(it_tr, a.raw, a.crop, train=True, jitter=a.jitter, color=a.color, nframes=a.nframes); ds_ev = CropDS(it_ev, a.raw, a.crop, train=False, nframes=a.nframes)
    model = HeatNet(a.backbone, in_chans=3*a.nframes+2, drop_path=a.drop_path).to(dev)
    json.dump(dict(backbone=a.backbone, crop=a.crop, nframes=a.nframes, stride=4), open(os.path.join(a.out_dir, "config.json"), "w"))
    if a.ckpt: model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    if not a.eval_only:
        dl = DataLoader(ds_tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=False, prefetch_factor=2,
                        timeout=900, worker_init_fn=_winit, multiprocessing_context="forkserver")
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        ema = None
        if a.ema > 0:
            import copy
            ema = copy.deepcopy(model).eval()
            for p_ in ema.parameters(): p_.requires_grad_(False)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs*len(dl), pct_start=0.05)
        scaler = torch.amp.GradScaler("cuda")
        for ep in range(a.epochs):
            model.train(); t0 = time.time(); tl = 0; n = 0
            for b in dl:
                x = b["x"].to(dev, non_blocking=True); tg = b["target"].to(dev); ins = b["inside"].to(dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    hm, conf = model(x)
                    # heat-map BCE (positives weighted) + confidence BCE
                    w = 1.0 + 20.0*tg
                    l_hm = (F.binary_cross_entropy_with_logits(hm.float(), tg, reduction="none")*w).mean()
                    l_c = F.binary_cross_entropy_with_logits(conf.float(), ins)
                    loss = l_hm*10 + l_c
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0); scaler.step(opt); scaler.update(); sched.step()
                if ema is not None:
                    with torch.no_grad():
                        for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(a.ema).add_(pm.detach(), alpha=1-a.ema)
                        for be, bm_ in zip(ema.buffers(), model.buffers()): be.copy_(bm_)
                tl += loss.item(); n += 1
                if n % 50 == 0: print(f"  ep{ep+1} it{n}/{len(dl)} loss {tl/n:.4f} {time.time()-t0:.0f}s", flush=True)
            print(f"ep {ep+1}/{a.epochs} loss {tl/max(n,1):.4f} time {time.time()-t0:.0f}s", flush=True)
            save_m = ema if ema is not None else model
            torch.save(save_m.state_dict(), os.path.join(a.out_dir, "model.pt")); torch.save(save_m.state_dict(), os.path.join(a.out_dir, f"model_ep{ep+1}.pt"))
            if (ep+1) % 2 == 0 or ep+1 == a.epochs:
                res = run_eval(save_m, ds_ev, dev, a.crop, 4, workers=a.workers); print(f"[eval ep{ep+1}] {res}", flush=True)
                json.dump(res, open(os.path.join(a.out_dir, f"eval_ep{ep+1}.json"), "w"))
    else:
        res = run_eval(model, ds_ev, dev, a.crop, 4, workers=a.workers); print("[eval]", res, flush=True)
        json.dump(res, open(os.path.join(a.out_dir, "eval.json"), "w"))

if __name__ == "__main__":
    main()
