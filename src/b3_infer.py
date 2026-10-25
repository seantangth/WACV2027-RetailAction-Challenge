#!/usr/bin/env python3
"""RAC — B3 inference: for each fuse_predict run (sid, run_id, segment frames [i, j], anchor px in both views) run the B3 model ->
per-view segment heat-map first/second peak (full-frame normalised coordinates) and peak values, segment confidence,
start/end offset probabilities (-2..+2) and frame-weight entropy.
runs[sid] = list of (run_id, i, j, ax0, ay0, ax1, ay1); returns dict[(sid, run_id)] -> dict(p0,k0,s0,q0,p1,k1,s1,q1,conf,ps,pe,went0,went1)
"""
import os, json, math, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from train_spatial import decode_frames, soft_argmax, second_peak, SPLIT_DIRS, W
from train_b3 import B3Net, make_view, seg_frames, OFF, NOFF

class B3Jobs(Dataset):
    def __init__(self, runs, R, raw, C, K, tta=False):
        self.keys = sorted(runs); self.runs = runs; self.R = R; self.raw = raw; self.C = C; self.K = K; self.tta = tta
    def __len__(self): return len(self.keys)
    def __getitem__(self, idx):
        sid = self.keys[idx]; r = self.R[sid]; items = self.runs[sid]
        frames_v = {}; n_v = {}
        for v in (0, 1):
            vw = r["views"][v]; n = vw["n_video"]
            if n <= 0: n = max(vw["poses"].shape[0], 1)
            n_v[v] = n; need = set()
            for (run_id, i, j, *_a) in items:
                for f in seg_frames(i - 1, j + 1, self.K, n):
                    for d in (-1, 0, 1): need.add(min(max(f + d, 0), n - 1))
            frames_v[v] = decode_frames(os.path.join(self.raw, SPLIT_DIRS[r["split"]], sid, f"rank{v}_video.mp4"), sorted(need))
        xs = []; exs = []; metas = []
        for (run_id, i, j, ax0, ay0, ax1, ay1) in items:
            a2, b2 = i, j + 1; lo, hi = a2 - 1, b2
            for flip in ([False, True] if self.tta else [False]):
                xv = []; ev = []; m = []
                for v, (ax, ay) in enumerate([(ax0, ay0), (ax1, ay1)]):
                    n = n_v[v]; fis = seg_frames(lo, hi, self.K, n)
                    x, ex, x0, y0 = make_view(frames_v[v], n, fis, ax, ay, self.C, r["views"][v]["poses"], lo, hi, a2, b2, v, flip=flip)
                    xv.append(x); ev.append(ex); m.append((x0, y0))
                xs.append(np.stack(xv)); exs.append(np.stack(ev)); metas.append((sid, run_id, flip, m[0][0], m[0][1], m[1][0], m[1][1]))
        if not xs: return torch.zeros(0, 2, self.K, 11, self.C, self.C, dtype=torch.float16), torch.zeros(0, 2, self.K, 10), []
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(exs)), metas

def collate(b): return torch.cat([x for x, _, _ in b], 0), torch.cat([e for _, e, _ in b], 0), sum([m for _, _, m in b], [])

def load_b3(ckpt, dev="cuda"):
    cfg = json.load(open(os.path.join(os.path.dirname(ckpt), "config.json")))
    model = B3Net(cfg["backbone"], in_chans=11, K=int(cfg["K"]), d=int(cfg["d"]), layers=int(cfg["layers"]), drop_path=0.0).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev)); model.eval()
    return model, int(cfg["crop"]), int(cfg["K"])

def run_b3(ckpts, runs, R, raw, workers=8, bs=2, tta=False, chunk=8, dev="cuda"):
    """ckpts: one or more B3 checkpoints (same crop/K); outputs are averaged."""
    models = []; C = K = None
    for ck in ckpts:
        m, C_, K_ = load_b3(ck, dev); models.append(m)
        assert C is None or (C == C_ and K == K_), "B3 ckpts must share crop/K"
        C, K = C_, K_
    ds = B3Jobs(runs, R, raw, C, K, tta=tta)
    dl = DataLoader(ds, batch_size=bs, num_workers=workers, collate_fn=collate, timeout=900, multiprocessing_context="forkserver")
    acc = {}
    with torch.no_grad():
        for x, ex, metas in dl:
            for s in range(0, len(x), chunk):
                xx = x[s:s + chunk].to(dev, non_blocking=True); ee = ex[s:s + chunk].to(dev, non_blocking=True)
                ps_sum = None; pb_sum = None; cf_sum = None; wn_sum = None
                with torch.autocast("cuda", dtype=torch.float16):
                    for m in models:
                        hm_seg, hm_frame, w, bnd, conf = m(xx, ee)
                        ps_ = torch.sigmoid(hm_seg.float()); pb_ = torch.softmax(bnd.float(), -1); cf_ = torch.sigmoid(conf.float()); wn_ = w.float()
                        ps_sum = ps_ if ps_sum is None else ps_sum + ps_; pb_sum = pb_ if pb_sum is None else pb_sum + pb_
                        cf_sum = cf_ if cf_sum is None else cf_sum + cf_; wn_sum = wn_ if wn_sum is None else wn_sum + wn_
                nm = len(models); ps = (ps_sum / nm).cpu().numpy(); pb = (pb_sum / nm).cpu().numpy(); cf = (cf_sum / nm).cpu().numpy(); wn = (wn_sum / nm).cpu().numpy()
                for k, (sid, run_id, flip, x00, y00, x10, y10) in enumerate(metas[s:s + chunk]):
                    key = (sid, run_id); hm = ps[k][:, :, ::-1] if flip else ps[k]
                    if key not in acc: acc[key] = dict(hm=np.zeros_like(hm), n=0, ps=np.zeros(NOFF), pe=np.zeros(NOFF), conf=0.0, went=np.zeros(2), off=(x00, y00, x10, y10))
                    e = acc[key]; e["hm"] += hm; e["n"] += 1; e["ps"] += pb[k, 0]; e["pe"] += pb[k, 1]; e["conf"] += float(cf[k]); e["went"] += -(wn[k] * np.log(wn[k] + 1e-9)).sum(-1)
    out = {}
    for key, e in acc.items():
        n = max(e["n"], 1); hm = e["hm"] / n; x00, y00, x10, y10 = e["off"]
        res = dict(conf=e["conf"] / n, ps=(e["ps"] / n).tolist(), pe=(e["pe"] / n).tolist(), went0=float(e["went"][0] / n), went1=float(e["went"][1] / n))
        for v, (x0, y0) in enumerate([(x00, y00), (x10, y10)]):
            px, py, pk = soft_argmax(hm[v]); qx, qy, qk = second_peak(hm[v], px, py)
            f = lambda cx, cy: (float(np.clip((x0 + cx * 4 + 2) / W, 0, 1)), float(np.clip((y0 + cy * 4 + 2) / W, 0, 1)))
            res[f"p{v}"] = f(px, py); res[f"k{v}"] = float(pk); res[f"s{v}"] = f(qx, qy); res[f"q{v}"] = float(qk)
        out[key] = res
    return out
