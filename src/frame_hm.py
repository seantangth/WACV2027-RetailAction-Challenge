#!/usr/bin/env python3
"""RAC — per-frame heat maps in full-frame coordinates, so that any candidate segment can aggregate its own contact point on the CPU.

For every clip, view and frame: a 320x320 crop anchored on the temporal models' per-frame point (top-left aligned to a multiple of 4, so heat-map cells align with the full-frame stride-4 canvas);
input = frames (t-1, t, t+1) RGB + two hand Gaussian maps (as in v4 training), averaged over several spatial checkpoints + horizontal-flip TTA (flipped on the GPU);
stores 80x80 float16 heat maps and the canvas offset (x0/4, y0/4). Heat-map cell c <-> full-frame pixel x0+4c (no +2; the training-target convention). Output <out>/<split>/<id>.npz: hm0, off0, hm1, off1.

Usage:
  python frame_hm.py --meta_dir M --raw_root RAW --split test --temporal_raw a.pkl b.pkl ... \
     --spatial_ckpt v4/model_ep2.pt v4/model_ep3.pt v4/model_ep4.pt --out frame_hm --workers 12 --bs 2 [--limit 20]
"""
import argparse, os, json, pickle, time, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from metric import JIDX
from train_spatial import HeatNet, decode_frames, gauss_map, SPLIT_DIRS, MEAN, STD, W

C = 320; STRIDE = 4; HH = C // STRIDE; CANVAS = W // STRIDE   # 80, 150

def crop_tl(img, x0, y0, c=C):
    """Crop c x c with top-left corner (x0, y0); out-of-bounds pixels are 0."""
    out = np.zeros((c, c, 3), np.uint8); H, Wd = img.shape[:2]
    xs0, ys0 = max(0, x0), max(0, y0); xs1, ys1 = min(Wd, x0 + c), min(H, y0 + c)
    if xs1 > xs0 and ys1 > ys0: out[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = img[ys0:ys1, xs0:xs1]
    return out

def mean_raw(raws, sid):
    rs = [x[sid] for x in raws if sid in x]
    L = min(len(x["pts"]) for x in rs)
    return np.mean([x["pts"][:L] for x in rs], 0), np.mean([x["a"][:L] for x in rs], 0)

class ClipFrames(Dataset):
    def __init__(self, recs, raw_root, pts):
        self.recs = recs; self.raw = raw_root; self.pts = pts
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]; sid = r["id"]; P = self.pts[sid]; out = {"id": sid}
        for v in (0, 1):
            vw = r["views"][v]; n = vw["n_video"]
            path = os.path.join(self.raw, SPLIT_DIRS[r["split"]], sid, f"rank{v}_video.mp4")
            frames = decode_frames(path, list(range(max(n, 1)))) if n > 0 else {}
            n = len(frames) if frames else 0
            xs = []; offs = []; Pose = vw["poses"]
            for fi in range(n):
                k = min(fi, len(P) - 1); ax, ay = float(P[k, 2 * v]) * W, float(P[k, 2 * v + 1]) * W
                x0 = int(4 * round((ax - C / 2) / 4)); y0 = int(4 * round((ay - C / 2) / 4))
                img = frames[fi]; chans = []
                for d in (-1, 0, 1):
                    im = frames.get(min(max(fi + d, 0), n - 1), img)
                    chans.append(((crop_tl(im, x0, y0).astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1))
                hm = np.zeros((2, C, C), np.float32)
                if fi < Pose.shape[0]:
                    for kk, j in enumerate((JIDX["right_hand"], JIDX["left_hand"])):
                        yx = Pose[fi, j, :2]
                        if not np.isnan(yx).any():
                            hx, hy = yx[1] * W - x0, yx[0] * W - y0
                            if -C < hx < 2 * C and -C < hy < 2 * C: hm[kk] = gauss_map(C, C, hx, hy, 12.0)
                xs.append(np.concatenate(chans + [hm], 0).astype(np.float16)); offs.append((x0 // 4, y0 // 4))
            out[f"x{v}"] = torch.from_numpy(np.stack(xs)) if xs else torch.zeros(0, 11, C, C, dtype=torch.float16)
            out[f"off{v}"] = np.array(offs, np.int16).reshape(-1, 2)
        return out

def collate(b): return b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", required=True); ap.add_argument("--raw_root", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--temporal_raw", nargs="+", required=True); ap.add_argument("--spatial_ckpt", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=96); ap.add_argument("--no_tta", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); dev = "cuda"
    recs = [r for r in pickle.load(open(os.path.join(a.meta_dir, f"meta_{a.split}.pkl"), "rb")) if "error" not in r]
    raws = [pickle.load(open(f, "rb")) for f in a.temporal_raw]
    pts = {}
    for r in recs:
        if any(r["id"] in x for x in raws): pts[r["id"]] = mean_raw(raws, r["id"])[0]
    od = os.path.join(a.out, a.split); os.makedirs(od, exist_ok=True)
    todo = [r for r in recs if r["id"] in pts and not os.path.exists(os.path.join(od, f"{r['id']}.npz"))]
    if a.limit: todo = todo[:a.limit]
    models = []
    for ck in a.spatial_ckpt:
        cfg = json.load(open(os.path.join(os.path.dirname(ck), "config.json")))
        assert int(cfg["crop"]) == C and int(cfg["nframes"]) == 3, cfg
        m = HeatNet(cfg["backbone"], in_chans=11).to(dev); m.load_state_dict(torch.load(ck, map_location=dev)); m.eval().half()
        models.append(m.to(memory_format=torch.channels_last))
    print(f"clips {len(todo)} (of {len(recs)}) ckpts {len(models)} tta {not a.no_tta}", flush=True)
    dl = DataLoader(ClipFrames(todo, a.raw_root, pts), batch_size=a.bs, num_workers=a.workers, collate_fn=collate, multiprocessing_context="forkserver", timeout=900, prefetch_factor=2)
    t0 = time.time(); done = 0
    with torch.no_grad():
        for batch in dl:
            for item in batch:
                res = {}
                for v in (0, 1):
                    x = item[f"x{v}"]
                    if len(x) == 0:
                        res[f"hm{v}"] = np.zeros((0, HH, HH), np.float16); res[f"off{v}"] = item[f"off{v}"]; continue
                    acc = []
                    for s in range(0, len(x), a.chunk):
                        xx = x[s:s + a.chunk].to(dev, non_blocking=True).to(memory_format=torch.channels_last); pr = None
                        for m in models:
                            hm, _ = m(xx); p = torch.sigmoid(hm.float()); pr = p if pr is None else pr + p
                            if not a.no_tta:
                                # exact flip-TTA alignment: in training, cell j <-> pixel 4j; a plain flip would shift the restored heat map by 3 pixels.
                                # so shift the input right by 3 pixels before flipping: flipped[u] = img[x0+C-4-u]; after restoring (reversed cell order) cell j' <-> pixel 4j'.
                                xs_ = torch.nn.functional.pad(xx, (3, 0, 0, 0))[..., :C]
                                xf = torch.flip(xs_, dims=[3]); xf = torch.cat([xf[:, :9], xf[:, 10:11], xf[:, 9:10]], 1)   # swap left/right hand channels
                                hf, _ = m(xf); pr = pr + torch.flip(torch.sigmoid(hf.float()), dims=[2])
                        acc.append((pr / (len(models) * (1 if a.no_tta else 2))).half().cpu().numpy())
                    res[f"hm{v}"] = np.concatenate(acc, 0); res[f"off{v}"] = item[f"off{v}"]
                np.savez(os.path.join(od, f"{item['id']}.npz"), **res); done += 1
            if done % 100 < a.bs: print(f"  {done}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print(f"FRAME_HM_DONE {done} in {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
