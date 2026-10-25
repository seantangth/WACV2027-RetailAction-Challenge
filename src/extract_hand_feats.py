#!/usr/bin/env python3
"""RAC — per-frame features from a trained, frozen spatial heat-map model (v4): every frame of every video is cropped around the hands (3 frames + hand channels);
output per frame: global pooling of the stride-32 backbone map (768) + heat-map peak value + peak offset from the anchor (3) -> float16 [n_frames, 771].
Same file layout as extract_feats.py (<out>/<split>/<id>_<v>.npy); can be fed directly to --img_feats of baseline_meta / temporal_v3.
Usage: python extract_hand_feats.py --meta_dir META_DIR --raw RAW_DIR --ckpt spatial_v4/model_ep2.pt --out FEATS_DIR --splits train,validation,test,challenge
"""
import argparse, os, pickle, json, time, numpy as np, torch, av
from torch.utils.data import Dataset, DataLoader
from train_spatial import HeatNet, crop_at, gauss_map, soft_argmax, SPLIT_DIRS, MEAN, STD, W
from metric import JIDX
HANDS = [JIDX["right_hand"], JIDX["left_hand"], JIDX["right_wrist"], JIDX["left_wrist"]]

def decode_all(path):
    out = []
    try:
        with av.open(path) as c:
            s = c.streams.video[0]; s.thread_type = "NONE"; s.thread_count = 1
            for fr in c.decode(s): out.append(fr.to_ndarray(format="rgb24"))
    except Exception: pass
    return out

def collate_list(b):
    return b

class VidDS(Dataset):
    def __init__(self, items, C, nframes): self.items = items; self.C = C; self.nf = nframes
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        split, sid, v, path, P, faces = self.items[i]
        frames = decode_all(path); n = len(frames)
        if n == 0: return split, sid, v, torch.zeros(0, 3*self.nf+2, self.C, self.C), torch.zeros(0,2)
        half = self.nf // 2; xs = []; anchors = []; last = None
        for fi in range(n):
            ax = ay = None
            if fi < P.shape[0]:
                pts = [P[fi, j, :2] for j in HANDS if not np.isnan(P[fi, j, 0])]
                if pts: ax = float(np.mean([p[1] for p in pts]))*W; ay = float(np.mean([p[0] for p in pts]))*W
            if ax is None and fi < faces.shape[0] and not np.isnan(faces[fi,0]): ax, ay = float(faces[fi,0])*W, float(faces[fi,1])*W
            if ax is None: ax, ay = (last if last is not None else (W/2, W/2))
            last = (ax, ay)
            chans = []
            for d in range(-half, half+1):
                im = frames[min(max(fi+d, 0), n-1)]
                c_, x0, y0 = crop_at(im, ax, ay, self.C)
                chans.append(((c_.astype(np.float32)/255.0 - MEAN)/STD).transpose(2,0,1))
            hm = np.zeros((2, self.C, self.C), np.float32)
            if fi < P.shape[0]:
                for k, j in enumerate((JIDX["right_hand"], JIDX["left_hand"])):
                    yx = P[fi, j, :2]
                    if not np.isnan(yx).any():
                        hx, hy = yx[1]*W - x0, yx[0]*W - y0
                        if -self.C < hx < 2*self.C and -self.C < hy < 2*self.C: hm[k] = gauss_map(self.C, self.C, hx, hy, 12.0)
            xs.append(np.concatenate(chans + [hm], 0)); anchors.append((x0, y0))
        return split, sid, v, torch.from_numpy(np.stack(xs)), torch.tensor(anchors, dtype=torch.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", required=True); ap.add_argument("--raw", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,validation,test,challenge"); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args(); dev = "cuda"
    cfg = json.load(open(os.path.join(os.path.dirname(a.ckpt), "config.json"))); C = int(cfg["crop"]); nf = int(cfg["nframes"])
    model = HeatNet(cfg["backbone"], in_chans=3*nf+2).to(dev); model.load_state_dict(torch.load(a.ckpt, map_location=dev)); model.eval().half()
    items = []
    for sp in a.splits.split(","):
        recs = [r for r in pickle.load(open(os.path.join(a.meta_dir, f"meta_{sp}.pkl"),"rb")) if "error" not in r]
        os.makedirs(os.path.join(a.out, sp), exist_ok=True)
        for r in recs:
            for v in (0,1):
                outp = os.path.join(a.out, sp, f"{r['id']}_{v}.npy")
                if os.path.exists(outp): continue
                items.append((sp, r["id"], v, os.path.join(a.raw, SPLIT_DIRS[sp], r["id"], f"rank{v}_video.mp4"), r["views"][v]["poses"], r["views"][v]["faces"]))
    print(f"videos: {len(items)} crop={C} nframes={nf}", flush=True)
    dl = DataLoader(VidDS(items, C, nf), batch_size=a.bs, num_workers=a.workers, collate_fn=collate_list, multiprocessing_context="forkserver", timeout=900)
    t0 = time.time(); done = 0
    with torch.no_grad():
        for batch in dl:
            for (sp, sid, v, x, anchors) in batch:
                outp = os.path.join(a.out, sp, f"{sid}_{v}.npy")
                if len(x) == 0: np.save(outp, np.zeros((0, 771), np.float16)); continue
                xx = x.to(dev).half()
                fs = model.bb(xx); g = fs[3].float().mean((2,3))                          # [n,768]
                p = model.lat[3](fs[3])
                for i in (2,1,0):
                    p = torch.nn.functional.interpolate(p, size=fs[i].shape[-2:], mode="bilinear", align_corners=False) + model.lat[i](fs[i])
                hm = torch.sigmoid(model.out(model.smooth(p))[:,0].float()).cpu().numpy()   # [n,H,W]
                ex = []
                for k in range(len(hm)):
                    px, py, pk = soft_argmax(hm[k]); ex.append((pk, (px*4 + 2 - C/2)/C, (py*4 + 2 - C/2)/C))
                F = np.concatenate([g.cpu().numpy(), np.array(ex, np.float32)], 1).astype(np.float16)
                np.save(outp, F); done += 1
            if done % 400 == 0: print(f"{done}/{len(items)} {time.time()-t0:.0f}s", flush=True)
    print(f"DONE {done} in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
