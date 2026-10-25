#!/usr/bin/env python3
"""RAC — fusion inference: temporal models (raw_*.pkl per-frame probabilities, averaged over models) -> segment candidates; the spatial heat-map model crops frames
inside each segment, anchored on the temporal models' predicted point -> averaged heat map -> first/second peak -> candidate points. Anchors use only information available at inference (no GT).

Usage:
  python fuse_predict.py --meta_dir META_DIR --raw_root RAW_DIR --split test \
     --temporal_raw out/b1_s0/raw_test.pkl out/meta_v0/raw_test.pkl --spatial_ckpt out/spatial_v0/model.pt [--band] [--grid]
  python fuse_predict.py ... --split challenge --temporal_raw ... raw_challenge.pkl ... --write sub.csv --sample_sub .../sample_submission.csv
"""
import argparse, os, pickle, json, csv, math, bisect, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from metric import evaluate, factors_from_meta, gts_from_meta, JIDX
from baseline_meta import decode, write_submission
from reeval import band, ranknorm, merge_queries
from train_spatial import HeatNet, decode_frames, crop_at, gauss_map, soft_argmax, second_peak, SPLIT_DIRS, MEAN, STD, W

class VideoJobs(Dataset):
    """Each item = all crop jobs of one video (sid, v). With tta=True each crop also gets a horizontally flipped copy. nframes=3 uses input (t-1, t, t+1)."""
    def __init__(self, jobs, recs_by_id, raw_root, C, tta=False, nframes=1):
        self.keys = sorted(jobs.keys()); self.jobs = jobs; self.recs = recs_by_id; self.raw = raw_root; self.C = C; self.tta = tta; self.nframes = nframes
    def __len__(self): return len(self.keys)
    def __getitem__(self, i):
        sid, v = self.keys[i]; r = self.recs[sid]
        path = os.path.join(self.raw, SPLIT_DIRS[r["split"]], sid, f"rank{v}_video.mp4")
        items = self.jobs[(sid, v)]  # list of (run_id, fi, ax_px, ay_px)
        half = self.nframes // 2
        need = sorted(set(max(0, it[1] + d) for it in items for d in range(-half, half+1)))
        frames = decode_frames(path, need)
        P = r["views"][v]["poses"]
        xs = []; metas = []
        for (run_id, fi, ax, ay) in items:
            img = frames.get(fi)
            if img is None: img = np.zeros((W,W,3), np.uint8)
            chans = []
            for d in range(-half, half+1):
                im = frames.get(max(0, fi + d), img) if d != 0 else img
                c_, x0, y0 = crop_at(im, ax, ay, self.C)
                chans.append(((c_.astype(np.float32)/255.0 - MEAN)/STD).transpose(2,0,1))
            x = np.concatenate(chans, 0)  # [3*nframes, C, C]
            hm = np.zeros((2, self.C, self.C), np.float32)
            if fi < P.shape[0]:
                for k, j in enumerate((JIDX["right_hand"], JIDX["left_hand"])):
                    yx = P[fi, j, :2]
                    if not np.isnan(yx).any():
                        hx, hy = yx[1]*W - x0, yx[0]*W - y0
                        if -self.C < hx < 2*self.C and -self.C < hy < 2*self.C: hm[k] = gauss_map(self.C, self.C, hx, hy, 12.0)
            xs.append(np.concatenate([x, hm], 0)); metas.append((sid, v, run_id, x0, y0, False))
            if self.tta:
                xf = x[:, :, ::-1].copy(); hmf = hm[::-1, :, ::-1].copy()   # flip the image horizontally; swap and mirror the hand channels
                xs.append(np.concatenate([xf, hmf], 0)); metas.append((sid, v, run_id, x0, y0, True))
        return torch.from_numpy(np.stack(xs)), metas

def collate(b):
    return torch.cat([x for x,_ in b], 0), sum([m for _,m in b], [])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", required=True); ap.add_argument("--raw_root", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--temporal_raw", nargs="+", required=True); ap.add_argument("--spatial_ckpt", nargs="+", required=True)
    ap.add_argument("--backbone", default="convnext_tiny.fb_in22k_ft_in1k"); ap.add_argument("--crop", type=int, default=288); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--k_runs", type=int, default=2); ap.add_argument("--thr", type=float, default=0.4); ap.add_argument("--n_frames", type=int, default=5)
    ap.add_argument("--band", action="store_true"); ap.add_argument("--ranknorm", action="store_true"); ap.add_argument("--grid", action="store_true")
    ap.add_argument("--write", default=""); ap.add_argument("--sample_sub", default=""); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--min_peak", type=float, default=0.05); ap.add_argument("--save_pts", default="")
    ap.add_argument("--snap", action="store_true"); ap.add_argument("--shift", action="store_true"); ap.add_argument("--bnd", action="store_true"); ap.add_argument("--tta", action="store_true")
    ap.add_argument("--multi", action="store_true"); ap.add_argument("--thr2", type=float, default=0.25); ap.add_argument("--split_len", type=int, default=10)
    ap.add_argument("--queries", action="store_true"); ap.add_argument("--save_feats", default=""); ap.add_argument("--refine", action="store_true")
    ap.add_argument("--bvar", action="store_true", help="add +-1-frame boundary variants for tier-0/1 candidates (tier 3, kind 6-9)")
    ap.add_argument("--alt_secondary", action="store_true", help="also add the 4 spatial alternative points for candidates of secondary runs (tier 1)")
    ap.add_argument("--b3_ckpt", nargs="*", default=[], help="B3 segment-model checkpoints (averaged if several): adds B3 point candidates (kind 11), boundary-refined candidates (kind 10) and rescorer features b3_*")
    ap.add_argument("--b3_points", action="store_true", help="use the B3 first peak for main candidates; the v4 first peak becomes an alternative (kind 12)")
    ap.add_argument("--b3_tta", action="store_true"); ap.add_argument("--b3_bs", type=int, default=2)
    ap.add_argument("--legacy_feats_order", action="store_true", help="write feats in the old append order (differs from the CSV's out+extra order; rescore2.align reorders automatically). Default: same order as the CSV")
    a = ap.parse_args(); dev = "cuda"
    recs = [r for r in pickle.load(open(os.path.join(a.meta_dir, f"meta_{a.split}.pkl"),"rb")) if "error" not in r]
    R = {r["id"]: r for r in recs}
    raws = [pickle.load(open(f,"rb")) for f in a.temporal_raw]
    raw = {}
    for sid in raws[0]:
        rs = [x[sid] for x in raws]
        L = min(len(x["a"]) for x in rs)
        raw[sid] = dict(a=np.mean([x["a"][:L] for x in rs],0), s=np.mean([x["s"][:L] for x in rs],0), e=np.mean([x["e"][:L] for x in rs],0),
                        pts=np.mean([x["pts"][:L] for x in rs],0), has=float(np.mean([x["has"] for x in rs])), N=min(int(x["N"]) for x in rs))
        bs_ = [x for x in rs if "ds" in x]
        if bs_: raw[sid]["bnd"] = (np.mean([x["ds"][:L] for x in bs_],0), np.mean([x["de"][:L] for x in bs_],0))
    # 1) temporal candidates (with tier)
    tcands = {sid: decode(r["a"], r["s"], r["e"], r["pts"], r["has"], r["N"], k_runs=a.k_runs, boundary_variants=True, thr=a.thr, use_has=False, return_tier=True,
                          snap_se=a.snap, shift_variants=a.shift, bnd=(r.get("bnd") if a.bnd else None), multi=a.multi, thr2=a.thr2, split_len=a.split_len) for sid, r in raw.items()}
    if a.queries: tcands = merge_queries(tcands, raws)
    # 2) crop jobs: one run per main candidate (tier 0, or tier 2 and not a variant; identified by (start, end))
    jobs = {}; run_info = {}; run_anchor = {}
    for sid, cs in tcands.items():
        N = raw[sid]["N"]; seen = {}
        for c in cs:
            key = (round(c[1]*N), round(c[2]*N))
            if key in seen: continue
            # only main candidates (not variants) need heat maps; variants share the main candidate's point. Simplification: one run per (sid, key).
            run_id = len(seen); seen[key] = run_id
            i, j = key[0], max(key[0], key[1]-1)
            fis = sorted(set(np.linspace(i, j, min(a.n_frames, j-i+1)).round().astype(int).tolist()))
            for v in (0,1):
                ax, ay = c[3+2*v]*W, c[4+2*v]*W
                jobs.setdefault((sid, v), []).extend([(run_id, fi, ax, ay) for fi in fis])
            run_info[(sid, run_id)] = key; run_anchor[(sid, run_id)] = (c[3]*W, c[4]*W, c[5]*W, c[6]*W)
        tcands[sid] = (cs, seen)
    print(f"videos with jobs: {len(jobs)}  crops: {sum(len(v) for v in jobs.values())}", flush=True)
    # 3) heat maps (optional second pass: re-crop anchored on the first-pass peak)
    models = []; nframes_set = set(); crop_set = set()
    for ck in a.spatial_ckpt:
        cfgp = os.path.join(os.path.dirname(ck), "config.json")
        cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else dict(backbone=a.backbone, crop=a.crop, nframes=1)
        nf = int(cfg.get("nframes", 1)); nframes_set.add(nf); crop_set.add(int(cfg.get("crop", a.crop)))
        m = HeatNet(cfg.get("backbone", a.backbone), in_chans=3*nf+2).to(dev); m.load_state_dict(torch.load(ck, map_location=dev)); m.eval(); models.append(m)
    assert len(nframes_set) == 1 and len(crop_set) == 1, f"all spatial ckpts must share nframes/crop, got {nframes_set} {crop_set}"
    nframes = nframes_set.pop(); a.crop = crop_set.pop()
    print(f"spatial ckpts: {len(models)} nframes={nframes} crop={a.crop}", flush=True)

    def run_heatmaps(jobs_):
        ds = VideoJobs(jobs_, R, a.raw_root, a.crop, tta=a.tta, nframes=nframes)
        dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, collate_fn=collate, timeout=900, multiprocessing_context="forkserver")
        acc_ = {}
        with torch.no_grad():
            for x, metas in dl:
                xx = x.to(dev, non_blocking=True); prob = None
                with torch.autocast("cuda", dtype=torch.float16):
                    for m in models:
                        hm, conf = m(xx); pr = torch.sigmoid(hm.float())
                        prob = pr if prob is None else prob + pr
                prob = (prob / len(models)).cpu().numpy()
                for k, (sid, v, run_id, x0, y0, flipped) in enumerate(metas):
                    key = (sid, v, run_id)
                    if key not in acc_: acc_[key] = [np.zeros_like(prob[k]), 0, x0, y0]
                    acc_[key][0] += (prob[k][:, ::-1] if flipped else prob[k]); acc_[key][1] += 1
        pts_ = {}
        for (sid, v, run_id), (s_, n, x0, y0) in acc_.items():
            pm = s_ / max(n, 1)
            px, py, pk = soft_argmax(pm); qx, qy, qk = second_peak(pm, px, py)
            f = lambda cx, cy: (float(np.clip((x0 + cx*a.stride + a.stride/2)/W, 0, 1)), float(np.clip((y0 + cy*a.stride + a.stride/2)/W, 0, 1)))
            pts_[(sid, v, run_id)] = (f(px, py), pk, f(qx, qy), qk)
        return pts_

    pts = run_heatmaps(jobs)
    if a.refine:
        # second pass: anchor = first-pass peak (keep the original anchor if the peak is too low), same frame set
        jobs2 = {}
        for (sid, v), items in jobs.items():
            new_items = []
            for (run_id, fi, ax, ay) in items:
                p1 = pts.get((sid, v, run_id))
                if p1 is not None and p1[1] >= a.min_peak:
                    ax, ay = p1[0][0]*W, p1[0][1]*W
                new_items.append((run_id, fi, ax, ay))
            jobs2[(sid, v)] = new_items
        pts2 = run_heatmaps(jobs2)
        merged = {}
        for key, p1 in pts.items():
            p2 = pts2.get(key)
            merged[key] = p2 if (p2 is not None and p2[1] >= a.min_peak) else p1
        pts = merged
        print("refine pass done", flush=True)
    if a.save_pts: pickle.dump(pts, open(a.save_pts, "wb"))
    b3 = None
    if a.b3_ckpt:
        assert not a.legacy_feats_order, "--b3_ckpt requires the new feats order"
        runs_b3 = {}
        for (sid, run_id), key in run_info.items():
            i, j = key[0], max(key[0], key[1]-1); ax0, ay0, ax1, ay1 = run_anchor[(sid, run_id)]
            runs_b3.setdefault(sid, []).append((run_id, i, j, ax0, ay0, ax1, ay1))
        from b3_infer import run_b3
        t0 = __import__("time").time()
        b3 = run_b3(a.b3_ckpt, runs_b3, R, a.raw_root, workers=a.workers, bs=a.b3_bs, tta=a.b3_tta)
        print(f"b3 runs: {len(b3)} in {__import__('time').time()-t0:.0f}s", flush=True)
        if a.save_pts: pickle.dump(b3, open(a.save_pts + ".b3.pkl", "wb"))
    # 5) final candidates: main candidate at the heat-map first peak (temporal-model point if the peak is too low); add second-peak and temporal-point candidates (tier 2)
    final = {}; feats = {}
    for sid, (cs, seen) in tcands.items():
        N = raw[sid]["N"]; out = []; extra = []; fl = []
        ra = raw[sid]["a"]
        for c in cs:
            key = (round(c[1]*N), round(c[2]*N)); run_id = seen[key]
            p0 = pts.get((sid, 0, run_id)); p1 = pts.get((sid, 1, run_id))
            i0, i1 = key[0], max(key[0], key[1]-1)
            seg = ra[i0:i1+1] if i1 >= i0 and i1 < len(ra) else ra[:1]
            base = dict(score=c[0], tier=c[7], start=c[1], end=c[2], run_len=(i1-i0+1), N=N, a_mean=float(np.mean(seg)), a_max=float(np.max(seg)),
                        a_out=float(np.mean(np.concatenate([ra[:i0], ra[i1+1:]])) if N > (i1-i0+1) else 0.0), has=raw[sid]["has"], n_cands=len(cs),
                        run_id=run_id, run_s=c[1], run_e=c[2])
            if p0 is None or p1 is None:
                out.append(c); fl.append(dict(base, kind=0, k0=0.0, k1=0.0, q0=0.0, q1=0.0, d0=0.0, d1=0.0, x0=c[3], y0=c[4], x1=c[5], y1=c[6])); continue
            (m0, k0, s0, q0), (m1, k1, s1, q1) = p0, p1
            d0 = float(np.hypot(m0[0]-c[3], m0[1]-c[4])); d1 = float(np.hypot(m1[0]-c[5], m1[1]-c[6]))
            x0, y0 = m0 if k0 >= a.min_peak else (c[3], c[4]); x1, y1 = m1 if k1 >= a.min_peak else (c[5], c[6])
            out.append((c[0], c[1], c[2], x0, y0, x1, y1, c[7]))
            fl.append(dict(base, kind=1, k0=float(k0), k1=float(k1), q0=float(q0), q1=float(q1), d0=d0, d1=d1, x0=x0, y0=y0, x1=x1, y1=y1))
            bo = b3.get((sid, run_id)) if b3 else None
            if bo is not None:
                bx0, by0 = bo["p0"] if bo["k0"] >= a.min_peak else (x0, y0); bx1, by1 = bo["p1"] if bo["k1"] >= a.min_peak else (x1, y1)
                fb = dict(k0=float(k0), k1=float(k1), q0=float(q0), q1=float(q1), d0=d0, d1=d1)
                if a.b3_points:   # main candidate uses the B3 point; the v4 point becomes an alternative (kind 12)
                    out[-1] = (c[0], c[1], c[2], bx0, by0, bx1, by1, c[7]); fl[-1].update(x0=bx0, y0=by0, x1=bx1, y1=by1)
                    extra.append((c[0]*0.49, c[1], c[2], x0, y0, x1, y1, 2)); fl.append(dict(base, score=c[0]*0.49, tier=2, kind=12, x0=x0, y0=y0, x1=x1, y1=y1, **fb))
                    x0, y0, x1, y1 = bx0, by0, bx1, by1
                else:             # B3 point as an alternative candidate (kind 11)
                    extra.append((c[0]*0.49, c[1], c[2], bx0, by0, bx1, by1, 2)); fl.append(dict(base, score=c[0]*0.49, tier=2, kind=11, x0=bx0, y0=by0, x1=bx1, y1=by1, **fb))
                ds_, de_ = int(np.argmax(bo["ps"])) - 2, int(np.argmax(bo["pe"])) - 2
                if (ds_, de_) != (0, 0):   # boundary-refined candidate (kind 10)
                    dN = 1.0 / max(N, 1); s2, e2 = c[1] + ds_*dN, min(1.0, c[2] + de_*dN)
                    if 0.0 <= s2 < e2:
                        extra.append((c[0]*0.6, s2, e2, x0, y0, x1, y1, 4)); fl.append(dict(base, score=c[0]*0.6, tier=4, kind=10, start=s2, end=e2, run_len=int(round((e2-s2)*N)), x0=x0, y0=y0, x1=x1, y1=y1, b3_refined=1.0, **fb))
            if c[7] == 0 or (a.alt_secondary and c[7] == 1):  # main candidate of the main run (or of a secondary run): add alternative points
                alts = [((c[0]*0.5, c[1], c[2], s0[0], s0[1], s1[0], s1[1], 2), 2), ((c[0]*0.45, c[1], c[2], c[3], c[4], c[5], c[6], 2), 3),
                        ((c[0]*0.48, c[1], c[2], x0, y0, s1[0], s1[1], 2), 4), ((c[0]*0.47, c[1], c[2], s0[0], s0[1], x1, y1, 2), 5)]
                for cand, kind in alts:
                    extra.append(cand); fl.append(dict(base, score=cand[0], tier=2, kind=kind, k0=float(k0), k1=float(k1), q0=float(q0), q1=float(q1), d0=d0, d1=d1, x0=cand[3], y0=cand[4], x1=cand[5], y1=cand[6]))
            if a.bvar and c[7] in (0, 1):  # +-1-frame boundary variants (using the final point)
                dN = 1.0 / max(N, 1); bv = []
                if c[1] - dN >= 0.0: bv.append(((c[0]*0.40, c[1]-dN, c[2], x0, y0, x1, y1, 3), 6))
                if c[2] + dN <= 1.0: bv.append(((c[0]*0.40, c[1], c[2]+dN, x0, y0, x1, y1, 3), 7))
                if (i1 - i0 + 1) >= 2:
                    bv.append(((c[0]*0.38, c[1]+dN, c[2], x0, y0, x1, y1, 3), 8)); bv.append(((c[0]*0.38, c[1], c[2]-dN, x0, y0, x1, y1, 3), 9))
                for cand, kind in bv:
                    extra.append(cand); fl.append(dict(base, score=cand[0], tier=3, kind=kind, start=cand[1], end=cand[2], run_len=int(round((cand[2]-cand[1])*N)), k0=float(k0), k1=float(k1), q0=float(q0), q1=float(q1), d0=d0, d1=d1, x0=x0, y0=y0, x1=x1, y1=y1))
        if not a.legacy_feats_order: fl = [f for f in fl if f["kind"] in (0, 1)] + [f for f in fl if f["kind"] not in (0, 1)]   # = out + extra order
        cands = out + extra
        rs, re_ = raw[sid]["s"], raw[sid]["e"]
        for cand, f in zip(cands, fl):   # temporal boundary evidence per candidate: start head at the start, end head at the end, actionness one frame outside, minimum actionness inside
            st = int(round(f["start"] * N)); en = int(round(f["end"] * N)); st = max(0, min(N - 1, st)); en = max(st + 1, min(N, en))
            f.update(s_st=float(rs[st]) if st < len(rs) else 0.0, e_en=float(re_[en - 1]) if en - 1 < len(re_) else 0.0,
                     a_bef=float(ra[st - 1]) if st >= 1 else 0.0, a_aft=float(ra[en]) if en < len(ra) else 0.0, a_in_min=float(np.min(ra[st:en])) if en <= len(ra) and en > st else 0.0)
        if b3:
            for cand, f in zip(cands, fl):
                bo = b3.get((sid, f["run_id"]))
                if bo is None: continue
                ds_ = int(round((f["start"] - f["run_s"]) * N)); de_ = int(round((f["end"] - f["run_e"]) * N))
                f.update(b3_conf=bo["conf"], b3_k0=bo["k0"], b3_k1=bo["k1"], b3_q0=bo["q0"], b3_q1=bo["q1"],
                         b3_ps=bo["ps"][ds_ + 2] if -2 <= ds_ <= 2 else 0.0, b3_pe=bo["pe"][de_ + 2] if -2 <= de_ <= 2 else 0.0,
                         b3_pmax_s=max(bo["ps"]), b3_pmax_e=max(bo["pe"]), b3_went=(bo["went0"] + bo["went1"]) / 2,
                         b3_dv0=float(np.hypot(f["x0"] - bo["p0"][0], f["y0"] - bo["p0"][1])), b3_dv1=float(np.hypot(f["x1"] - bo["p1"][0], f["y1"] - bo["p1"][1])), b3_refined=f.get("b3_refined", 0.0))
            cands, fl = dedupe(cands, fl, N)
        final[sid] = cands; feats[sid] = fl
    if a.save_feats:
        pickle.dump(feats, open(a.save_feats, "wb")); print("saved feats", a.save_feats, flush=True)
    if a.band: preds = band(final)
    else:
        preds = {k: [tuple(c[:7]) for c in v] for k, v in final.items()}
        if a.ranknorm: preds = ranknorm(preds)
    if any(r["labels"] for r in recs):
        gts = gts_from_meta(recs); fac = factors_from_meta(recs)
        res = evaluate(gts, preds, fac, return_grid=a.grid)
        print("FUSED:", {k:(round(v,4) if isinstance(v,float) else v) for k,v in res.items() if k!="grid"}, flush=True)
        # reference: temporal-model points only
        tp = {k: [tuple(c[:7]) for c in v] for k, v in tcands_plain(tcands).items()}
        res0 = evaluate(gts, band(tcands_plain(tcands)) if a.band else tp, fac)
        print("TEMPORAL-ONLY:", {k:(round(v,4) if isinstance(v,float) else v) for k,v in res0.items()}, flush=True)
        if a.grid:
            g = res["grid"]; np.set_printoptions(precision=2, suppress=True, linewidth=200)
            print("mean AP per spatial thr (1.0→0.05):", np.round(g.mean(1),3)); print("mean AP per temporal thr (0.05→0.5):", np.round(g.mean(0),3))
    if a.write:
        ids = [r["id"] for r in csv.DictReader(open(a.sample_sub))] if a.sample_sub else None
        write_submission(preds, a.write, ids); print("wrote", a.write, "cands:", sum(len(v) for v in preds.values()), flush=True)

def dedupe(cands, fl, N):
    """Among identical (start, end, four coordinates) candidates keep only the highest-scored one; if one of them is a B3 boundary-refined candidate, mark the kept one b3_refined=1."""
    best = {}; order = []
    for c, f in zip(cands, fl):
        key = (round(c[1]*N), round(c[2]*N), round(c[3], 4), round(c[4], 4), round(c[5], 4), round(c[6], 4))
        if key in best:
            c0, f0 = best[key]; ref = max(f.get("b3_refined", 0.0), f0.get("b3_refined", 0.0))
            if c[0] > c0[0]: best[key] = (c, f)
            best[key][1]["b3_refined"] = ref
        else: best[key] = (c, f); order.append(key)
    return [best[k][0] for k in order], [best[k][1] for k in order]

def tcands_plain(tcands):
    return {sid: cs for sid, (cs, seen) in tcands.items()}

if __name__ == "__main__":
    main()
