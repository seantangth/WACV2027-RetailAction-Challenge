#!/usr/bin/env python3
"""RAC — vectorised port of the official mAP Global metric (Kaggle notebook davidemaz/map-global-metric-code, version of 2026-10-15).

Official semantics (see official_metric.py):
  * Clips without GT actions are skipped (their predictions are not FPs).
  * m_px_factor[rank] = nanmedian( L_bone_m / dist_norm ) over all pose x bone pairs (normalised coordinates); 3.07 if it cannot be computed.
  * tIoU = max(0, inter/(union+1e-4)); a match needs tIoU >= tau_t and dist_norm*factor <= tau_s in either view (min rule, inclusive).
  * Greedy matching per (tau_s, tau_t, conf): predictions in descending score order, scores < conf skipped; each takes the unmatched GT with the largest tIoU (strict >, ties go to the first).
  * precision = tp/(tp+fp+1e-4), recall = tp/(tp+fn+1e-4) (summed over clips); precision envelope taken from right to left;
    AP = sum_k (R[k+1]-R[k]) * P_env[k], k = 0..99 (hits with score >= 1.0 do not add recall).
Input format:
  gts:   dict[id] -> list of (start,end,x0,y0,x1,y1)
  preds: dict[id] -> list of (score,start,end,x0,y0,x1,y1)
  factors: dict[id] -> (factor_rank0, factor_rank1)   unit: metres / normalised unit
"""
import math, numpy as np

BONE_LENGTH_MEANS = {
    ("neck","nose"):0.19354,
    ("left_shoulder","left_elbow"):0.27096, ("left_elbow","left_wrist"):0.21228,
    ("right_shoulder","right_elbow"):0.27210, ("right_elbow","right_wrist"):0.21316,
    ("left_hip","left_knee"):0.39204, ("left_knee","left_ankle"):0.39530,
    ("right_hip","right_knee"):0.39266, ("right_knee","right_ankle"):0.39322,
    ("left_shoulder","right_shoulder"):0.35484, ("left_hip","right_hip"):0.17150,
    ("neck","left_shoulder"):0.18136, ("neck","right_shoulder"):0.18081,
    ("left_shoulder","left_hip"):0.51375, ("right_shoulder","right_hip"):0.51226,
}
JOINTS = ["top_of_head","neck","right_shoulder","right_elbow","right_wrist","right_hand",
          "left_shoulder","left_elbow","left_wrist","left_hand","middle_of_waist",
          "right_hip","right_knee","right_ankle","right_foot","left_hip","left_knee","left_ankle","left_foot"]
JIDX = {j:i for i,j in enumerate(JOINTS)}
M_PX_FACTOR_AVG = 3.07
EPS = 1e-4
TAU_T = np.linspace(0.05, 0.5, 10)
TAU_S = np.linspace(1.0, 0.05, 20)
CONF = np.linspace(1.0, 0.0, 101)

def compute_factor(poses):
    """poses: float [n,19,3] (y,x,conf; NaN = missing). Official: median over (pose, bone) of L_m/dist_norm; 3.07 if empty.
    Official inclusion rule: joint present and (score > 0 or non-zero coordinates)."""
    ratios = []
    for (a,b),L in BONE_LENGTH_MEANS.items():
        if a not in JIDX or b not in JIDX: continue
        pa, pb = poses[:,JIDX[a],:], poses[:,JIDX[b],:]
        oka = ~np.isnan(pa[:,0]) & ((np.nan_to_num(pa[:,2])>0) | (pa[:,0]!=0) | (pa[:,1]!=0))
        okb = ~np.isnan(pb[:,0]) & ((np.nan_to_num(pb[:,2])>0) | (pb[:,0]!=0) | (pb[:,1]!=0))
        ok = oka & okb
        if not ok.any(): continue
        d = np.sqrt((pa[ok,0]-pb[ok,0])**2 + (pa[ok,1]-pb[ok,1])**2)
        d = d[d > 0]
        if d.size: ratios.append(L / d)
    if not ratios: return M_PX_FACTOR_AVG
    f = float(np.nanmedian(np.concatenate(ratios)))
    return M_PX_FACTOR_AVG if math.isnan(f) else f

def factors_from_meta(recs):
    """Use the m_px_factors computed with the official function at parse time (identical); otherwise a local approximation."""
    out = {}
    for r in recs:
        f = r.get("m_px_factors")
        out[r["id"]] = tuple(f) if f else (compute_factor(r["views"][0]["poses"]), compute_factor(r["views"][1]["poses"]))
    return out

def gts_from_meta(recs):
    return {r["id"]: [tuple(l[:6]) for l in r["labels"]] for r in recs}

def _clip_flags(scores, T, D, tau_s, tau_t):
    """Official greedy matching for one clip and one (tau_s, tau_t); returns tp_flag[P] (bool). scores are already in descending order."""
    P, G = T.shape
    used = np.zeros(G, bool); tp = np.zeros(P, bool)
    elig = (T >= tau_t) & (D <= tau_s)
    for p in range(P):
        best = -1; best_iou = 0.0
        for g in range(G):
            if used[g] or not elig[p,g]: continue
            if T[p,g] > best_iou: best_iou = T[p,g]; best = g
        if best >= 0: used[best] = True; tp[p] = True
    return tp

def _build_clips(gts, preds, factors):
    ids = sorted(set(gts.keys()) | set(preds.keys()))
    clips = []; n_gt = 0
    for i in ids:
        g = gts.get(i, [])
        if not g: continue                      # official: skip clips without GT
        p = sorted(preds.get(i, []), key=lambda x: -x[0])   # stable sort, descending
        f0, f1 = factors.get(i, (M_PX_FACTOR_AVG, M_PX_FACTOR_AVG))
        n_gt += len(g)
        P, G = len(p), len(g)
        T = np.zeros((P,G)); D = np.full((P,G), np.inf)
        for a,(sc,ps,pe,px0,py0,px1,py1) in enumerate(p):
            for b,(gs,ge,gx0,gy0,gx1,gy1) in enumerate(g):
                inter = min(pe,ge) - max(ps,gs); union = max(pe,ge) - min(ps,gs)
                T[a,b] = max(0.0, inter/(union+EPS))
                d0 = math.hypot(px0-gx0, py0-gy0)*f0; d1 = math.hypot(px1-gx1, py1-gy1)*f1
                ds = [d for d in (d0,d1) if math.isfinite(d)]
                D[a,b] = min(ds) if ds else np.inf
        clips.append((np.array([x[0] for x in p], dtype=float), T, D, G))
    return clips, n_gt

def evaluate(gts, preds, factors, return_grid=False):
    clips, n_gt = _build_clips(gts, preds, factors)
    grid = np.zeros((len(TAU_S), len(TAU_T)))
    for si, ts in enumerate(TAU_S):
        for ti, tt in enumerate(TAU_T):
            tp = np.zeros(len(CONF)); fp = np.zeros(len(CONF)); fn = np.zeros(len(CONF))
            for scores, T, D, G in clips:
                if len(scores) == 0:
                    fn += G; continue
                flags = _clip_flags(scores, T, D, ts, tt)
                # for each conf threshold: include predictions with scores >= conf (a prefix)
                incl = scores[None,:] >= CONF[:,None]            # [101,P]
                tpc = (incl & flags[None,:]).sum(1); fpc = (incl & ~flags[None,:]).sum(1)
                tp += tpc; fp += fpc; fn += (G - tpc)
            prec = tp/(tp+fp+EPS); rec = tp/(tp+fn+EPS)
            prec = np.flip(np.maximum.accumulate(np.flip(prec)))
            grid[si,ti] = float(np.sum((rec[1:]-rec[:-1]) * prec[:-1]))
    out = dict(mAP=float(grid.mean()), mAPs=float(grid[:,0].mean()), mAPt=float(grid[0,:].mean()), n_gt=int(n_gt), n_clips=len(clips))
    if return_grid: out["grid"] = grid
    return out

if __name__ == "__main__":
    gts = {f"{i:06d}": [(0.3,0.5,0.5,0.5,0.4,0.6)] for i in range(50)}
    fac = {k:(3.2,3.2) for k in gts}
    preds = {k:[(0.9,)+g[0]] for k,g in gts.items()}
    print("perfect(0.9):", evaluate(gts,preds,fac))
    preds1 = {k:[(1.0,)+g[0]] for k,g in gts.items()}
    print("perfect but score=1.0 (official: not counted):", evaluate(gts,preds1,fac))
    preds2 = {k:[(0.9,0.3,0.5,0.5+0.02,0.5,0.9,0.9)] for k in gts}   # rank0 error 0.02*3.2=0.064 m, rank1 far off
    print("rank0 0.064m, rank1 bad:", evaluate(gts,preds2,fac))
    preds4 = {k:[(0.9,0.3,0.5,0.5,0.5,0.4,0.6),(0.1,0.0,0.1,0.0,0.0,0.0,0.0)] for k in gts}
    print("extra low FP:", evaluate(gts,preds4,fac))
    gts5 = dict(gts); gts5.update({f"n{i}":[] for i in range(50)}); preds5 = dict(preds); preds5.update({f"n{i}":[(0.95,0.3,0.5,0.5,0.5,0.4,0.6)] for i in range(50)})
    print("FPs on no-GT clips (official: skipped):", evaluate(gts5,preds5,fac))
