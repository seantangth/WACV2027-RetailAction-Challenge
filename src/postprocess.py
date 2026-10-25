#!/usr/bin/env python3
"""RAC — post-processing (step 8): coverage expansion and two-tier confidence scores.

Scores (two_tier):
  * T = the top candidate of each of M clips (the clips chosen by tsel.py, or the M clips with the highest score): score = 0.995 - 1e-6 * rank, all within [0.99, 1.0).
  * All other candidates: score = 0.001 + 0.008 * F(s), with F the empirical distribution function of all input scores s (order preserved), i.e. within [0.001, 0.009].
Coverage expansion (augment2 / augment3): lattice points, start/end variants and sliding windows with per-frame heat-map peaks, added as further candidates.
T refinement (refine_T): the point of each T candidate is moved to the aggregated per-frame heat-map peak.

Subcommands:
  eval   (test, with GT): mAP and per-cell coverage for the original scores, two-tier (several M) and coverage expansion
  apply  (challenge): write the submission file (two-tier, optionally with coverage expansion and T refinement)
Usage:
  python postprocess.py eval  --meta_dir M --split test --csv test_oof.csv [--Ms 1,3,5,10,20,50,100,200] [--aug ring10,ring20,tvar]
  python postprocess.py apply --meta_dir M --split challenge --csv sub.csv --out sub_tt.csv --M 20 [--aug ...] --sample_sub S
"""
import argparse, csv, json, math, pickle, os, bisect, collections, time, numpy as np
from metric import evaluate, gts_from_meta, factors_from_meta, TAU_S, TAU_T, EPS, M_PX_FACTOR_AVG

W = 600.0

def load_csv(path):
    out = {}
    for r in csv.DictReader(open(path)):
        acts = json.loads(r["prediction"])["actions"]
        out[r["id"]] = [(float(a["score"]), float(a["start"]), float(a["end"]),
                         float(a["spatial"]["action_cam"]["rank0"]["x"]), float(a["spatial"]["action_cam"]["rank0"]["y"]),
                         float(a["spatial"]["action_cam"]["rank1"]["x"]), float(a["spatial"]["action_cam"]["rank1"]["y"])) for a in acts]
    return out

def write_csv(preds, path, ids):
    """Compact JSON: scores and times to 6 decimals, coordinates to 5, to keep the file small."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL); w.writerow(["id", "prediction"])
        for sid in ids:
            acts = []
            for (sc, s, e, x0, y0, x1, y1) in preds.get(sid, []):
                s = min(max(s, 0.0), 1.0); e = min(max(e, 0.0), 1.0)
                if e <= s: e = min(1.0, s + 1e-3)
                acts.append({"score": round(float(sc), 6), "start": round(s, 6), "end": round(e, 6),
                             "spatial": {"action_cam": {"rank0": {"x": round(float(np.clip(x0, 0, 1)), 5), "y": round(float(np.clip(y0, 0, 1)), 5)},
                                                        "rank1": {"x": round(float(np.clip(x1, 0, 1)), 5), "y": round(float(np.clip(y1, 0, 1)), 5)}}}})
            w.writerow([sid, json.dumps({"actions": acts}, separators=(",", ":"))])

def grid_N_from_meta(recs):
    out = {}
    for r in recs:
        n = max(r["views"][0]["n_video"], r["views"][1]["n_video"])
        if n <= 0: n = max(len(r["views"][0]["poses"]), len(r["views"][1]["poses"]))
        out[r["id"]] = max(int(n), 1)
    return out

# ---------------- coverage expansion ----------------
def ring(radius_px, n):
    return [(radius_px * math.cos(2 * math.pi * k / n) / W, radius_px * math.sin(2 * math.pi * k / n) / W) for k in range(n)]

def augment(preds, Ns, spec, topk=2):
    """spec: comma-separated. Adds variants of the main candidates of the top-k distinct segments of each clip (in the bulk; T is unchanged).
       ringR[:n]  : ring of n points (default 6) at radius R px, for view 0 (view 1 fixed) and for view 1 (view 0 fixed)
       tvar       : start/end +-1 frame (including both), 8 variants
       tvar2      : additionally +-2 frames (one side), 4 variants
       cross      : tvar x ring (top-1 segment only; uses the first ring in spec)
    Returns new preds (original candidates first, new variants after; scores = original score x a decreasing factor)."""
    if not spec: return preds
    items = [x.strip() for x in spec.split(",") if x.strip()]
    rings = []
    for it in items:
        if it.startswith("ring"):
            rr = it[4:]; n = 6
            if ":" in rr: rr, n = rr.split(":"); n = int(n)
            rings.append((float(rr), n))
    out = {}
    for sid, cs in preds.items():
        N = Ns.get(sid, 32); dN = 1.0 / N
        cs_sorted = sorted(cs, key=lambda c: -c[0]); new = list(cs_sorted)
        seen_seg = []; picked = []
        for c in cs_sorted:
            key = (round(c[1] * N), round(c[2] * N))
            if key in seen_seg: continue
            seen_seg.append(key); picked.append(c)
            if len(picked) >= topk: break
        for ri, c in enumerate(picked):
            sc, s, e, x0, y0, x1, y1 = c; base_sc = sc * 0.5
            tvars = []
            if "tvar" in items or "tvar2" in items or "cross" in items:
                for ds_, de_ in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1), (-1, -1), (1, 1)]:
                    s2, e2 = s + ds_ * dN, e + de_ * dN
                    if 0.0 <= s2 < e2 <= 1.0 + 1e-9: tvars.append((s2, min(e2, 1.0)))
            if "tvar2" in items:
                for ds_, de_ in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                    s2, e2 = s + ds_ * dN, e + de_ * dN
                    if 0.0 <= s2 < e2 <= 1.0 + 1e-9: tvars.append((s2, min(e2, 1.0)))
            if "tvar" in items or "tvar2" in items:
                for (s2, e2) in tvars: new.append((base_sc * 0.9, s2, e2, x0, y0, x1, y1))
            for (R, n) in rings:
                for (dx, dy) in ring(R, n):
                    new.append((base_sc * 0.8, s, e, x0 + dx, y0 + dy, x1, y1))
                    new.append((base_sc * 0.8, s, e, x0, y0, x1 + dx, y1 + dy))
            if "cross" in items and ri == 0 and rings:
                R, n = rings[0]
                for (s2, e2) in tvars[:4]:
                    for (dx, dy) in ring(R, n):
                        new.append((base_sc * 0.6, s2, e2, x0 + dx, y0 + dy, x1, y1))
                        new.append((base_sc * 0.6, s2, e2, x0, y0, x1 + dx, y1 + dy))
        out[sid] = new
    return out

# ---------------- coverage expansion v2 (designed from the coverage gaps) ----------------
def hex_offsets(rings):
    """Hexagonal lattice with unit spacing (centre + `rings` rings); returns [(dx, dy)] in units of the spacing."""
    pts = [(0.0, 0.0)]
    dirs = [(1, 0), (0.5, math.sqrt(3) / 2), (-0.5, math.sqrt(3) / 2), (-1, 0), (-0.5, -math.sqrt(3) / 2), (0.5, -math.sqrt(3) / 2)]
    for k in range(1, rings + 1):
        x, y = -k * dirs[4][0], -k * dirs[4][1]   # start at one corner and walk along the six sides
        x, y = k * dirs[4][0], k * dirs[4][1]
        for d in range(6):
            for _ in range(k):
                pts.append((x, y)); x += dirs[d][0]; y += dirs[d][1]
    return pts

def augment2(preds, Ns, facs, cfg):
    """cfg (dict):
       k_lat: number of top segments that get the hexagonal lattice; lat_rings: rings; lat_rings_top1: rings for the top segment (default lat_rings); lat_c: spacing = lat_c x the 5 cm pixel tolerance (per clip and view, from m_px_factor)
       k_tv: number of top segments that get boundary variants; tv_r: boundary shift radius (frames)
       cross_r: +-cross_r boundary variants of the top-1 segment x the first lattice ring; win: sliding-window lengths (frames), win_stride; k_seg: segments used as reference for new segments
    Lattice points of the two views are paired (i-th with i-th): the metric uses the closer view, so pairing loses no coverage and halves the candidate count."""
    hexo = hex_offsets(cfg.get("lat_rings", 2)); hex1 = hex_offsets(1); hexo_top = hex_offsets(cfg.get("lat_rings_top1", cfg.get("lat_rings", 2)))
    out = {}
    for sid, cs in preds.items():
        N = Ns.get(sid, 32); dN = 1.0 / N; f0, f1 = facs.get(sid, (M_PX_FACTOR_AVG, M_PX_FACTOR_AVG))
        tol0 = 0.05 / f0; tol1 = 0.05 / f1                      # 5 cm radius in normalised coordinates
        sp0 = cfg.get("lat_c", 1.6) * tol0; sp1 = cfg.get("lat_c", 1.6) * tol1
        cs_sorted = sorted(cs, key=lambda c: -c[0]); new = list(cs_sorted); have = set()
        for c in cs_sorted: have.add((round(c[1] * N), round(c[2] * N), round(c[3], 4), round(c[4], 4), round(c[5], 4), round(c[6], 4)))
        def add(sc, s, e, x0, y0, x1, y1):
            s = round(s * N) / N; e = round(e * N) / N        # snap to the frame grid to avoid zero-length segments from rounding
            if not (0.0 <= s and e - s >= 0.5 / N and e <= 1.0 + 1e-9): return
            key = (round(s * N), round(e * N), round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4))
            if key in have: return
            have.add(key); new.append((sc, s, min(e, 1.0), x0, y0, x1, y1))
        segs = []; seen = set()
        for c in cs_sorted:
            k = (round(c[1] * N), round(c[2] * N))
            if k in seen: continue
            seen.add(k); segs.append(c)
        for ri, c in enumerate(segs):
            sc, s, e, x0, y0, x1, y1 = c; b = sc * 0.5
            if ri < cfg.get("k_lat", 2):
                for (u, v) in (hexo_top if ri == 0 else hexo)[1:]:
                    add(b * 0.8, s, e, x0 + u * sp0, y0 + v * sp0, x1 + u * sp1, y1 + v * sp1)
            if ri < cfg.get("k_tv", 3):
                R = cfg.get("tv_r", 1)
                for ds_ in range(-R, R + 1):
                    for de_ in range(-R, R + 1):
                        if ds_ == 0 and de_ == 0: continue
                        add(b * 0.7, s + ds_ * dN, e + de_ * dN, x0, y0, x1, y1)
            if ri == 0 and cfg.get("cross_r", 0) > 0:
                R = cfg["cross_r"]
                for ds_ in range(-R, R + 1):
                    for de_ in range(-R, R + 1):
                        if ds_ == 0 and de_ == 0: continue
                        for (u, v) in hex1[1:]:
                            add(b * 0.5, s + ds_ * dN, e + de_ * dN, x0 + u * sp0, y0 + v * sp0, x1 + u * sp1, y1 + v * sp1)
        if cfg.get("win"):
            ref = segs[:cfg.get("k_seg", 3)]
            mids = [((c[1] + c[2]) / 2, c) for c in ref]
            for L in cfg["win"]:
                for i in range(0, max(N - L, 0) + 1, cfg.get("win_stride", 2)):
                    s, e = i * dN, min(N, i + L) * dN
                    if any((min(e, c[2]) - max(s, c[1])) / (max(e, c[2]) - min(s, c[1]) + EPS) >= 0.5 for c in segs[:6]): continue
                    m = (s + e) / 2; c = min(mids, key=lambda t: abs(t[0] - m))[1]
                    add(c[0] * 0.05, s, e, c[3], c[4], c[5], c[6])
        out[sid] = new
    return out

# ---------------- coverage expansion v3: per-frame heat maps (frame_hm.py output) give each segment / sliding window its own point ----------------
_HH = 80; _CAN = 150

def load_fhm(d, sid):
    """d may list several comma-separated directories (e.g. per-frame heat maps of two spatial models): anchors come from the same temporal models, so offsets are identical and frames are averaged directly."""
    ds = [x for x in d.split(",") if x]; acc = None; n = 0
    for dd in ds:
        p = os.path.join(dd, f"{sid}.npz")
        if not os.path.exists(p): continue
        z = np.load(p); cur = [[z["hm0"].astype(np.float32), z["off0"].astype(np.int32)], [z["hm1"].astype(np.float32), z["off1"].astype(np.int32)]]
        if acc is None: acc = cur; n = 1; continue
        ok = all(acc[v][0].shape == cur[v][0].shape and np.array_equal(acc[v][1], cur[v][1]) for v in (0, 1))
        if not ok: continue
        for v in (0, 1): acc[v][0] += cur[v][0]
        n += 1
    if acc is None: return None
    return [(acc[v][0] / n, acc[v][1]) for v in (0, 1)]

def agg_peaks(view, fis, k=2, nms=3):
    """Paste the heat maps of frames fis onto the full-frame stride-4 canvas (padded by 80 cells on each side) and average -> top-k local maxima inside the image (NMS) -> soft-argmax refinement -> normalised coordinates. Cell c <-> pixel 4c."""
    hm, off = view; fis = [f for f in fis if 0 <= f < len(hm)]
    if not fis: return []
    can = np.zeros((_CAN + 2 * _HH, _CAN + 2 * _HH), np.float32)
    for f in fis:
        ox, oy = int(off[f, 0]) + _HH, int(off[f, 1]) + _HH
        can[oy:oy + _HH, ox:ox + _HH] += hm[f]
    can /= len(fis); inner = can[_HH:_HH + _CAN, _HH:_HH + _CAN].copy(); out = []
    for _ in range(k):
        idx = int(np.argmax(inner)); py, px = divmod(idx, _CAN); pk = float(inner[py, px])
        if pk <= 1e-4: break
        y0, y1 = max(0, py - 1), min(_CAN, py + 2); x0, x1 = max(0, px - 1), min(_CAN, px + 2)
        sub = inner[y0:y1, x0:x1]; ys, xs = np.mgrid[y0:y1, x0:x1]; w = sub.sum() + 1e-9
        cx, cy = float((xs * sub).sum() / w), float((ys * sub).sum() / w)
        out.append((min(max(4 * cx / W, 0.0), 1.0), min(max(4 * cy / W, 0.0), 1.0), pk))
        inner[max(0, py - nms):py + nms + 1, max(0, px - nms):px + nms + 1] = 0
    return out

def frame_peak(view, f):
    r = agg_peaks(view, [f], k=1); return r[0] if r else None

def augment3(preds, Ns, facs, cfg, fhm_dir):
    """On top of augment2 (applies the augment2 cfg first), adds:
       fhm_seg K: for the top K segments, the top 2 peaks of the aggregated heat map (views paired by rank); fhm_lat: one hexagonal ring around the first peak
       fhm_frames K: for the top K segments, each frame's own peak (views paired by frame)
       fhm_win: sliding windows use the aggregated in-window heat-map peak as their point (instead of the nearest segment's point); fhm_win_k: peaks per window (default 1); fhm_win_nms: NMS radius between peaks (heat-map cells, default 3 = 12 px); fhm_seg_k: peaks per segment for fhm_seg (default 2); win_lat: one lattice ring around window points"""
    cfg2 = dict(cfg); win = cfg2.pop("win", None)
    base = augment2(preds, Ns, facs, cfg2)
    hex1 = hex_offsets(1); out = {}; miss = 0
    for sid, cs in base.items():
        N = Ns.get(sid, 32); dN = 1.0 / N; f0, f1 = facs.get(sid, (M_PX_FACTOR_AVG, M_PX_FACTOR_AVG))
        sp0 = cfg.get("lat_c", 1.6) * 0.05 / f0; sp1 = cfg.get("lat_c", 1.6) * 0.05 / f1
        F = load_fhm(fhm_dir, sid); new = list(cs)
        have = {(round(c[1] * N), round(c[2] * N), round(c[3], 4), round(c[4], 4), round(c[5], 4), round(c[6], 4)) for c in cs}
        def add(sc, s, e, x0, y0, x1, y1):
            s = round(s * N) / N; e = round(e * N) / N
            if not (0.0 <= s and e - s >= 0.5 / N and e <= 1.0 + 1e-9): return
            key = (round(s * N), round(e * N), round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4))
            if key in have: return
            have.add(key); new.append((sc, s, min(e, 1.0), x0, y0, x1, y1))
        orig = sorted(preds[sid], key=lambda c: -c[0]); segs = []; seen = set()
        for c in orig:
            kk = (round(c[1] * N), round(c[2] * N))
            if kk not in seen: seen.add(kk); segs.append(c)
        if F is None:
            miss += 1; out[sid] = new; continue
        for ri, c in enumerate(segs):
            sc, s, e = c[0], c[1], c[2]; i0, i1 = int(round(s * N)), max(int(round(s * N)) + 1, int(round(e * N)))
            fis = list(range(i0, i1))
            if ri < cfg.get("fhm_seg", 0):
                ks = cfg.get("fhm_seg_k", 2); nm = cfg.get("fhm_win_nms", 3)
                p0 = agg_peaks(F[0], fis, k=ks, nms=nm); p1 = agg_peaks(F[1], fis, k=ks, nms=nm)
                for r in range(min(len(p0), len(p1))):
                    add(sc * 0.45, s, e, p0[r][0], p0[r][1], p1[r][0], p1[r][1])
                if cfg.get("fhm_lat") and p0 and p1:
                    for (u, v) in hex1[1:]: add(sc * 0.3, s, e, p0[0][0] + u * sp0, p0[0][1] + v * sp0, p1[0][0] + u * sp1, p1[0][1] + v * sp1)
            if ri < cfg.get("fhm_frames", 0):
                for f in fis:
                    a0 = frame_peak(F[0], f); a1 = frame_peak(F[1], f)
                    if a0 and a1: add(sc * 0.35, s, e, a0[0], a0[1], a1[0], a1[1])
        if win:
            wl = []   # (peak sum, s, e, p0, p1), used by win_lat_k to pick the strongest windows for lattice points
            for L in win:
                for i in range(0, max(N - L, 0) + 1, cfg.get("win_stride", 2)):
                    s, e = i * dN, min(N, i + L) * dN
                    if any((min(e, c[2]) - max(s, c[1])) / (max(e, c[2]) - min(s, c[1]) + EPS) >= 0.5 for c in segs[:6]): continue
                    fis = list(range(i, min(N, i + L)))
                    if cfg.get("fhm_win", 1):
                        kw = cfg.get("fhm_win_k", 1); nm = cfg.get("fhm_win_nms", 3)
                        p0 = agg_peaks(F[0], fis, k=kw, nms=nm); p1 = agg_peaks(F[1], fis, k=kw, nms=nm)
                        if p0 and p1:
                            add(0.02, s, e, p0[0][0], p0[0][1], p1[0][0], p1[0][1])
                            for r in range(1, max(len(p0), len(p1))):   # peaks 2..k (views paired by rank; a view with too few peaks reuses its first peak)
                                q0 = p0[r] if r < len(p0) else p0[0]; q1 = p1[r] if r < len(p1) else p1[0]
                                add(0.018, s, e, q0[0], q0[1], q1[0], q1[1])
                            if cfg.get("win_lat"):
                                for (u, v) in hex1[1:]: add(0.01, s, e, p0[0][0] + u * sp0, p0[0][1] + v * sp0, p1[0][0] + u * sp1, p1[0][1] + v * sp1)
                            wl.append((p0[0][2] + p1[0][2], s, e, p0[0], p1[0]))
                            continue
                    m = (s + e) / 2; c = min(segs[:3], key=lambda t: abs((t[1] + t[2]) / 2 - m)); add(0.015, s, e, c[3], c[4], c[5], c[6])
            if cfg.get("win_lat_k") and not cfg.get("win_lat"):
                for (pk, s, e, q0, q1) in sorted(wl, key=lambda t: -t[0])[:cfg["win_lat_k"]]:
                    for (u, v) in hex1[1:]: add(0.012, s, e, q0[0] + u * sp0, q0[1] + v * sp0, q1[0] + u * sp1, q1[1] + v * sp1)
        out[sid] = new
    if miss: print(f"augment3: {miss} clips without frame heatmaps (fell back to augment2)", flush=True)
    return out

# ---------------- normal-plus: conventional ranking + low-confidence expansion tail (not two-tier) ----------------
def normal_plus(preds, n_orig, f, cap=0):
    """Original candidates (the first n_orig per clip; augment2/3 keep them first) keep their scores; expansion candidates are scored x f (small f -> ranked after almost all original candidates).
       cap > 0: keep at most cap candidates per clip (all originals, then expansions by score). The ordering is meaningful throughout: a conventional confidence ranking."""
    out = {}
    for sid, cs in preds.items():
        k = n_orig.get(sid, len(cs)); org = list(cs[:k]); aug = sorted(((c[0] * f,) + tuple(c[1:]) for c in cs[k:]), key=lambda c: -c[0])
        if cap > 0: aug = aug[:max(cap - k, 0)]
        out[sid] = org + aug
    return out

# ---------------- two-tier ----------------
def two_tier(preds, M, top_score=0.995, bulk_lo=0.001, bulk_hi=0.009, sel=None, T_ids=None):
    """T = top candidate of each of the first M clips ranked by sel (default: original score); T scores start at 0.995 and decrease by 1e-6 per rank (all within [0.99, 1.0)).
       All other candidates are mapped linearly to [bulk_lo, bulk_hi], preserving their global order (which fixes the greedy matching order within a clip)."""
    best = []
    for sid, cs in preds.items():
        if not cs: continue
        j = int(np.argmax([c[0] for c in cs])); s = sel[sid] if sel is not None else cs[j][0]
        best.append((s, sid, j))
    best.sort(reverse=True)
    if T_ids is not None:   # clips given for T (e.g. chosen by tsel.py by hit probability at the tightest cell), in the given order
        order = {sid: r for r, sid in enumerate(T_ids)}; best = sorted([b for b in best if b[1] in order], key=lambda b: order[b[1]]) + [b for b in best if b[1] not in order]
        M = len(order)
    T = {sid: j for _, sid, j in best[:M]}
    allsc = sorted(c[0] for cs in preds.values() for c in cs); n = max(len(allsc), 1)
    out = {}; rank_T = {sid: r for r, (_, sid, _) in enumerate(best[:M])}
    for sid, cs in preds.items():
        new = []
        for j, c in enumerate(cs):
            if T.get(sid, -1) == j: sc = top_score - 1e-6 * rank_T[sid]
            else: sc = bulk_lo + (bulk_hi - bulk_lo) * bisect.bisect_right(allsc, c[0]) / n
            new.append((sc,) + tuple(c[1:]))
        out[sid] = new
    return out, set(T)

def refine_T(tt, Tset, Ns, fhm_dir, bulk_hi=0.009):
    """Replace the point of each T candidate (score >= 0.99) by the first peak of the per-frame heat maps aggregated over its segment (only if both views have a peak); the original candidate stays in the bulk at bulk_hi, so coverage is unchanged.
    This also removes the +2 px decoding bias of fuse_predict, which the per-frame heat maps do not have."""
    out = {}; n_ref = 0; moved = []
    for sid, cs in tt.items():
        if sid not in Tset: out[sid] = cs; continue
        j = max(range(len(cs)), key=lambda k: cs[k][0]); c = cs[j]
        F = load_fhm(fhm_dir, sid) if fhm_dir else None
        if F is None or c[0] < 0.99: out[sid] = cs; continue
        N = Ns.get(sid, 32); i0 = int(round(c[1] * N)); i1 = max(i0 + 1, int(round(c[2] * N))); fis = list(range(i0, i1))
        p0 = agg_peaks(F[0], fis, k=1); p1 = agg_peaks(F[1], fis, k=1)
        if not (p0 and p1): out[sid] = cs; continue
        new = list(cs); new[j] = (c[0], c[1], c[2], p0[0][0], p0[0][1], p1[0][0], p1[0][1]); new.append((bulk_hi,) + tuple(c[1:]))
        moved.append(max(math.hypot(p0[0][0] - c[3], p0[0][1] - c[4]), math.hypot(p1[0][0] - c[5], p1[0][1] - c[6])) * W)
        out[sid] = new; n_ref += 1
    if moved: print(f"refine_T: {n_ref}/{len(Tset)} T points moved to frame-heatmap peaks (max-view shift px: median {np.median(moved):.1f}, p90 {np.percentile(moved, 90):.1f})", flush=True)
    return out

# ---------------- fast coverage (union of cells covered for each GT by any candidate) ----------------
def coverage(gts, preds, fac):
    cov = []; per_clip = {}
    for sid, g in gts.items():
        if not g: continue
        cs = preds.get(sid, []); f0, f1 = fac.get(sid, (M_PX_FACTOR_AVG, M_PX_FACTOR_AVG))
        if not cs:
            for _ in g: cov.append(np.zeros((len(TAU_S), len(TAU_T)), bool))
            continue
        P = np.array([c[1:] for c in cs], float)
        for (gs, ge, gx0, gy0, gx1, gy1) in g:
            inter = np.minimum(P[:, 1], ge) - np.maximum(P[:, 0], gs); union = np.maximum(P[:, 1], ge) - np.minimum(P[:, 0], gs)
            T = np.maximum(0.0, inter / (union + EPS))
            d0 = np.hypot(P[:, 2] - gx0, P[:, 3] - gy0) * f0; d1 = np.hypot(P[:, 4] - gx1, P[:, 5] - gy1) * f1
            D = np.fmin(d0, d1)
            C = (TAU_S[None, :, None] >= D[:, None, None]) & (TAU_T[None, None, :] <= T[:, None, None])
            cov.append(C.any(0))
    cov = np.stack(cov)
    return cov.mean(0)   # [20,10] upper bound R_all

def prec_T_cells(gts, preds_tt, Tset, fac):
    """Per-cell precision of the T candidates (score >= 0.99) (official greedy matching: T is the first prediction of its clip, so TP = covers any GT)."""
    hits = []
    for sid in Tset:
        g = gts.get(sid, [])
        if not g: continue
        c = max(preds_tt[sid], key=lambda c: c[0]); f0, f1 = fac.get(sid, (M_PX_FACTOR_AVG, M_PX_FACTOR_AVG))
        best = np.zeros((len(TAU_S), len(TAU_T)), bool)
        for (gs, ge, gx0, gy0, gx1, gy1) in g:
            inter = min(c[2], ge) - max(c[1], gs); union = max(c[2], ge) - min(c[1], gs); T = max(0.0, inter / (union + EPS))
            D = min(math.hypot(c[3] - gx0, c[4] - gy0) * f0, math.hypot(c[5] - gx1, c[6] - gy1) * f1)
            best |= (TAU_S[:, None] >= D) & (TAU_T[None, :] <= T)
        hits.append(best)
    return np.stack(hits).mean(0) if hits else np.zeros((len(TAU_S), len(TAU_T)))

def summarize(grid):
    return f"mean {grid.mean():.4f} | 1m/t.05 {grid[0,0]:.3f} | 5cm col {grid[-1].mean():.3f} | 10cm col {grid[-2].mean():.3f} | tIoU.5 row {grid[:,-1].mean():.3f} | tightest {grid[-1,-1]:.3f}"

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("eval", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--meta_dir", required=True); p.add_argument("--split", default="test" if name == "eval" else "challenge")
        p.add_argument("--csv", required=True); p.add_argument("--aug", default=""); p.add_argument("--topk", type=int, default=2)
        p.add_argument("--bias", default="", help="systematic offset 'dx,dy' in pixels subtracted from all candidate points (the fuse_predict bias is about 0.1,1.94)")
        p.add_argument("--fhm", default="", help="frame_hm.py output directory (with <split>/<id>.npz); if given, augment3 is used (the --aug2 JSON may contain fhm_* keys)")
        p.add_argument("--aug2", default="", help='JSON settings, e.g. {"k_lat":2,"lat_rings":2,"k_tv":3,"tv_r":1,"cross_r":1,"win":[4,7,11]}')
        p.add_argument("--trefine", action="store_true", help="replace T points by the aggregated per-frame heat-map peak (needs --fhm); original candidates stay in the bulk")
        p.add_argument("--trefine_fhm", default="", help="per-frame heat-map directories for T refinement (comma-separated, averaged); default: same as --fhm")
        p.add_argument("--np_f", default="", help="normal-plus: score multiplier for expansion candidates (eval: comma-separated values; apply: a single value writes a normal-plus file instead of two-tier)")
        p.add_argument("--np_cap", type=int, default=0, help="normal-plus: maximum candidates per clip (0 = unlimited)")
        if name == "eval":
            p.add_argument("--Ms", default="1,3,5,10,20,50,100,200"); p.add_argument("--full_eval", action="store_true", help="run the full official-equivalent metric on two-tier (slower)")
            p.add_argument("--tsel_oof", default="", help="OOF probabilities of tsel on test as JSON ({clip: p}); if given, T is chosen by them (otherwise by rescorer score)")
        else:
            p.add_argument("--out", required=True); p.add_argument("--M", type=int, default=20); p.add_argument("--sample_sub", required=True)
            p.add_argument("--tsel", default="", help="JSON written by tsel.py apply ({clip_id: probability}); selects the T clips")
    a = ap.parse_args()
    recs = [r for r in pickle.load(open(os.path.join(a.meta_dir, f"meta_{a.split}.pkl"), "rb")) if "error" not in r]
    Ns = grid_N_from_meta(recs); preds = load_csv(a.csv)
    print(f"loaded {sum(len(v) for v in preds.values())} cands in {len(preds)} clips", flush=True)
    if a.bias:
        bx, by = [float(t) / W for t in a.bias.split(",")]
        preds = {k: [(c[0], c[1], c[2], c[3] - bx, c[4] - by, c[5] - bx, c[6] - by) for c in v] for k, v in preds.items()}
        print(f"bias-corrected by ({a.bias}) px", flush=True)
    n_orig = {k: len(v) for k, v in preds.items()}; preds_orig = preds
    if a.aug2:
        facs = factors_from_meta(recs)
        preds = augment3(preds, Ns, facs, json.loads(a.aug2), ",".join(os.path.join(x, a.split) for x in a.fhm.split(","))) if a.fhm else augment2(preds, Ns, facs, json.loads(a.aug2))
        print(f"after aug2 {a.aug2}: {sum(len(v) for v in preds.values())} cands ({sum(len(v) for v in preds.values())/max(len(preds),1):.1f}/clip)", flush=True)
    if a.aug:
        preds = augment(preds, Ns, a.aug, topk=a.topk)
        print(f"after aug '{a.aug}': {sum(len(v) for v in preds.values())} cands ({sum(len(v) for v in preds.values())/max(len(preds),1):.1f}/clip)", flush=True)
    if a.cmd == "eval":
        gts = gts_from_meta(recs); fac = factors_from_meta(recs)
        t0 = time.time(); base = evaluate(gts, preds, fac, return_grid=True)
        print(f"ORIGINAL scores : mAP {base['mAP']:.4f} | {summarize(base['grid'])} ({time.time()-t0:.0f}s)", flush=True)
        cov = coverage(gts, preds, fac); print(f"R_all (cover)   : {summarize(cov)}", flush=True)
        if a.np_f:
            r0 = evaluate(gts, preds_orig, fac, return_grid=True); print(f"NORMAL (originals only, {sum(n_orig.values())} cands): mAP {r0['mAP']:.4f} | {summarize(r0['grid'])}", flush=True)
            for f in [float(x) for x in a.np_f.split(",")]:
                npd = normal_plus(preds, n_orig, f, a.np_cap); r = evaluate(gts, npd, fac, return_grid=True)
                print(f"NORMAL-PLUS f={f:g} cap={a.np_cap} ({sum(len(v) for v in npd.values())/max(len(npd),1):.1f}/clip): mAP {r['mAP']:.4f} | {summarize(r['grid'])}", flush=True)
        T_order = None
        if a.tsel_oof:
            d = json.load(open(a.tsel_oof)); T_order = [k for k, _ in sorted(d.items(), key=lambda kv: -kv[1])]
        fdirs = ",".join(os.path.join(x, a.split) for x in a.fhm.split(",")) if a.fhm else ""
        for M in [int(x) for x in a.Ms.split(",")]:
            tt, Tset = two_tier(preds, M, T_ids=T_order[:M] if T_order else None)
            if a.trefine: tt = refine_T(tt, Tset, Ns, ",".join(os.path.join(x, a.split) for x in a.trefine_fhm.split(",")) if a.trefine_fhm else fdirs)
            pt = prec_T_cells(gts, tt, Tset, fac); approx = cov * pt
            line = f"two-tier M={M:<4d}{' T-refined' if a.trefine else ''}{' tsel' if T_order else ''}: approx {approx.mean():.4f} | prec_T mean {pt.mean():.4f} tightest {pt[-1,-1]:.3f} 5cm col {pt[-1].mean():.3f} 10cm col {pt[-2].mean():.3f} tIoU.5 row {pt[:,-1].mean():.3f} min {pt.min():.3f}"
            if a.full_eval:
                r = evaluate(gts, tt, fac, return_grid=True); line += f" || FULL mAP {r['mAP']:.4f} | {summarize(r['grid'])}"
            print(line, flush=True)
    else:
        if a.np_f:
            npd = normal_plus(preds, n_orig, float(a.np_f), a.np_cap); ids = [r["id"] for r in csv.DictReader(open(a.sample_sub))]
            write_csv(npd, a.out, ids); n = sum(len(v) for v in npd.values())
            print(f"wrote {a.out}: normal-plus f={a.np_f} cap={a.np_cap}, {n} cands ({n/max(len(npd),1):.1f}/clip), max score {max(c[0] for v in npd.values() for c in v):.4f}, size {os.path.getsize(a.out)/1e6:.1f} MB", flush=True)
            return
        T_ids = None
        if a.tsel:
            d = json.load(open(a.tsel)); T_ids = [k for k, _ in sorted(d.items(), key=lambda kv: -kv[1])]
        tt, Tset = two_tier(preds, a.M, T_ids=T_ids)
        if a.trefine:
            assert a.fhm, "--trefine requires --fhm"
            tt = refine_T(tt, Tset, Ns, ",".join(os.path.join(x, a.split) for x in (a.trefine_fhm or a.fhm).split(",")))
        ids = [r["id"] for r in csv.DictReader(open(a.sample_sub))]
        write_csv(tt, a.out, ids)
        n = sum(len(v) for v in tt.values()); ntop = sum(1 for v in tt.values() for c in v if c[0] >= 0.99)
        print(f"wrote {a.out}: {n} cands, T={ntop}, size {os.path.getsize(a.out)/1e6:.1f} MB", flush=True)

if __name__ == "__main__":
    main()
