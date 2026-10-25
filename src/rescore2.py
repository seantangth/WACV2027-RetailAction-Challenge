#!/usr/bin/env python3
"""RAC — second-stage rescoring: from the features of the fused candidates, predict each candidate's potential share of TP cells on the official 20x10 threshold grid and use it as the new score (followed by global rank normalisation).
Training data = candidates on the test split (with GT); GroupKFold by clip gives out-of-fold predictions; metric.evaluate compares BASE vs RESCORED.
Usage:
  python rescore2.py --meta_dir M --csv dump_s10_test.csv --feats dump_s10_test_feats.pkl \
      [--apply_csv dump_s10_challenge.csv --apply_feats dump_s10_challenge_feats.pkl --sample_sub sample_submission.csv --out sub_rescored.csv]"""
import argparse, csv, json, pickle, math, zlib, numpy as np, collections
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from metric import evaluate, gts_from_meta, factors_from_meta, TAU_S, TAU_T, EPS, M_PX_FACTOR_AVG
from baseline_meta import write_submission

B3F = ["b3_conf","b3_k0","b3_k1","b3_q0","b3_q1","b3_ps","b3_pe","b3_pmax_s","b3_pmax_e","b3_went","b3_dv0","b3_dv1","b3_refined"]
TF = ["s_st","e_en","a_bef","a_aft","a_in_min"]          # temporal boundary evidence per candidate (newer fuse_predict only)
XF = ["ov_top1","ov_max_hi","n_hi_ov50","dp_min_hi","dp_top1"]   # cross-candidate (marginal value) features: overlap and point distance to higher-scored candidates
FH = [f"fh_{k}{v}" for k in ("hv", "pk", "r2", "d", "sp") for v in (0, 1)] + ["fh_hvmin", "fh_hvmax", "fh_dmin"]   # per-frame heat-map features (fhfeat.py; only with --fh)
FEATS = ["score","tier","kind","k0","k1","q0","q1","d0","d1","start","end","dur","run_len","N","a_mean","a_max","a_out","has","n_cands",
         "rank_in_clip","score_rel","kmin","kmax","dmin","dmax","is_alt","n_same_seg","rank_in_seg"] + B3F + TF + XF

def load_csv(path):
    out = {}
    for r in csv.DictReader(open(path)):
        acts = json.loads(r["prediction"])["actions"]
        out[r["id"]] = [(a["score"], a["start"], a["end"], a["spatial"]["action_cam"]["rank0"]["x"], a["spatial"]["action_cam"]["rank0"]["y"],
                         a["spatial"]["action_cam"]["rank1"]["x"], a["spatial"]["action_cam"]["rank1"]["y"]) for a in acts]
    return out

def _mism(cands, fl):
    return sum(1 for c, f in zip(cands, fl) if abs(f["start"]-c[1]) + abs(f["end"]-c[2]) + abs(f["x0"]-c[3]) + abs(f["y0"]-c[4]) + abs(f["x1"]-c[5]) + abs(f["y1"]-c[6]) > 1e-3)

def align(preds, feats):
    """Align candidates (CSV order = fuse's out+extra) with feature rows. Older fuse_predict wrote feats in append order (main candidates interleaved with their alternatives),
    which differs from the CSV order -> on a position mismatch, reorder as kind 0/1 first, others after (= out+extra order) and report remaining mismatches."""
    rows = []; n_geo = 0; n_miss = 0; n_reord = 0; n_bad = 0
    for sid, cands in preds.items():
        fl = feats.get(sid, [])
        if len(fl) == len(cands):
            m0 = _mism(cands, fl)
            if m0 > 0:
                fl2 = [f for f in fl if f.get("kind", 1) in (0, 1)] + [f for f in fl if f.get("kind", 1) not in (0, 1)]
                m2 = _mism(cands, fl2)
                if m2 < m0: fl = fl2; n_reord += 1; n_bad += m2
                else: n_bad += m0
            for i, f in enumerate(fl): rows.append((sid, i, f))
            continue
        used = set()
        for i, c in enumerate(cands):
            best, bd = None, 1e9
            for j, f in enumerate(fl):
                if j in used: continue
                d = abs(f["start"]-c[1]) + abs(f["end"]-c[2]) + abs(f["x0"]-c[3]) + abs(f["y0"]-c[4]) + abs(f["x1"]-c[5]) + abs(f["y1"]-c[6])
                if d < bd: bd, best = d, j
            if best is not None and bd < 0.05: used.add(best); rows.append((sid, i, fl[best])); n_geo += 1
            else: rows.append((sid, i, None)); n_miss += 1
    print(f"align: reordered clips {n_reord}, still-mismatched rows {n_bad}, geometric {n_geo}, unmatched {n_miss}", flush=True)
    return rows

def featurize(preds, rows, feats=None):
    feats = feats or FEATS
    X = np.full((len(rows), len(feats)), np.nan, dtype=np.float32)
    # per-clip rank and same-segment statistics
    by_clip = collections.defaultdict(list)
    for k, (sid, i, f) in enumerate(rows): by_clip[sid].append(k)
    for sid, ks in by_clip.items():
        sc = np.array([preds[sid][rows[k][1]][0] for k in ks]); order = np.argsort(-sc); rank = np.empty(len(ks)); rank[order] = np.arange(len(ks))
        segs = collections.defaultdict(list)
        for k in ks: c = preds[sid][rows[k][1]]; segs[(round(c[1],4), round(c[2],4))].append(k)
        # cross-candidate features: in descending score order, look at the higher-scored candidates of each candidate
        cl = [preds[sid][rows[k][1]] for k in ks]; order_hi = [ks[j] for j in order]
        xf = {}
        for r_, k in enumerate(order_hi):
            c = preds[sid][rows[k][1]]; ov_top1 = 0.0; ov_max = 0.0; n50 = 0; dp_min = 1.0; dp_top1 = 1.0
            for r2 in range(r_):
                c2 = preds[sid][rows[order_hi[r2]][1]]
                inter = min(c[2], c2[2]) - max(c[1], c2[1]); union = max(c[2], c2[2]) - min(c[1], c2[1]); t = max(0.0, inter / (union + 1e-9))
                dp = 0.5 * (math.hypot(c[3]-c2[3], c[4]-c2[4]) + math.hypot(c[5]-c2[5], c[6]-c2[6]))
                if r2 == 0: ov_top1 = t; dp_top1 = dp
                ov_max = max(ov_max, t)
                if t >= 0.5: n50 += 1; dp_min = min(dp_min, dp)
            xf[k] = dict(ov_top1=ov_top1, ov_max_hi=ov_max, n_hi_ov50=n50, dp_min_hi=dp_min, dp_top1=dp_top1)
        for pos, k in enumerate(ks):
            sid_, i, f = rows[k]; c = preds[sid][i]
            seg = segs[(round(c[1],4), round(c[2],4))]; seg_sc = sorted([preds[sid][rows[j][1]][0] for j in seg], reverse=True)
            base = dict(score=c[0], start=c[1], end=c[2], dur=c[2]-c[1], rank_in_clip=rank[pos], score_rel=c[0]/(sc.max()+1e-9),
                        n_same_seg=len(seg), rank_in_seg=seg_sc.index(c[0]), **xf[k])
            if f is not None:
                base.update(tier=f["tier"], kind=f["kind"], k0=f["k0"], k1=f["k1"], q0=f["q0"], q1=f["q1"], d0=f["d0"], d1=f["d1"], run_len=f["run_len"], N=f["N"],
                            a_mean=f["a_mean"], a_max=f["a_max"], a_out=f["a_out"], has=f["has"], n_cands=f["n_cands"],
                            kmin=min(f["k0"],f["k1"]), kmax=max(f["k0"],f["k1"]), dmin=min(f["d0"],f["d1"]), dmax=max(f["d0"],f["d1"]), is_alt=float(f["tier"]==2))
                base.update({k_: f[k_] for k_ in B3F + TF + FH if k_ in f})
            for j, name in enumerate(feats):
                if name in base: X[k, j] = base[name]
    return X

def marginal(preds, rows, gts, fac):
    """Marginal-value target: for each GT, candidates are sorted by (number of covered grid cells, original score), descending, and each is credited only with the cells not already covered by better candidates (so alternatives of the same GT are not all labelled perfect).
    Candidate target = maximum marginal coverage share over all GTs."""
    y = np.zeros(len(rows), np.float32); by_clip = collections.defaultdict(list)
    for k, (sid, i, f) in enumerate(rows): by_clip[sid].append(k)
    for sid, ks in by_clip.items():
        g = gts.get(sid, [])
        if not g: continue
        f0, f1 = fac.get(sid, (M_PX_FACTOR_AVG, M_PX_FACTOR_AVG))
        for (gs, ge, gx0, gy0, gx1, gy1) in g:
            cov = {}
            for k in ks:
                sc, ps, pe, px0, py0, px1, py1 = preds[sid][rows[k][1]]
                inter = min(pe, ge) - max(ps, gs); union = max(pe, ge) - min(ps, gs); T = max(0.0, inter / (union + EPS))
                d0 = math.hypot(px0-gx0, py0-gy0)*f0; d1 = math.hypot(px1-gx1, py1-gy1)*f1; ds = [d for d in (d0, d1) if math.isfinite(d)]; Dm = min(ds) if ds else np.inf
                C = (TAU_S >= Dm)[:, None] & (TAU_T <= T)[None, :]
                if C.any(): cov[k] = C
            covered = np.zeros((len(TAU_S), len(TAU_T)), bool)
            for k in sorted(cov, key=lambda k: (-int(cov[k].sum()), -preds[sid][rows[k][1]][0])):
                gain = float((cov[k] & ~covered).sum()) / covered.size; covered |= cov[k]
                y[k] = max(y[k], gain)
    return y

def suppress(preds, rows, s, factor, tiou_thr, dist_thr):
    """Duplicate suppression within a clip (soft-NMS): in descending score order, a candidate with tIoU >= tiou_thr and point distance <= dist_thr in both views to a kept higher-scored candidate has its score multiplied by factor."""
    by = collections.defaultdict(list)
    for k, (sid, i, f) in enumerate(rows): by[sid].append(k)
    s = s.copy()
    for sid, ks in by.items():
        kept = []
        for k in sorted(ks, key=lambda k: -s[k]):
            c = preds[sid][rows[k][1]]; dup = False
            for k2 in kept:
                c2 = preds[sid][rows[k2][1]]
                inter = min(c[2], c2[2]) - max(c[1], c2[1]); union = max(c[2], c2[2]) - min(c[1], c2[1]); t = inter / (union + 1e-9) if inter > 0 else 0.0
                if t >= tiou_thr and max(math.hypot(c[3]-c2[3], c[4]-c2[4]), math.hypot(c[5]-c2[5], c[6]-c2[6])) <= dist_thr: dup = True; break
            if dup: s[k] *= factor
            else: kept.append(k)
    return s

def ranknorm_scores(vals, cap=0.99):
    order = np.argsort(np.argsort(vals)); return cap * (order + 1) / len(vals)

def rebuild(preds, rows, new_scores):
    out = collections.defaultdict(list)
    for k, (sid, i, f) in enumerate(rows):
        c = preds[sid][i]; out[sid].append((float(new_scores[k]),) + tuple(c[1:7]))
    return dict(out)

PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", required=True); ap.add_argument("--csv", required=True); ap.add_argument("--feats", required=True)
    ap.add_argument("--apply_csv", default=""); ap.add_argument("--apply_feats", default=""); ap.add_argument("--sample_sub", default=""); ap.add_argument("--out", default="")
    ap.add_argument("--rounds", type=int, default=600); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--save_model", default="", help="after training on all data, save the LightGBM model and mode as JSON + txt (for Docker inference)")
    ap.add_argument("--load_model", default="", help="load a saved model and apply it to --apply_csv (skips training and OOF; no test labels needed)")
    ap.add_argument("--mode", default="", help="fixed mode (an alpha value such as 1.0, or geo/base), overriding the OOF choice; used for Docker reproduction")
    ap.add_argument("--seed", type=int, default=0, help="LightGBM random seed (for measuring the OOF noise floor)"); ap.add_argument("--objective", default="regression", help="regression/tweedie/huber/binary (label = marginal value > 0, outputs a probability)/xentropy (marginal value as soft label)")
    ap.add_argument("--leaves", type=int, default=31); ap.add_argument("--lr", type=float, default=0.03); ap.add_argument("--grid", action="store_true", help="print the 5 cm column and the tIoU 0.5 row for BASE/RESCORED/ORACLE")
    ap.add_argument("--nms_factor", type=float, default=0.0, help="duplicate-suppression factor within a clip (0 = off)"); ap.add_argument("--nms_tiou", type=float, default=0.5); ap.add_argument("--nms_dist", type=float, default=0.03)
    ap.add_argument("--oof_out", default="", help="write the OOF-rescored (alpha=1.0) test candidates as CSV (for postprocess.py)")
    ap.add_argument("--fh", action="store_true", help="add per-frame heat-map features FH (feats must first be extended with fh_* by fhfeat.py)")
    ap.add_argument("--score_power", type=float, default=0.0, help=">0: final score = rescorer prediction raised to the power p (monotonic, ranking preserved; clipped at 0.99) instead of global rank normalisation")
    ap.add_argument("--score_maps", action="store_true", help="also compare monotonic score maps of the OOF predictions (rank-uniform vs raw prediction/powers) and the oracle under each map")
    a = ap.parse_args()
    if a.fh: FEATS.extend([k for k in FH if k not in FEATS]); print(f"FH features on: {len(FEATS)} features", flush=True)
    PARAMS.update(seed=a.seed, bagging_seed=a.seed, feature_fraction_seed=a.seed, objective=a.objective, num_leaves=a.leaves, learning_rate=a.lr)
    if a.load_model:
        import json as _json
        meta = _json.load(open(a.load_model + ".json")); m = lgb.Booster(model_file=a.load_model + ".txt"); mode = meta["mode"]
        p2 = load_csv(a.apply_csv); f2 = pickle.load(open(a.apply_feats, "rb")); r2 = align(p2, f2); X2 = featurize(p2, r2, meta["feats"]); pr = m.predict(X2)
        o2 = np.array([p2[sid][i][0] for (sid, i, f) in r2])
        if mode == "geo": s2 = ranknorm_scores(np.sqrt(np.clip(pr, 1e-6, None) * np.clip(o2, 1e-6, None)))
        elif mode == "base": s2 = ranknorm_scores(o2)
        else: s2 = ranknorm_scores(float(mode) * ranknorm_scores(pr) + (1 - float(mode)) * ranknorm_scores(o2))
        if a.score_power > 0:   # monotonic map: prediction to the power p (ties broken slightly by the original rank); same ranking as mode=1.0
            s2 = np.clip(np.clip(pr, 0, None) ** a.score_power + 1e-9 * ranknorm_scores(o2), 0, 0.99); print(f"score_power {a.score_power}: scores in [{s2.min():.2e}, {s2.max():.3f}]", flush=True)
        ids = [r["id"] for r in csv.DictReader(open(a.sample_sub))] if a.sample_sub else None
        write_submission(rebuild(p2, r2, s2), a.out, ids); print("wrote", a.out, "mode", mode, "cands", len(r2), "(loaded model)", flush=True); print("RESCORE2_DONE"); return
    recs = [r for r in pickle.load(open(f"{a.meta_dir}/meta_test.pkl", "rb")) if "error" not in r]
    gts = gts_from_meta(recs); fac = factors_from_meta(recs)
    preds = load_csv(a.csv); feats = pickle.load(open(a.feats, "rb")); rows = align(preds, feats)
    X = featurize(preds, rows); y = marginal(preds, rows, gts, fac)
    groups = np.array([zlib.crc32(r[0].encode()) % 100003 for r in rows])  # fixed hash: folds are reproducible across runs
    print(f"test candidates {len(rows)} in {len(preds)} clips | marginal target mean {y.mean():.3f}, >0: {(y>0).mean():.3f}, ==1: {(y>=0.999).mean():.3f}", flush=True)
    def gridline(res, tag):
        if not a.grid or "grid" not in res: return
        g = res["grid"]; np.set_printoptions(precision=3, suppress=True, linewidth=220)
        print(f"  [{tag}] spatial cols (1.0→0.05 m): {np.round(g.mean(1), 3)}  | temporal rows (0.05→0.5): {np.round(g.mean(0), 3)}  | 5cm={g[-1].mean():.3f} tIoU0.5={g[:, -1].mean():.3f}", flush=True)
    base = evaluate(gts, {sid: [tuple(c[:7]) for c in v] for sid, v in preds.items()}, fac, return_grid=a.grid); print("BASE     :", {k: round(v,4) for k, v in base.items() if isinstance(v, float)}, flush=True); gridline(base, "BASE")
    orc = evaluate(gts, rebuild(preds, rows, ranknorm_scores(y + 1e-6 * np.array([preds[sid][i][0] for (sid, i, f) in rows]))), fac, return_grid=a.grid)
    print("ORACLE   :", {k: round(v,4) for k, v in orc.items() if isinstance(v, float)}, "(target as score; upper bound of rescoring)", flush=True); gridline(orc, "ORACLE")
    ytrain = (y > 1e-6).astype(np.float32) if a.objective == "binary" else np.clip(y, 0, 1) if a.objective == "xentropy" else y
    oof = np.zeros(len(rows), np.float32); imp = np.zeros(len(FEATS))
    for fold, (tr, te) in enumerate(GroupKFold(a.folds).split(X, y, groups)):
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], ytrain[tr]), num_boost_round=a.rounds)
        oof[te] = m.predict(X[te]); imp += m.feature_importance("gain")
    print("feature gain:", sorted([(FEATS[j], int(imp[j])) for j in range(len(FEATS))], key=lambda t: -t[1])[:16], flush=True)
    print("feature present:", {FEATS[j]: float(np.isfinite(X[:, j]).mean()) for j in range(len(FEATS)) if np.isfinite(X[:, j]).mean() < 1.0}, flush=True)
    orig = np.array([preds[sid][i][0] for (sid, i, f) in rows])
    best = (base["mAP"], "base")
    if a.score_maps:
        # monotonic maps: keep the OOF ranking and change only the score distribution
        def tiebreak(v): return v + 1e-9 * ranknorm_scores(orig)
        maps = {"rank(uniform)": lambda v: ranknorm_scores(v), "raw": lambda v: np.clip(v, 0, 0.99),
                "raw^0.5": lambda v: np.clip(np.clip(v, 0, None) ** 0.5, 0, 0.99), "raw^2": lambda v: np.clip(np.clip(v, 0, None) ** 2, 0, 0.99),
                "raw/max": lambda v: 0.99 * np.clip(v, 0, None) / max(float(np.max(v)), 1e-9), "raw^1.5": lambda v: np.clip(np.clip(v, 0, None) ** 1.5, 0, 0.99), "raw^3": lambda v: np.clip(np.clip(v, 0, None) ** 3, 0, 0.99), "raw^4": lambda v: np.clip(np.clip(v, 0, None) ** 4, 0, 0.99)}
        for nm, fn in maps.items():
            r1 = evaluate(gts, rebuild(preds, rows, fn(tiebreak(oof))), fac, return_grid=True)
            r2 = evaluate(gts, rebuild(preds, rows, fn(tiebreak(y.astype(np.float64)))), fac, return_grid=True)
            g1, g2 = r1["grid"], r2["grid"]
            print(f"SCOREMAP {nm:14s} OOF mAP {r1['mAP']:.4f} (5cm {g1[-1].mean():.3f} tIoU.5 {g1[:, -1].mean():.3f} loosest {g1[0, 0]:.3f}) | oracle {r2['mAP']:.4f}", flush=True)
        q = np.quantile(np.clip(oof, 0, None), [0.5, 0.75, 0.9, 0.95, 0.99]); print("OOF prediction quantiles 50/75/90/95/99%:", np.round(q, 4).tolist(), flush=True)
    for alpha in (1.0, 0.8, 0.6, 0.4):
        s = alpha * ranknorm_scores(oof) + (1 - alpha) * ranknorm_scores(orig); s = ranknorm_scores(s)
        res = evaluate(gts, rebuild(preds, rows, s), fac, return_grid=(a.grid and alpha == 1.0))
        print(f"RESCORED alpha={alpha}: mAP {res['mAP']:.4f} mAPs {res['mAPs']:.4f} mAPt {res['mAPt']:.4f} (OOF)", flush=True)
        if alpha == 1.0: gridline(res, "RESCORED")
        if alpha == 1.0 and a.score_power > 0:
            sp = np.clip(np.clip(oof, 0, None) ** a.score_power + 1e-9 * ranknorm_scores(orig), 0, 0.99); rp = evaluate(gts, rebuild(preds, rows, sp), fac)
            print(f"RESCORED score_power={a.score_power}: mAP {rp['mAP']:.4f} mAPs {rp['mAPs']:.4f} mAPt {rp['mAPt']:.4f} (OOF)", flush=True)
            if a.oof_out:
                fp = a.oof_out.replace(".csv", f"_p{a.score_power:g}.csv"); write_submission(rebuild(preds, rows, sp), fp, sorted(preds)); print("wrote OOF csv (score_power)", fp, flush=True)
        if alpha == 1.0 and a.oof_out:
            write_submission(rebuild(preds, rows, s), a.oof_out, sorted(preds)); print("wrote OOF csv", a.oof_out, flush=True)
        if alpha == 1.0 and a.nms_factor > 0:
            s2 = ranknorm_scores(suppress(preds, rows, s, a.nms_factor, a.nms_tiou, a.nms_dist)); res2 = evaluate(gts, rebuild(preds, rows, s2), fac)
            print(f"RESCORED+NMS f={a.nms_factor} t={a.nms_tiou} d={a.nms_dist}: mAP {res2['mAP']:.4f} mAPs {res2['mAPs']:.4f} mAPt {res2['mAPt']:.4f} (OOF)", flush=True)
            if res2["mAP"] > best[0]: best = (res2["mAP"], f"nms{a.nms_factor}")
        if res["mAP"] > best[0]: best = (res["mAP"], alpha)
    # also try: multiplicative OOF fusion (geometric mean)
    s = ranknorm_scores(np.sqrt(np.clip(oof, 1e-6, None) * np.clip(orig, 1e-6, None)))
    res = evaluate(gts, rebuild(preds, rows, s), fac); print(f"RESCORED geo-mean: mAP {res['mAP']:.4f} mAPs {res['mAPs']:.4f} mAPt {res['mAPt']:.4f} (OOF)", flush=True)
    if res["mAP"] > best[0]: best = (res["mAP"], "geo")
    if a.mode: best = (best[0], a.mode if a.mode in ("geo", "base") else float(a.mode)); print("MODE fixed by --mode:", best[1], flush=True)
    print("BEST:", best, flush=True)
    if a.save_model or (a.apply_csv and a.out):
        m = lgb.train(PARAMS, lgb.Dataset(X, ytrain), num_boost_round=a.rounds)
        if a.save_model:
            import json as _json
            m.save_model(a.save_model + ".txt"); _json.dump({"mode": best[1], "feats": FEATS, "oof_mAP": best[0]}, open(a.save_model + ".json", "w")); print("saved model", a.save_model, "mode", best[1], flush=True)
    if a.apply_csv and a.out:
        p2 = load_csv(a.apply_csv); f2 = pickle.load(open(a.apply_feats, "rb")); r2 = align(p2, f2); X2 = featurize(p2, r2); pr = m.predict(X2)
        o2 = np.array([p2[sid][i][0] for (sid, i, f) in r2]); mode = best[1]
        if mode == "geo": s2 = ranknorm_scores(np.sqrt(np.clip(pr, 1e-6, None) * np.clip(o2, 1e-6, None)))
        elif mode == "base": s2 = ranknorm_scores(o2)
        else: s2 = ranknorm_scores(mode * ranknorm_scores(pr) + (1 - mode) * ranknorm_scores(o2))
        ids = [r["id"] for r in csv.DictReader(open(a.sample_sub))] if a.sample_sub else None
        write_submission(rebuild(p2, r2, s2), a.out, ids); print("wrote", a.out, "mode", mode, "cands", len(r2), flush=True)
    print("RESCORE2_DONE")

if __name__ == "__main__":
    main()
