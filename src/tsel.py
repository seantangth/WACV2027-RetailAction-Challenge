#!/usr/bin/env python3
"""RAC — T selection for the two-tier structure: for the top candidate of each clip, predict the probability that it is a TP at the tightest cell (5 cm, tIoU 0.5), i.e. at all 200 cells.
Trained on test (OOF: KFold, one sample per clip); reports the precision of the top M by OOF probability vs the top M by rescorer score; the model can be saved and applied to challenge.
Features: rescorer score and candidate features (k, q, d, run_len, a_*, ...) + the 5 cm pixel tolerance (from m_px_factor) + peak dominance (k/q).
Usage:
  python tsel.py fit   --meta_dir M --csv test_oof.csv --feats test_feats.pkl --save tsel_model
  python tsel.py apply --meta_dir M --split challenge --csv sub.csv --feats sub_feats.pkl --model tsel_model --out sel.json --M 60
"""
import argparse, json, math, os, pickle, numpy as np, lightgbm as lgb
from sklearn.model_selection import KFold
from metric import gts_from_meta, factors_from_meta, EPS, M_PX_FACTOR_AVG
from rescore2 import load_csv, align
from postprocess import load_fhm, agg_peaks, frame_peak, grid_N_from_meta
F = ["score", "k0", "k1", "q0", "q1", "d0", "d1", "kmin", "kmax", "dom0", "dom1", "run_len", "N", "a_mean", "a_max", "a_out", "has", "n_cands",
     "tol0", "tol1", "tolmax", "d0t", "d1t", "dur", "tier", "kind", "gap12"]
FH = ["fpk0", "fpk1", "fr20", "fr21", "fdist0", "fdist1", "fspread0", "fspread1", "fin0", "fin1"]   # per-frame heat-map features (only with --fhm)

def fhm_feats(F, c, N, tol0, tol1):
    """Per-frame heat maps of a candidate segment: aggregated peak value, second/first peak ratio, distance between candidate point and aggregated peak (in units of the 5 cm tolerance), spread of the per-frame peaks and the share within 5 cm of the aggregated peak."""
    out = {k: np.nan for k in FH}
    if F is None: return out
    i0 = int(round(c[1] * N)); i1 = max(i0 + 1, int(round(c[2] * N))); fis = list(range(i0, i1))
    for v, tol in ((0, tol0), (1, tol1)):
        p = agg_peaks(F[v], fis, k=2)
        if not p: continue
        out[f"fpk{v}"] = p[0][2]; out[f"fr2{v}"] = (p[1][2] / (p[0][2] + 1e-6)) if len(p) > 1 else 0.0
        px, py = c[3 + 2 * v], c[4 + 2 * v]; out[f"fdist{v}"] = math.hypot(px - p[0][0], py - p[0][1]) * 600 / tol
        fp = [frame_peak(F[v], f) for f in fis]; fp = [q for q in fp if q]
        if len(fp) >= 2:
            xs = np.array([[q[0], q[1]] for q in fp]); out[f"fspread{v}"] = float(np.sqrt(((xs - xs.mean(0)) ** 2).sum(1).mean())) * 600 / tol
            out[f"fin{v}"] = float(np.mean([math.hypot(q[0] - p[0][0], q[1] - p[0][1]) * 600 <= tol for q in fp]))
    return out

def top1_rows(preds, feats, fac, fhm_dir="", Ns=None):
    rows = align(preds, feats); by = {}
    for (sid, i, f) in rows:
        c = preds[sid][i]
        if sid not in by or c[0] > preds[sid][by[sid][1]][0]: by[sid] = (sid, i, f)
    out = []
    for sid, (_, i, f) in by.items():
        c = preds[sid][i]; f0, f1 = fac.get(sid, (M_PX_FACTOR_AVG, M_PX_FACTOR_AVG)); sc = sorted([x[0] for x in preds[sid]], reverse=True)
        f = f or {}
        tol0 = 0.05 / f0 * 600; tol1 = 0.05 / f1 * 600
        x = dict(score=c[0], k0=f.get("k0", np.nan), k1=f.get("k1", np.nan), q0=f.get("q0", np.nan), q1=f.get("q1", np.nan), d0=f.get("d0", np.nan), d1=f.get("d1", np.nan),
                 run_len=f.get("run_len", np.nan), N=f.get("N", np.nan), a_mean=f.get("a_mean", np.nan), a_max=f.get("a_max", np.nan), a_out=f.get("a_out", np.nan),
                 has=f.get("has", np.nan), n_cands=f.get("n_cands", np.nan), tol0=tol0, tol1=tol1, tolmax=max(tol0, tol1), dur=c[2] - c[1],
                 tier=f.get("tier", np.nan), kind=f.get("kind", np.nan), gap12=sc[0] - (sc[1] if len(sc) > 1 else 0))
        x["kmin"] = np.nanmin([x["k0"], x["k1"]]); x["kmax"] = np.nanmax([x["k0"], x["k1"]])
        x["dom0"] = x["k0"] / (x["q0"] + 1e-3); x["dom1"] = x["k1"] / (x["q1"] + 1e-3)
        x["d0t"] = x["d0"] * 600 / tol0 if np.isfinite(x["d0"]) else np.nan; x["d1t"] = x["d1"] * 600 / tol1 if np.isfinite(x["d1"]) else np.nan
        if fhm_dir: x.update(fhm_feats(load_fhm(fhm_dir, sid), c, Ns[sid], tol0, tol1))
        out.append((sid, c, x))
    return out

def hit_tight(c, g, fac_sid):
    f0, f1 = fac_sid
    for (gs, ge, gx0, gy0, gx1, gy1) in g:
        inter = min(c[2], ge) - max(c[1], gs); union = max(c[2], ge) - min(c[1], gs); T = max(0.0, inter / (union + EPS))
        D = min(math.hypot(c[3] - gx0, c[4] - gy0) * f0, math.hypot(c[5] - gx1, c[6] - gy1) * f1)
        if T >= 0.5 and D <= 0.05: return 1
    return 0

P = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=6)

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--meta_dir", required=True); f.add_argument("--csv", required=True); f.add_argument("--feats", required=True); f.add_argument("--save", default=""); f.add_argument("--rounds", type=int, default=400)
    f.add_argument("--fhm", default="", help="per-frame heat-map directory (with test/); if given, FH features are added"); f.add_argument("--oof_out", default="", help="save the OOF probabilities on test as JSON ({clip: p}) for postprocess.py eval --tsel_oof")
    g = sub.add_parser("apply"); g.add_argument("--meta_dir", required=True); g.add_argument("--split", default="challenge"); g.add_argument("--csv", required=True); g.add_argument("--feats", required=True)
    g.add_argument("--model", required=True); g.add_argument("--out", required=True); g.add_argument("--M", type=int, default=60); g.add_argument("--fhm", default="")
    a = ap.parse_args()
    split = "test" if a.cmd == "fit" else a.split
    recs = [r for r in pickle.load(open(os.path.join(a.meta_dir, f"meta_{split}.pkl"), "rb")) if "error" not in r]; fac = factors_from_meta(recs)
    preds = load_csv(a.csv); feats = pickle.load(open(a.feats, "rb"))
    FF = F + (FH if a.fhm else [])
    rows = top1_rows(preds, feats, fac, os.path.join(a.fhm, split) if a.fhm else "", grid_N_from_meta(recs)); X = np.array([[x.get(k, np.nan) for k in FF] for (_, _, x) in rows], np.float32)
    if a.cmd == "fit":
        gts = gts_from_meta(recs)
        keep = [i for i, (sid, c, x) in enumerate(rows) if gts.get(sid)]
        X = X[keep]; rows = [rows[i] for i in keep]; y = np.array([hit_tight(c, gts[sid], fac[sid]) for (sid, c, x) in rows])
        print(f"clips with GT {len(rows)} | top-1 tight hit rate {y.mean():.3f}", flush=True)
        oof = np.zeros(len(y)); imp = np.zeros(len(FF))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            m = lgb.train(P, lgb.Dataset(X[tr], y[tr]), num_boost_round=a.rounds); oof[te] = m.predict(X[te]); imp += m.feature_importance("gain")
        sc = np.array([x["score"] for (_, _, x) in rows])
        if a.oof_out: json.dump({rows[i][0]: float(oof[i]) for i in range(len(rows))}, open(a.oof_out, "w")); print("saved OOF", a.oof_out, flush=True)
        for M in (10, 20, 30, 40, 60, 80, 100, 150, 200):
            by_s = np.argsort(-sc)[:M]; by_p = np.argsort(-oof)[:M]
            print(f"  M={M:<4d} prec(tight) by rescorer {y[by_s].mean():.3f} | by T-classifier OOF {y[by_p].mean():.3f}", flush=True)
        print("  gain:", sorted([(FF[j], int(imp[j])) for j in range(len(FF))], key=lambda t: -t[1])[:12], flush=True)
        if a.save:
            m = lgb.train(P, lgb.Dataset(X, y), num_boost_round=a.rounds); m.save_model(a.save + ".txt"); json.dump({"F": FF}, open(a.save + ".json", "w")); print("saved", a.save, flush=True)
    else:
        m = lgb.Booster(model_file=a.model + ".txt"); p = m.predict(X)
        order = np.argsort(-p)[:a.M]; sel = {rows[i][0]: float(p[i]) for i in order}
        json.dump(sel, open(a.out, "w")); print(f"selected {len(sel)} clips; prob range {min(sel.values()):.3f}-{max(sel.values()):.3f}", flush=True)

if __name__ == "__main__":
    main()
