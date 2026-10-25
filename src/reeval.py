#!/usr/bin/env python3
"""RAC — re-decode raw_<split>.pkl (per-frame probabilities) and evaluate with the ported official metric, without retraining; can also write a challenge submission file.
Usage:
  python reeval.py --meta_dir META_DIR --raw OUT_DIR/raw_test.pkl --split test [--grid]
  python reeval.py --meta_dir ... --raw OUT_DIR/raw_challenge.pkl --split challenge --write sub.csv --sample_sub .../sample_submission.csv --k_runs 3 --no_has --ranknorm
"""
import argparse, os, pickle, json, csv, bisect, numpy as np
from metric import evaluate, factors_from_meta, gts_from_meta
from baseline_meta import decode, write_submission

def ranknorm(preds, lo=0.01, hi=0.99):
    alls = sorted(sc for v in preds.values() for (sc,*_) in v); n = len(alls)
    out = {}
    for k,v in preds.items():
        out[k] = [(lo + (hi-lo)*bisect.bisect_right(alls, sc)/n,) + tuple(rest) for (sc,*rest) in v]
    return out


def _tiou_(s1,e1,s2,e2):
    inter = max(0.0, min(e1,e2)-max(s1,s2)); union = max(e1,e2)-min(s1,s2)
    return inter/union if union > 0 else 0.0

def merge_queries(preds_t, raws_list, dup_scale=0.6, q_min=0.05):
    """Merge each model's raw['queries'] (set-prediction candidates) into the decoded candidates (8-tuples with tier): tIoU > 0.7 with an existing candidate counts as a duplicate and is down-weighted; otherwise tier 2."""
    out = {}
    for sid, cs in preds_t.items():
        cs2 = list(cs)
        for rw in raws_list:
            for q in (rw.get(sid, {}).get("queries") or []):
                if q[0] < q_min: continue
                dup = any(_tiou_(q[1],q[2],c[1],c[2]) > 0.7 for c in cs)
                cs2.append((q[0]*(dup_scale if dup else 1.0), q[1], q[2], q[3], q[4], q[5], q[6], 2 if dup else 0))
        out[sid] = cs2
    return out

BANDS = {0:(0.60,0.99), 1:(0.30,0.60), 2:(0.10,0.30)}
def band(preds_t):
    """preds_t: dict[id] -> list of 8-tuples (..., tier). Within each tier, ranks are normalised to that tier's score band; returns 7-tuples."""
    per = {t: sorted(c[0] for v in preds_t.values() for c in v if c[7]==t) for t in BANDS}
    out = {}
    for k,v in preds_t.items():
        cs = []
        for c in v:
            lo,hi = BANDS[c[7]]; arr = per[c[7]]; n = max(len(arr),1)
            sc = lo + (hi-lo)*(bisect.bisect_right(arr, c[0])/n)
            sc = min(sc, 0.99)
            cs.append((sc,)+tuple(c[1:7]))
        out[k] = cs
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", required=True); ap.add_argument("--raw", required=True, nargs="+"); ap.add_argument("--split", required=True)
    ap.add_argument("--k_runs", type=int, default=3); ap.add_argument("--no_variants", action="store_true"); ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--no_has", action="store_true"); ap.add_argument("--cap", type=float, default=0.99); ap.add_argument("--ranknorm", action="store_true")
    ap.add_argument("--write", default=""); ap.add_argument("--sample_sub", default=""); ap.add_argument("--grid", action="store_true")
    ap.add_argument("--sweep", action="store_true"); ap.add_argument("--band", action="store_true")
    ap.add_argument("--shift", action="store_true"); ap.add_argument("--snap", action="store_true"); ap.add_argument("--bnd", action="store_true")
    ap.add_argument("--multi", action="store_true"); ap.add_argument("--thr2", type=float, default=0.25); ap.add_argument("--split_len", type=int, default=10)
    ap.add_argument("--queries", action="store_true")
    a = ap.parse_args()
    recs = [r for r in pickle.load(open(os.path.join(a.meta_dir, f"meta_{a.split}.pkl"),"rb")) if "error" not in r]
    raws = [pickle.load(open(f,"rb")) for f in a.raw]
    # several models: average per-frame probabilities (a, s, e, pts); average has
    raw = {}
    for sid in raws[0]:
        rs = [r[sid] for r in raws]
        L = min(len(r["a"]) for r in rs)   # models with different T (32/64): truncate to the shortest length
        raw[sid] = dict(a=np.mean([r["a"][:L] for r in rs],0), s=np.mean([r["s"][:L] for r in rs],0), e=np.mean([r["e"][:L] for r in rs],0),
                        pts=np.mean([r["pts"][:L] for r in rs],0), has=float(np.mean([r["has"] for r in rs])), N=min(int(r["N"]) for r in rs))
        bs_ = [r for r in rs if "ds" in r]
        if bs_: raw[sid]["bnd"] = (np.mean([r["ds"][:L] for r in bs_],0), np.mean([r["de"][:L] for r in bs_],0))
    def run(k_runs, variants, thr, use_has, cap, do_band=None):
        do_band = a.band if do_band is None else do_band
        preds = {sid: decode(r["a"], r["s"], r["e"], r["pts"], r["has"], r["N"], k_runs=k_runs, boundary_variants=variants, thr=thr, use_has=use_has, cap=cap, return_tier=True, shift_variants=a.shift, snap_se=a.snap, bnd=(r.get("bnd") if a.bnd else None), multi=a.multi, thr2=a.thr2, split_len=a.split_len) for sid,r in raw.items()}
        if a.queries: preds = merge_queries(preds, raws)
        if do_band: return band(preds)
        preds = {k:[tuple(c[:7]) for c in v] for k,v in preds.items()}
        return ranknorm(preds) if a.ranknorm else preds
    has_labels = any(r["labels"] for r in recs)
    if has_labels:
        gts = gts_from_meta(recs); fac = factors_from_meta(recs)
        if a.sweep:
            for k_runs in (2,3,4):
                for use_has in (True, False):
                    for thr in (0.4,0.5):
                        for do_band in (False, True):
                            res = evaluate(gts, run(k_runs, True, thr, use_has, a.cap, do_band), fac)
                            print(f"k={k_runs} var=1 has={int(use_has)} thr={thr} band={int(do_band)}: mAP {res['mAP']:.4f} mAPs {res['mAPs']:.4f} mAPt {res['mAPt']:.4f}", flush=True)
        preds = run(a.k_runs, not a.no_variants, a.thr, not a.no_has, a.cap)
        res = evaluate(gts, preds, fac, return_grid=a.grid)
        print({k:(round(v,4) if isinstance(v,float) else v) for k,v in res.items() if k!="grid"})
        if a.grid:
            g = res["grid"]; np.set_printoptions(precision=2, suppress=True, linewidth=200)
            print("rows=tau_s 1.0→0.05, cols=tau_t 0.05→0.5\n", g)
            print("mean AP per spatial thr:", np.round(g.mean(1),3)); print("mean AP per temporal thr:", np.round(g.mean(0),3))
    else:
        preds = run(a.k_runs, not a.no_variants, a.thr, not a.no_has, a.cap)
    if a.write:
        ids = [r["id"] for r in csv.DictReader(open(a.sample_sub))] if a.sample_sub else None
        write_submission(preds, a.write, ids); print("wrote", a.write, "cands:", sum(len(v) for v in preds.values()))

if __name__ == "__main__":
    main()
