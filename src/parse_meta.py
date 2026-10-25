#!/usr/bin/env python3
"""RAC — parse RetailAction metadata.json + video frame counts (ffprobe); writes one pickle per split.

Usage:
  python parse_meta.py --root DATA_ROOT --out META_DIR --workers 28

Output (per split): meta_<split>.pkl = list[dict], each dict:
  id, split, seg_dur, labels=[(start,end,x0,y0,x1,y1,label)], views=[v0,v1]
  v = dict(n_video, fts(list[float]|None), face_t(list[float|None]|None),
           poses: float32 [n_list,19,3] (y,x,conf; NaN = missing), faces: float32 [n_list,2] (x,y; NaN = missing),
           ss_t: float32 [m] (seconds) | None, ss_v: float32 [m] | None)
plus summary_<split>.csv for EDA.

Coordinate conventions (verified by overlay):
  pose[k] = [row, col] = [y, x]; face_positions = (col,row) = (x,y); labels spatial = (x,y).
"""
import argparse, json, os, sys, glob, pickle, subprocess, csv, math
import datetime as dt
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

JOINTS = ["top_of_head","neck","right_shoulder","right_elbow","right_wrist","right_hand",
          "left_shoulder","left_elbow","left_wrist","left_hand","middle_of_waist",
          "right_hip","right_knee","right_ankle","right_foot","left_hip","left_knee","left_ankle","left_foot"]
JIDX = {j:i for i,j in enumerate(JOINTS)}

def pts(s):
    if s is None: return None
    s = s.replace("+00:00","").replace("Z","")
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp()

def ffprobe_nframes(path):
    # fast path: read the header with PyAV (milliseconds); fall back to ffprobe
    try:
        import av
        with av.open(path) as c:
            st = c.streams.video[0]; nf = int(st.frames or 0); w = int(st.codec_context.width); h = int(st.codec_context.height)
        if nf > 0 and w > 0: return nf, w, h
    except Exception:
        pass
    try:
        out = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
                              "stream=nb_frames,width,height,avg_frame_rate","-of","csv=p=0",path],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        parts = out.split(",")
        # order per -show_entries: width,height,avg_frame_rate,nb_frames (ffprobe prints in stream order)
        vals = {}
        for p in parts:
            pass
        # robust: query separately
        nf = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=nb_frames","-of","default=nw=1:nk=1",path],capture_output=True,text=True,timeout=60).stdout.strip()
        wh = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height","-of","csv=p=0",path],capture_output=True,text=True,timeout=60).stdout.strip()
        w,h = [int(x) for x in wh.split(",")[:2]]
        return int(nf), w, h
    except Exception as e:
        return -1, -1, -1

def parse_view(v, t0):
    out = {}
    fts = v.get("frame_timestamps")
    out["fts"] = [pts(x)-t0 for x in fts] if fts else None
    faces = v.get("face_positions") or []
    out["face_t"] = [ (pts(f.get("sampled_at"))-t0 if (f and f.get("sampled_at")) else None) for f in faces ] if faces else None
    n_list = max(len(v.get("poses") or []), len(faces))
    poses = np.full((n_list,19,3), np.nan, dtype=np.float32)
    for i,p in enumerate(v.get("poses") or []):
        if not p: continue
        pp = p.get("pose") or {}; js = p.get("joint_scores") or {}
        for j,(yx) in pp.items():
            if j in JIDX and yx is not None:
                poses[i,JIDX[j],0] = yx[0]; poses[i,JIDX[j],1] = yx[1]; poses[i,JIDX[j],2] = js.get(j, np.nan)
    fa = np.full((n_list,2), np.nan, dtype=np.float32)
    for i,f in enumerate(faces):
        if f and f.get("col") is not None:
            fa[i,0] = f["col"]; fa[i,1] = f["row"]
    out["poses"] = poses; out["faces"] = fa
    ss = v.get("sampling_scores")
    if ss:
        out["ss_t"] = np.array([pts(a)-t0 for a,_ in ss], dtype=np.float32)
        out["ss_v"] = np.array([b for _,b in ss], dtype=np.float32)
    else:
        out["ss_t"] = None; out["ss_v"] = None
    return out

def official_factors(raw_json):
    """m_px_factor (metres per normalised unit) computed with the official scorer's function, so it matches Kaggle exactly."""
    try:
        import official_metric as OM
        tgt = OM.parse_groundtruth_from_json(raw_json)
        f = OM.compute_m_px_factor_from_bones(tgt)
        if f is None: return None
        return (float(f.get("rank0", OM.M_PX_FACTOR_AVG)), float(f.get("rank1", OM.M_PX_FACTOR_AVG)))
    except Exception:
        return None

def parse_sample(args):
    sdir, split = args
    sid = os.path.basename(sdir.rstrip("/"))
    try:
        raw_json = open(os.path.join(sdir,"metadata.json")).read()
        m = json.loads(raw_json)["content"]
    except Exception as e:
        return dict(id=sid, split=split, error=f"json:{e}")
    seg = m.get("segment_info",{})
    t0 = pts(seg.get("sampled_at_start")); t1 = pts(seg.get("sampled_at_end"))
    seg_dur = (t1-t0) if (t0 is not None and t1 is not None) else None
    rec = dict(id=sid, split=split, seg_dur=seg_dur, labels=[], views=[], m_px_factors=official_factors(raw_json))
    for a in (m.get("labels",{}) or {}).get("action",[]) or []:
        sp = a["spatial"]["action_cam"]
        rec["labels"].append((float(a["start"]),float(a["end"]),float(sp["rank0"]["x"]),float(sp["rank0"]["y"]),float(sp["rank1"]["x"]),float(sp["rank1"]["y"]),a.get("label")))
    for r in ["rank0","rank1"]:
        v = parse_view(m["action_cam"][r], t0 if t0 is not None else 0.0)
        nf,w,h = ffprobe_nframes(os.path.join(sdir,f"{r}_video.mp4"))
        v["n_video"] = nf; v["w"]=w; v["h"]=h
        rec["views"].append(v)
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--splits", default="train,validation,test,challenge")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    SPLIT_DIRS = {"train":"train/train","validation":"validation/validation","test":"test/test",
                  "challenge":"challenge_test_set/vam-dataset-challenge_test_no_labels/data"}
    for split in args.splits.split(","):
        base = os.path.join(args.root, SPLIT_DIRS[split])
        sdirs = sorted(d for d in glob.glob(os.path.join(base,"*")) if os.path.isdir(d))
        if args.limit: sdirs = sdirs[:args.limit]
        print(f"[{split}] {len(sdirs)} samples at {base}", flush=True)
        with Pool(args.workers) as pool:
            recs = pool.map(parse_sample, [(d,split) for d in sdirs], chunksize=16)
        errs = [r for r in recs if "error" in r]
        print(f"[{split}] errors: {len(errs)}", errs[:3], flush=True)
        with open(os.path.join(args.out, f"meta_{split}.pkl"),"wb") as f:
            pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
        # summary csv
        with open(os.path.join(args.out, f"summary_{split}.csv"),"w",newline="") as f:
            w = csv.writer(f)
            w.writerow(["id","seg_dur","n0_video","n1_video","n0_list","n1_list","has_fts0","has_fts1","has_ss","n_actions",
                        "lab_int_32","lab_int_n0","lab_int_n1","lab_int_nmax","w0","h0"])
            for r in recs:
                if "error" in r: continue
                v0,v1 = r["views"]
                def isint(x,N): return N>0 and abs(x*N-round(x*N))<1e-6
                def allint(N): return all(isint(s,N) and isint(e,N) for (s,e,*_) in r["labels"]) if r["labels"] else ""
                nmax = max(v0["n_video"], v1["n_video"])
                w.writerow([r["id"], f"{r['seg_dur']:.3f}" if r["seg_dur"] is not None else "", v0["n_video"], v1["n_video"],
                            len(v0["poses"]), len(v1["poses"]), int(v0["fts"] is not None), int(v1["fts"] is not None),
                            int(v0["ss_t"] is not None), len(r["labels"]), allint(32), allint(v0["n_video"]), allint(v1["n_video"]), allint(nmax), v0["w"], v0["h"]])
        print(f"[{split}] wrote meta_{split}.pkl + summary_{split}.csv", flush=True)

if __name__ == "__main__":
    main()
