#!/usr/bin/env python3
"""RAC — metadata-only baseline (no images): pose + face + sampling score -> per-frame actionness / start-end / interaction point.

Usage:
  python baseline_meta.py --meta_dir META_DIR --out_dir OUT_DIR \
      --train_splits train,validation --eval_split test --predict_split challenge --epochs 40
Outputs: out_dir/model.pt, out_dir/eval_<split>.json, out_dir/pred_<split>.json, out_dir/submission_<tag>.csv
"""
import argparse, os, json, pickle, math, time, csv, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from metric import evaluate, factors_from_meta, gts_from_meta, JOINTS, JIDX

T = 32
HANDS = ["right_hand","left_hand","right_wrist","left_wrist"]

def frame_times(v, N):
    """Time (seconds) of each slot; None if unknown. Uses fts first, then face_t."""
    t = [None]*N
    if v["fts"]:
        for i,x in enumerate(v["fts"][:N]): t[i]=x
    elif v["face_t"]:
        for i,x in enumerate(v["face_t"][:N]):
            if x is not None: t[i]=x
    return t

def grid_N(rec):
    n0, n1 = rec["views"][0]["n_video"], rec["views"][1]["n_video"]
    n = max(n0, n1)
    if n <= 0: n = max(len(rec["views"][0]["poses"]), len(rec["views"][1]["poses"]))
    return int(min(max(n,1), T))

def build_features(rec):
    N = grid_N(rec); dur = rec["seg_dur"] or 0.0
    feats = []
    for v in rec["views"]:
        P = v["poses"]; L = P.shape[0]
        pose = np.zeros((T,19,3),np.float32); pres = np.zeros((T,19),np.float32)
        n = min(L,T)
        if n: 
            pp = P[:n]; ok = ~np.isnan(pp[:,:,0])
            pose[:n] = np.nan_to_num(pp); pres[:n] = ok
        face = np.zeros((T,3),np.float32)
        Fc = v["faces"]; n2 = min(Fc.shape[0],T)
        if n2:
            okf = ~np.isnan(Fc[:n2,0]); face[:n2,:2] = np.nan_to_num(Fc[:n2]); face[:n2,2] = okf
        tms = frame_times(v, N)
        tf = np.zeros((T,5),np.float32)  # ss_val, ss_localmax, has_ss, t_norm, has_t
        if v["ss_t"] is not None and len(v["ss_t"])>1:
            st, sv = v["ss_t"], v["ss_v"]
            for i in range(N):
                if tms[i] is None: continue
                tf[i,0] = np.interp(tms[i], st, sv)
                w = (st >= tms[i]-0.35) & (st <= tms[i]+0.35)
                tf[i,1] = sv[w].max() if w.any() else tf[i,0]
                tf[i,2] = 1.0
        for i in range(N):
            if tms[i] is not None and dur>0: tf[i,3] = tms[i]/dur; tf[i,4] = 1.0
        idx = np.zeros((T,4),np.float32)
        for i in range(N): idx[i] = [i/N, (i+0.5)/N, N/T, 1.0]
        # hand kinematics: displacement to the previous/next slot and distance to the neck
        kin = np.zeros((T,len(HANDS)*5),np.float32)
        neck = pose[:,JIDX["neck"],:2]
        for h_i,h in enumerate(HANDS):
            j = JIDX[h]; xy = pose[:,j,:2]; ok = pres[:,j] > 0
            for i in range(N):
                base = h_i*5
                if ok[i] and i>0 and ok[i-1]: kin[i,base:base+2] = xy[i]-xy[i-1]
                if ok[i] and i+1<N and ok[i+1]: kin[i,base+2:base+4] = xy[i+1]-xy[i]
                if ok[i] and pres[i,JIDX["neck"]]>0: kin[i,base+4] = np.hypot(*(xy[i]-neck[i]))
        feats.append(np.concatenate([pose.reshape(T,-1), pres, face, tf, idx, kin], axis=1))
    X = np.concatenate(feats, axis=1)  # [T, F]
    valid = np.zeros(T,np.float32); valid[:N] = 1
    return X, valid, N

def build_targets(rec, N):
    act = np.zeros(T,np.float32); st = np.zeros(T,np.float32); en = np.zeros(T,np.float32)
    pts = np.zeros((T,4),np.float32); pm = np.zeros(T,np.float32)
    for (s,e,x0,y0,x1,y1,_) in rec["labels"]:
        a = int(round(s*N)); b = int(round(e*N))  # [a,b)
        a = max(0,min(a,N-1)); b = max(a+1,min(b,N))
        act[a:b] = 1; st[a] = 1; en[b-1] = 1
        pts[a:b] = [x0,y0,x1,y1]; pm[a:b] = 1
    has = 1.0 if rec["labels"] else 0.0
    return act, st, en, pts, pm, has

def load_img_feats(feats_dir, rec):
    """Read the output of extract_feats.py; returns [T, 2*K*d] float16 (index aligned with video frames)."""
    parts = []
    for v in (0,1):
        f = np.load(os.path.join(feats_dir, rec["split"], f"{rec['id']}_{v}.npy"))
        d = f.shape[1] if f.ndim == 2 and f.shape[0] > 0 else None
        if d is None or d == 1:
            d = load_img_feats.dim
            f = np.zeros((0, d), np.float16)
        load_img_feats.dim = d
        out = np.zeros((T, d), np.float16); n = min(len(f), T); out[:n] = f[:n]
        parts.append(out)
    return np.concatenate(parts, 1)
load_img_feats.dim = 1920

class DS(torch.utils.data.Dataset):
    def __init__(self, recs, with_targets=True, feats_dir=None):
        self.items = []
        for r in recs:
            X, valid, N = build_features(r)
            if feats_dir:
                X = np.concatenate([X.astype(np.float16), load_img_feats(feats_dir, r)], 1)
            else:
                X = X.astype(np.float16)
            tg = build_targets(r, N) if with_targets else None
            self.items.append((r["id"], X, valid, N, tg))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        sid, X, valid, N, tg = self.items[i]
        out = dict(X=torch.from_numpy(X.astype(np.float32)), valid=torch.from_numpy(valid), N=N)
        if tg is not None:
            act, st, en, pts, pm, has = tg
            out.update(act=torch.from_numpy(act), st=torch.from_numpy(st), en=torch.from_numpy(en), pts=torch.from_numpy(pts), pm=torch.from_numpy(pm), has=torch.tensor(has))
        return out

class Net(nn.Module):
    def __init__(self, fdim, d=192, layers=3, heads=6, drop=0.1, in_drop=0.0):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(in_drop), nn.Linear(fdim,d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
        self.pos = nn.Parameter(torch.zeros(1,T,d)); nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 4*d, drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head_act = nn.Linear(d,3)   # act, start, end
        self.head_pts = nn.Linear(d,4)
        self.head_has = nn.Linear(d,1)
    def forward(self, X, valid):
        h = self.inp(X) + self.pos
        h = self.enc(h, src_key_padding_mask=(valid<0.5))
        a = self.head_act(h); p = torch.sigmoid(self.head_pts(h))
        m = valid.unsqueeze(-1); g = (h*m).sum(1)/m.sum(1).clamp(min=1)
        return a[...,0], a[...,1], a[...,2], p, self.head_has(g)[...,0]

def masked_bce(logit, target, mask, pos_weight=1.0):
    l = F.binary_cross_entropy_with_logits(logit, target, reduction="none", pos_weight=torch.tensor(pos_weight, device=logit.device))
    return (l*mask).sum()/mask.sum().clamp(min=1)

def train(model, dl, dl_val, epochs, lr, dev):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs*len(dl); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    for ep in range(epochs):
        model.train(); tl = 0; n = 0
        for b in dl:
            b = {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in b.items()}
            a,s,e,p,hs = model(b["X"], b["valid"])
            loss = masked_bce(a,b["act"],b["valid"],1.5) + 0.5*masked_bce(s,b["st"],b["valid"],4.0) + 0.5*masked_bce(e,b["en"],b["valid"],4.0)
            lp = (F.smooth_l1_loss(p,b["pts"],reduction="none",beta=0.02).sum(-1)*b["pm"]).sum()/b["pm"].sum().clamp(min=1)
            loss = loss + 10.0*lp + 0.5*F.binary_cross_entropy_with_logits(hs,b["has"])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl += loss.item()*len(b["X"]); n += len(b["X"])
        msg = f"ep {ep+1}/{epochs} train_loss {tl/n:.4f}"
        if dl_val is not None and ((ep+1)%5==0 or ep+1==epochs):
            model.eval(); vl=0; vn=0; accs=[]
            with torch.no_grad():
                for b in dl_val:
                    b = {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in b.items()}
                    a,s,e,p,hs = model(b["X"], b["valid"])
                    vl += masked_bce(a,b["act"],b["valid"]).item()*len(b["X"]); vn += len(b["X"])
                    accs.append((((a>0).float()==b["act"]).float()*b["valid"]).sum().item()/b["valid"].sum().item())
            msg += f" | val_act_bce {vl/vn:.4f} frame_acc {np.mean(accs):.4f}"
        print(msg, flush=True)
    return model

def decode(a, s, e, p, has, N, k_runs=3, boundary_variants=True, thr=0.5, use_has=True, cap=0.99, return_tier=False, shift_variants=False, snap_se=False, bnd=None, multi=False, thr2=0.25, split_len=10):
    """a,s,e: sigmoid probs [T]; p: [T,4]; returns list of (score,start,end,x0,y0,x1,y1[,tier])
    tier: 0 = main run, main candidate / 1 = main run, boundary variant / 2 = secondary run and its variants.
    Official scorer: clips without GT do not count FPs -> use_has can be disabled; scores are capped (hits at 1.0 are not scored)."""
    a = torch.as_tensor(a)[:N]; p = torch.as_tensor(p)[:N]
    has = float(has) if use_has else 1.0
    cands = []
    on = (a > thr).clone()
    for i in range(1,N-1):
        if not on[i] and on[i-1] and on[i+1]: on[i] = True
    runs = []; i = 0
    while i < N:
        if on[i]:
            j = i
            while j+1 < N and on[j+1]: j += 1
            runs.append((i,j)); i = j+1
        else: i += 1
    def mk(i,j,score,tier):
        w = a[i:j+1]; wp = (p[i:j+1]*w[:,None]).sum(0)/w.sum().clamp(min=1e-6)
        t = (float(min(score, cap)), i/N, (j+1)/N, float(wp[0]),float(wp[1]),float(wp[2]),float(wp[3]))
        return t + (tier,) if return_tier else t
    runs = sorted(runs, key=lambda r: -float(a[r[0]:r[1]+1].max()))[:k_runs]
    s_ = torch.as_tensor(s)[:N]; e_ = torch.as_tensor(e)[:N]
    for r,(i,j) in enumerate(runs):
        w = a[i:j+1]; sc = (0.5*float(w.mean())+0.5*float(w.max()))*has
        if bnd is not None:  # boundary regression head: median over the run's frames of the predicted (distance to start, distance to end)
            ds = torch.as_tensor(bnd[0])[:N]; de = torch.as_tensor(bnd[1])[:N]
            ks = torch.arange(i, j+1, dtype=torch.float32)
            st_est = torch.median(ks - ds[i:j+1]*32.0); en_est = torch.median(ks + de[i:j+1]*32.0)
            i2 = int(max(0, min(N-1, round(float(st_est))))); j2 = int(max(0, min(N-1, round(float(en_est)))))
            if j2 >= i2 and abs(i2-i) <= 3 and abs(j2-j) <= 3:
                cands.append(mk(i,j,sc*0.6, 1 if r==0 else 2))   # keep the original run as a variant
                i, j = i2, j2
        if snap_se:  # refine boundaries within +-2 frames with the start/end heads
            lo, hi = max(0,i-2), min(N-1,i+2); i2 = lo + int(torch.argmax(s_[lo:hi+1]))
            lo, hi = max(0,j-2), min(N-1,j+2); j2 = lo + int(torch.argmax(e_[lo:hi+1]))
            if j2 >= i2: i, j = i2, j2
        cands.append(mk(i,j,sc, 0 if r==0 else 2))
        if boundary_variants:
            if j-i >= 2: cands.append(mk(i+1,j-1,sc*0.5, 1 if r==0 else 2))
            cands.append(mk(max(0,i-1),min(N-1,j+1),sc*0.4, 1 if r==0 else 2))
        if shift_variants:
            if i-1 >= 0: cands.append(mk(i-1,max(i-1,j-1),sc*0.35, 1 if r==0 else 2))
            if j+1 <= N-1: cands.append(mk(min(i+1,j+1),j+1,sc*0.3, 1 if r==0 else 2))
    if multi:
        # (1) split long runs at the strongest end(k)+start(k+1) inside the run; one candidate per part
        for r,(i,j) in enumerate(runs):
            if j - i + 1 >= split_len:
                best_k = None; best_v = -1
                for k in range(i+1, j-1):
                    v = float(e_[k]) + float(s_[k+1])
                    if v > best_v and (k-i+1) >= 2 and (j-k) >= 2: best_v = v; best_k = k
                if best_k is not None:
                    w = a[i:j+1]; sc = (0.5*float(w.mean())+0.5*float(w.max()))*has
                    cands.append(mk(i, best_k, sc*0.55, 2)); cands.append(mk(best_k+1, j, sc*0.55, 2))
        # (2) secondary runs at a lower threshold (not overlapping existing runs)
        covered = torch.zeros(N, dtype=torch.bool)
        for (i,j) in runs: covered[i:j+1] = True
        on2 = (a > thr2) & ~covered; i = 0; extra = []
        while i < N:
            if on2[i]:
                j = i
                while j+1 < N and on2[j+1]: j += 1
                if j - i + 1 >= 2: extra.append((i,j))
                i = j+1
            else: i += 1
        extra = sorted(extra, key=lambda r: -float(a[r[0]:r[1]+1].max()))[:2]
        for (i,j) in extra:
            w = a[i:j+1]; sc = (0.5*float(w.mean())+0.5*float(w.max()))*has*0.8
            cands.append(mk(i,j,sc, 2))
            cands.append(mk(max(0,i-1),min(N-1,j+1),sc*0.5, 2))
    if not runs:
        c = int(torch.argmax(a)); i = max(0,c-2); j = min(N-1,c+3)
        cands.append(mk(i,j,float(a.max())*has*0.8, 0))
    return cands

def predict(model, ds, dev, bs=256, raw_out=None, **dk):
    model.eval(); out = {}
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)
    k = 0
    with torch.no_grad():
        for b in dl:
            X = b["X"].to(dev); valid = b["valid"].to(dev)
            a,s,e,p,hs = model(X, valid)
            a = torch.sigmoid(a).cpu(); s = torch.sigmoid(s).cpu(); e = torch.sigmoid(e).cpu(); p = p.cpu(); hs = torch.sigmoid(hs).cpu()
            for bi in range(len(X)):
                sid = ds.items[k][0]; N = ds.items[k][3]; k += 1
                out[sid] = decode(a[bi], s[bi], e[bi], p[bi], hs[bi], N, **dk)
                if raw_out is not None:
                    raw_out[sid] = dict(a=a[bi,:N].numpy(), s=s[bi,:N].numpy(), e=e[bi,:N].numpy(), pts=p[bi,:N].numpy(), has=float(hs[bi]), N=N)
    return out

def write_submission(preds, path, sample_ids=None):
    ids = sample_ids if sample_ids is not None else sorted(preds)
    with open(path,"w",newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL); w.writerow(["id","prediction"])
        for sid in ids:
            acts = []
            for (sc,s,e,x0,y0,x1,y1) in preds.get(sid,[]):
                s = float(min(max(s,0.0),1.0)); e = float(min(max(e,0.0),1.0))
                if e <= s: e = min(1.0, s+1e-3)
                acts.append({"score":float(min(max(sc,0.0),1.0)),"start":s,"end":e,
                             "spatial":{"action_cam":{"rank0":{"x":float(np.clip(x0,0,1)),"y":float(np.clip(y0,0,1))},
                                                      "rank1":{"x":float(np.clip(x1,0,1)),"y":float(np.clip(y1,0,1))}}}})
            w.writerow([sid, json.dumps({"actions":acts})])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_splits", default="train,validation"); ap.add_argument("--eval_split", default="test")
    ap.add_argument("--predict_split", default="challenge"); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="meta_v0")
    ap.add_argument("--sample_sub", default=""); ap.add_argument("--no_variants", action="store_true")
    ap.add_argument("--img_feats", default=""); ap.add_argument("--d", type=int, default=192); ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--drop", type=float, default=0.1); ap.add_argument("--in_drop", type=float, default=0.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    load = lambda sp: [r for r in pickle.load(open(os.path.join(args.meta_dir,f"meta_{sp}.pkl"),"rb")) if "error" not in r]
    tr = sum([load(s) for s in args.train_splits.split(",")], [])
    ev = load(args.eval_split) if args.eval_split else []
    print(f"train {len(tr)} eval {len(ev)} device {dev}", flush=True)
    fd = args.img_feats or None
    t0 = time.time(); ds_tr = DS(tr, feats_dir=fd); ds_ev = DS(ev, feats_dir=fd) if ev else None; print(f"features built in {time.time()-t0:.1f}s, fdim={ds_tr.items[0][1].shape[1]}", flush=True)
    dl = torch.utils.data.DataLoader(ds_tr, batch_size=args.bs, shuffle=True, drop_last=True)
    dl_ev = torch.utils.data.DataLoader(ds_ev, batch_size=args.bs) if ds_ev else None
    model = Net(ds_tr.items[0][1].shape[1], d=args.d, layers=args.layers, heads=max(1,args.d//32), drop=args.drop, in_drop=args.in_drop).to(dev)
    model = train(model, dl, dl_ev, args.epochs, args.lr, dev)
    torch.save(model.state_dict(), os.path.join(args.out_dir,"model.pt"))
    if ds_ev:
        gts = gts_from_meta(ev); fac = factors_from_meta(ev)
        res = {}
        raw_ev = {}
        for name,dk in [("full",dict()),("no_variants",dict(boundary_variants=False)),("top1",dict(k_runs=1,boundary_variants=False))]:
            pr = predict(model, ds_ev, dev, raw_out=(raw_ev if name=="full" else None), **dk)
            r = evaluate(gts, pr, fac); res[name] = r; print(f"[eval {args.eval_split}] {name}: {r}", flush=True)
            if name=="full": json.dump(pr, open(os.path.join(args.out_dir,f"pred_{args.eval_split}.json"),"w"))
        json.dump(res, open(os.path.join(args.out_dir,f"eval_{args.eval_split}.json"),"w"), indent=1)
        pickle.dump(raw_ev, open(os.path.join(args.out_dir,f"raw_{args.eval_split}.pkl"),"wb"))
    if args.predict_split:
        pdx = load(args.predict_split); ds_p = DS(pdx, with_targets=False, feats_dir=fd)
        raw_p = {}
        pr = predict(model, ds_p, dev, raw_out=raw_p, boundary_variants=not args.no_variants)
        json.dump(pr, open(os.path.join(args.out_dir,f"pred_{args.predict_split}.json"),"w"))
        pickle.dump(raw_p, open(os.path.join(args.out_dir,f"raw_{args.predict_split}.pkl"),"wb"))
        ids = None
        if args.sample_sub:
            ids = [r["id"] for r in csv.DictReader(open(args.sample_sub))]
            missing = [i for i in ids if i not in pr]; print("ids in sample_sub:",len(ids),"missing preds:",len(missing))
        write_submission(pr, os.path.join(args.out_dir,f"submission_{args.tag}.csv"), ids)
        print("wrote", os.path.join(args.out_dir,f"submission_{args.tag}.csv"), flush=True)

if __name__ == "__main__":
    main()
