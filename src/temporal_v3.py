#!/usr/bin/env python3
"""RAC — temporal model v3: adds a DETR-style set-prediction decoder (Q queries -> objectness / start-end / point per view) on top of tv2 (64 slots, per-frame heads, boundary regression),
with Hungarian matching (objectness + L1(start,end) + (1 - tIoU)), aimed at clips with several actions. raw also contains 'queries': list of (score,start,end,x0,y0,x1,y1).
Usage as temporal_v2.py (plus --nq, --w_set).
"""
import argparse, os, json, pickle, time, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import baseline_meta as bm
bm.T = 64
from baseline_meta import DS, Net, masked_bce, decode, write_submission
from temporal_v2 import bnd_targets
from metric import evaluate, factors_from_meta, gts_from_meta

class Net3(Net):
    def __init__(self, fdim, nq=6, **kw):
        super().__init__(fdim, **kw)
        d = self.head_pts.in_features
        self.head_bnd = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        self.queries = nn.Parameter(torch.randn(nq, d) * 0.02)
        dec = nn.TransformerDecoderLayer(d, max(1, d//32), 4*d, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.dec = nn.TransformerDecoder(dec, 2)
        self.q_obj = nn.Linear(d, 1); self.q_se = nn.Linear(d, 2); self.q_pts = nn.Linear(d, 4)
    def forward(self, X, valid):
        h = self.inp(X) + self.pos
        h = self.enc(h, src_key_padding_mask=(valid<0.5))
        a = self.head_act(h); p = torch.sigmoid(self.head_pts(h)); b = F.softplus(self.head_bnd(h))
        m = valid.unsqueeze(-1); g = (h*m).sum(1)/m.sum(1).clamp(min=1)
        q = self.dec(self.queries.unsqueeze(0).expand(len(X), -1, -1), h, memory_key_padding_mask=(valid<0.5))
        se = torch.sigmoid(self.q_se(q))                       # [B,Q,2] (center, width) in [0,1] of slot grid
        qs = (se[...,0] - se[...,1]/2).clamp(0,1); qe = (se[...,0] + se[...,1]/2).clamp(0,1)
        return a[...,0], a[...,1], a[...,2], p, self.head_has(g)[...,0], b, self.q_obj(q)[...,0], torch.stack([qs,qe],-1), torch.sigmoid(self.q_pts(q))

def gt_instances(rec_batch_act, N_list):
    """Recover GT intervals from the per-frame act mask (one instance per run) -> list of tensors [n_i,2] (start,end) in slot-fraction of N"""
    out = []
    for act, N in zip(rec_batch_act.cpu().numpy(), N_list):
        segs = []; i = 0
        while i < N:
            if act[i] > 0.5:
                j = i
                while j+1 < N and act[j+1] > 0.5: j += 1
                segs.append((i/N, (j+1)/N)); i = j+1
            else: i += 1
        out.append(segs)
    return out

def tiou_mat(p, g):  # p [Q,2], g [G,2]
    inter = (torch.min(p[:,None,1], g[None,:,1]) - torch.max(p[:,None,0], g[None,:,0])).clamp(min=0)
    union = (torch.max(p[:,None,1], g[None,:,1]) - torch.min(p[:,None,0], g[None,:,0])).clamp(min=1e-6)
    return inter/union

def set_loss(q_obj, q_se, q_pts, acts, pts_t, pm, N_list, dev):
    """Loss after Hungarian matching; the GT point is the mean pts target over the interval's frames."""
    B, Q = q_obj.shape; l_obj = 0; l_se = 0; l_pt = 0; n_match = 0
    insts = gt_instances(acts, N_list)
    tgt_obj = torch.zeros(B, Q, device=dev)
    for b in range(B):
        segs = insts[b]
        if not segs: continue
        g = torch.tensor(segs, device=dev, dtype=torch.float32)
        prob = torch.sigmoid(q_obj[b]).detach()
        cost = -prob[:,None] + (q_se[b][:,None,:] - g[None,:,:]).abs().sum(-1).detach() + (1 - tiou_mat(q_se[b].detach(), g))
        qi, gi = linear_sum_assignment(cost.cpu().numpy())
        for a_, c_ in zip(qi, gi):
            tgt_obj[b, a_] = 1.0
            l_se = l_se + F.l1_loss(q_se[b, a_], g[c_], reduction="sum") + (1 - tiou_mat(q_se[b, a_][None], g[c_][None])[0,0])
            s_, e_ = segs[c_]; i0, i1 = int(round(s_*N_list[b])), int(round(e_*N_list[b]))
            msk = pm[b, i0:i1]
            if msk.sum() > 0:
                gp = (pts_t[b, i0:i1] * msk[:,None]).sum(0) / msk.sum()
                l_pt = l_pt + F.smooth_l1_loss(q_pts[b, a_], gp, reduction="sum", beta=0.02)
            n_match += 1
    l_obj = F.binary_cross_entropy_with_logits(q_obj, tgt_obj, pos_weight=torch.tensor(2.0, device=dev))
    nm = max(n_match, 1)
    return l_obj + 2.0*l_se/nm + 10.0*l_pt/nm

def train(model, dl, dl_val, epochs, lr, dev, ls, w_bnd, w_set):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs*len(dl), pct_start=0.1)
    for ep in range(epochs):
        model.train(); tl = 0; n = 0
        for b in dl:
            b = {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in b.items()}
            a,s,e,p,hs,bd,qo,qse,qp = model(b["X"], b["valid"])
            act_t = b["act"]*(1-ls) + 0.5*ls
            loss = masked_bce(a,act_t,b["valid"],1.5) + 0.5*masked_bce(s,b["st"],b["valid"],4.0) + 0.5*masked_bce(e,b["en"],b["valid"],4.0)
            lp = (F.smooth_l1_loss(p,b["pts"],reduction="none",beta=0.02).sum(-1)*b["pm"]).sum()/b["pm"].sum().clamp(min=1)
            ds, de = bnd_targets(b["act"])
            lb = (F.smooth_l1_loss(bd[...,0], ds, reduction="none", beta=0.03) + F.smooth_l1_loss(bd[...,1], de, reduction="none", beta=0.03))
            lb = (lb*b["act"]).sum()/b["act"].sum().clamp(min=1)
            N_list = [int(x) for x in b["N"]]
            lset = set_loss(qo, qse, qp, b["act"], b["pts"], b["pm"], N_list, dev)
            loss = loss + 10.0*lp + 0.5*F.binary_cross_entropy_with_logits(hs,b["has"]) + w_bnd*lb + w_set*lset
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl += loss.item()*len(b["X"]); n += len(b["X"])
        msg = f"ep {ep+1}/{epochs} train_loss {tl/n:.4f}"
        if dl_val is not None and ((ep+1)%10==0 or ep+1==epochs):
            model.eval(); vl=0; vn=0
            with torch.no_grad():
                for b in dl_val:
                    b = {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in b.items()}
                    out = model(b["X"], b["valid"]); vl += masked_bce(out[0],b["act"],b["valid"]).item()*len(b["X"]); vn += len(b["X"])
            msg += f" | val_act_bce {vl/vn:.4f}"
        print(msg, flush=True)
    return model

def predict(model, ds, dev, bs=256, q_min=0.05, **dk):
    model.eval(); out = {}; raw = {}
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False); k = 0
    with torch.no_grad():
        for b in dl:
            X = b["X"].to(dev); valid = b["valid"].to(dev)
            a,s,e,p,hs,bd,qo,qse,qp = model(X, valid)
            a = torch.sigmoid(a).cpu(); s = torch.sigmoid(s).cpu(); e = torch.sigmoid(e).cpu(); p = p.cpu(); hs = torch.sigmoid(hs).cpu(); bd = bd.cpu()
            qo = torch.sigmoid(qo).cpu(); qse = qse.cpu(); qp = qp.cpu()
            for bi in range(len(X)):
                sid = ds.items[k][0]; N = ds.items[k][3]; k += 1
                qs = []
                for qi in range(qo.shape[1]):
                    sc = float(qo[bi,qi])
                    if sc < q_min: continue
                    st_ = round(float(qse[bi,qi,0])*N)/N; en_ = round(float(qse[bi,qi,1])*N)/N   # snap to the frame grid
                    if en_ <= st_: en_ = min(1.0, st_ + 1.0/N)
                    qs.append((sc, st_, en_, *[float(x) for x in qp[bi,qi]]))
                raw[sid] = dict(a=a[bi,:N].numpy(), s=s[bi,:N].numpy(), e=e[bi,:N].numpy(), pts=p[bi,:N].numpy(), has=float(hs[bi]), N=N,
                                ds=bd[bi,:N,0].numpy(), de=bd[bi,:N,1].numpy(), queries=qs)
                out[sid] = decode(a[bi], s[bi], e[bi], p[bi], hs[bi], N, **dk)
    return out, raw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_splits", default="train,validation"); ap.add_argument("--eval_split", default="test"); ap.add_argument("--predict_split", default="challenge")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="tv3"); ap.add_argument("--sample_sub", default="")
    ap.add_argument("--d", type=int, default=192); ap.add_argument("--layers", type=int, default=3); ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--ls", type=float, default=0.05); ap.add_argument("--w_bnd", type=float, default=2.0); ap.add_argument("--w_set", type=float, default=1.0); ap.add_argument("--nq", type=int, default=6)
    ap.add_argument("--img_feats", default=""); ap.add_argument("--in_drop", type=float, default=0.0)
    ap.add_argument("--predict_only", default="", help="load an existing model.pt and only predict --predict_split, writing raw (Docker inference; does not read the training set)")
    args = ap.parse_args(); os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    load = lambda sp: [r for r in pickle.load(open(os.path.join(args.meta_dir,f"meta_{sp}.pkl"),"rb")) if "error" not in r]
    fd = args.img_feats or None
    tr = [] if args.predict_only else sum([load(s) for s in args.train_splits.split(",")], []); ev = [] if args.predict_only else (load(args.eval_split) if args.eval_split else [])
    if args.predict_only:
        pdx = load(args.predict_split); ds_p = DS(pdx, with_targets=False, feats_dir=fd)
        model = Net3(ds_p.items[0][1].shape[1], nq=args.nq, d=args.d, layers=args.layers, heads=max(1,args.d//32), drop=args.drop, in_drop=args.in_drop).to(dev)
        model.load_state_dict(torch.load(args.predict_only, map_location=dev)); model.eval()
        pr, raw = predict(model, ds_p, dev, k_runs=3, thr=0.4, use_has=False, snap_se=True)
        pickle.dump(raw, open(os.path.join(args.out_dir, f"raw_{args.predict_split}.pkl"), "wb")); print(f"wrote raw {args.predict_split} (predict-only, {len(pdx)} clips)", flush=True); return
    t0 = time.time(); ds_tr = DS(tr, feats_dir=fd); ds_ev = DS(ev, feats_dir=fd) if ev else None; print(f"T={bm.T} features built in {time.time()-t0:.1f}s fdim={ds_tr.items[0][1].shape[1]} train {len(tr)} eval {len(ev)}", flush=True)
    dl = torch.utils.data.DataLoader(ds_tr, batch_size=args.bs, shuffle=True, drop_last=True)
    dl_ev = torch.utils.data.DataLoader(ds_ev, batch_size=args.bs) if ds_ev else None
    model = Net3(ds_tr.items[0][1].shape[1], nq=args.nq, d=args.d, layers=args.layers, heads=max(1,args.d//32), drop=args.drop, in_drop=args.in_drop).to(dev)
    model = train(model, dl, dl_ev, args.epochs, args.lr, dev, args.ls, args.w_bnd, args.w_set)
    torch.save(model.state_dict(), os.path.join(args.out_dir,"model.pt"))
    json.dump(dict(type="tv3", d=args.d, layers=args.layers, nq=args.nq), open(os.path.join(args.out_dir,"config.json"),"w"))
    if ds_ev:
        gts = gts_from_meta(ev); fac = factors_from_meta(ev)
        pr, raw = predict(model, ds_ev, dev, k_runs=3, thr=0.4, use_has=False, snap_se=True)
        print(f"[eval {args.eval_split}] frame-decode snap:", evaluate(gts, pr, fac), flush=True)
        # add query candidates (dedup: for tIoU > 0.7 with an existing candidate keep the higher score)
        pr2 = {}
        for sid, cs in pr.items():
            cs2 = list(cs)
            for q in raw[sid]["queries"]:
                dup = any(_tiou(q[1],q[2],c[1],c[2]) > 0.7 for c in cs)
                cs2.append((q[0]*(0.6 if dup else 1.0),) + tuple(q[1:]))
            pr2[sid] = cs2
        print(f"[eval {args.eval_split}] + queries:", evaluate(gts, pr2, fac), flush=True)
        pickle.dump(raw, open(os.path.join(args.out_dir,f"raw_{args.eval_split}.pkl"),"wb"))
    if args.predict_split:
        pdx = load(args.predict_split); ds_p = DS(pdx, with_targets=False, feats_dir=fd)
        pr, raw = predict(model, ds_p, dev, k_runs=3, thr=0.4, use_has=False, snap_se=True)
        pickle.dump(raw, open(os.path.join(args.out_dir,f"raw_{args.predict_split}.pkl"),"wb")); print("wrote raw challenge", flush=True)

def _tiou(s1,e1,s2,e2):
    inter = max(0.0, min(e1,e2)-max(s1,s2)); union = max(e1,e2)-min(s1,s2)
    return inter/union if union > 0 else 0.0

if __name__ == "__main__":
    main()
