#!/usr/bin/env python3
"""RAC — temporal model v2: 64 slots (no truncation of samples with >32 frames), boundary regression head (frame distance to start/end), label smoothing.
Reuses the features/targets/decoding of baseline_meta; raw additionally stores ds/de for reeval --bnd.
Usage as baseline_meta.py (plus --ls and --w_bnd).
"""
import argparse, os, json, pickle, time, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import baseline_meta as bm
bm.T = 64
from baseline_meta import DS, Net, masked_bce, decode, write_submission
from metric import evaluate, factors_from_meta, gts_from_meta

class Net2(Net):
    def __init__(self, fdim, **kw):
        super().__init__(fdim, **kw)
        d = self.head_pts.in_features
        self.head_bnd = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
    def forward(self, X, valid):
        h = self.inp(X) + self.pos
        h = self.enc(h, src_key_padding_mask=(valid<0.5))
        a = self.head_act(h); p = torch.sigmoid(self.head_pts(h)); b = F.softplus(self.head_bnd(h))
        m = valid.unsqueeze(-1); g = (h*m).sum(1)/m.sum(1).clamp(min=1)
        return a[...,0], a[...,1], a[...,2], p, self.head_has(g)[...,0], b

def bnd_targets(act):
    """act: [B,T] 0/1 -> ds, de (frames/32) and mask, per run."""
    B, T = act.shape; ds = torch.zeros_like(act); de = torch.zeros_like(act)
    a = act.cpu().numpy()
    for bi in range(B):
        i = 0
        while i < T:
            if a[bi,i] > 0.5:
                j = i
                while j+1 < T and a[bi,j+1] > 0.5: j += 1
                for k in range(i, j+1): ds[bi,k] = (k-i)/32.0; de[bi,k] = (j-k)/32.0
                i = j+1
            else: i += 1
    return ds.to(act.device), de.to(act.device)

def train(model, dl, dl_val, epochs, lr, dev, ls, w_bnd):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs*len(dl), pct_start=0.1)
    for ep in range(epochs):
        model.train(); tl = 0; n = 0
        for b in dl:
            b = {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in b.items()}
            a,s,e,p,hs,bd = model(b["X"], b["valid"])
            act_t = b["act"]*(1-ls) + 0.5*ls
            loss = masked_bce(a,act_t,b["valid"],1.5) + 0.5*masked_bce(s,b["st"],b["valid"],4.0) + 0.5*masked_bce(e,b["en"],b["valid"],4.0)
            lp = (F.smooth_l1_loss(p,b["pts"],reduction="none",beta=0.02).sum(-1)*b["pm"]).sum()/b["pm"].sum().clamp(min=1)
            ds, de = bnd_targets(b["act"])
            lb = (F.smooth_l1_loss(bd[...,0], ds, reduction="none", beta=0.03) + F.smooth_l1_loss(bd[...,1], de, reduction="none", beta=0.03))
            lb = (lb*b["act"]).sum()/b["act"].sum().clamp(min=1)
            loss = loss + 10.0*lp + 0.5*F.binary_cross_entropy_with_logits(hs,b["has"]) + w_bnd*lb
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl += loss.item()*len(b["X"]); n += len(b["X"])
        msg = f"ep {ep+1}/{epochs} train_loss {tl/n:.4f}"
        if dl_val is not None and ((ep+1)%10==0 or ep+1==epochs):
            model.eval(); vl=0; vn=0
            with torch.no_grad():
                for b in dl_val:
                    b = {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in b.items()}
                    a,s,e,p,hs,bd = model(b["X"], b["valid"]); vl += masked_bce(a,b["act"],b["valid"]).item()*len(b["X"]); vn += len(b["X"])
            msg += f" | val_act_bce {vl/vn:.4f}"
        print(msg, flush=True)
    return model

def predict(model, ds, dev, bs=256, **dk):
    model.eval(); out = {}; raw = {}
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False); k = 0
    with torch.no_grad():
        for b in dl:
            X = b["X"].to(dev); valid = b["valid"].to(dev)
            a,s,e,p,hs,bd = model(X, valid)
            a = torch.sigmoid(a).cpu(); s = torch.sigmoid(s).cpu(); e = torch.sigmoid(e).cpu(); p = p.cpu(); hs = torch.sigmoid(hs).cpu(); bd = bd.cpu()
            for bi in range(len(X)):
                sid = ds.items[k][0]; N = ds.items[k][3]; k += 1
                raw[sid] = dict(a=a[bi,:N].numpy(), s=s[bi,:N].numpy(), e=e[bi,:N].numpy(), pts=p[bi,:N].numpy(), has=float(hs[bi]), N=N,
                                ds=bd[bi,:N,0].numpy(), de=bd[bi,:N,1].numpy())
                out[sid] = decode(a[bi], s[bi], e[bi], p[bi], hs[bi], N, bnd=(raw[sid]["ds"], raw[sid]["de"]), **dk)
    return out, raw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_splits", default="train,validation"); ap.add_argument("--eval_split", default="test"); ap.add_argument("--predict_split", default="challenge")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="tv2"); ap.add_argument("--sample_sub", default="")
    ap.add_argument("--d", type=int, default=192); ap.add_argument("--layers", type=int, default=3); ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--ls", type=float, default=0.05); ap.add_argument("--w_bnd", type=float, default=2.0)
    args = ap.parse_args(); os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    load = lambda sp: [r for r in pickle.load(open(os.path.join(args.meta_dir,f"meta_{sp}.pkl"),"rb")) if "error" not in r]
    tr = sum([load(s) for s in args.train_splits.split(",")], []); ev = load(args.eval_split) if args.eval_split else []
    t0 = time.time(); ds_tr = DS(tr); ds_ev = DS(ev) if ev else None; print(f"T={bm.T} features built in {time.time()-t0:.1f}s fdim={ds_tr.items[0][1].shape[1]} train {len(tr)} eval {len(ev)}", flush=True)
    dl = torch.utils.data.DataLoader(ds_tr, batch_size=args.bs, shuffle=True, drop_last=True)
    dl_ev = torch.utils.data.DataLoader(ds_ev, batch_size=args.bs) if ds_ev else None
    model = Net2(ds_tr.items[0][1].shape[1], d=args.d, layers=args.layers, heads=max(1,args.d//32), drop=args.drop).to(dev)
    model = train(model, dl, dl_ev, args.epochs, args.lr, dev, args.ls, args.w_bnd)
    torch.save(model.state_dict(), os.path.join(args.out_dir,"model.pt"))
    if ds_ev:
        gts = gts_from_meta(ev); fac = factors_from_meta(ev)
        pr, raw = predict(model, ds_ev, dev, k_runs=3, thr=0.4, use_has=False, snap_se=True)
        print(f"[eval {args.eval_split}] bnd+snap k3 t0.4 nohas:", evaluate(gts, pr, fac), flush=True)
        pr0, _ = predict(model, ds_ev, dev, k_runs=3, thr=0.4, use_has=False, snap_se=False)
        pickle.dump(raw, open(os.path.join(args.out_dir,f"raw_{args.eval_split}.pkl"),"wb"))
    if args.predict_split:
        pdx = load(args.predict_split); ds_p = DS(pdx, with_targets=False)
        pr, raw = predict(model, ds_p, dev, k_runs=3, thr=0.4, use_has=False, snap_se=True)
        pickle.dump(raw, open(os.path.join(args.out_dir,f"raw_{args.predict_split}.pkl"),"wb"))
        import csv
        ids = [r["id"] for r in csv.DictReader(open(args.sample_sub))] if args.sample_sub else None
        write_submission(pr, os.path.join(args.out_dir,f"submission_{args.tag}.csv"), ids); print("wrote submission", flush=True)

if __name__ == "__main__":
    main()
