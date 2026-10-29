"""Step 2: train the zero-overhead gates and evaluate every routing policy.

Protocol (no leakage): the 10,000 CIFAR-100 test IMAGES are split by image index into
gate-train (4,000) / validation (1,000) / held-out test (5,000); all corrupted copies of an
image stay in the same split.  Backbones are frozen.  Repeated over N_SEEDS random splits.

Policies
  proposed  : gain-aware ZOCE  -- 10 descriptors, quadratic gate, score = P(heavy ok) - P(light ok)
  v1        : version-1 ZOCE   -- 3 descriptors (entropy, variance, gradient), linear, P(light ok)
  blind     : data-blind (system-state only) switching = random routing at the same ratio
  cnn       : learned deep gate (3-conv CNN on the raw image, same gain target), pays its FLOPs
  cascade   : post-execution confidence cascade (run light, escalate if max-softmax < thr)
  triage    : proposed ZOCE-T -- the pre-execution score picks an execution PLAN: bypass straight
              to heavy (light never runs) or run light and verify with a descriptor+confidence gate
  oracle    : genie pre-execution router (heavy only where it flips a light error)
"""
import os, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import zoce_features as zf

ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, 'results')
PAIRS = {'P1': ('shufflenetv2_x0_5', 'Hendrycks2020AugMix_WRN'),     # C_l/C_h = 0.036
         'P2': ('resnet20', 'Hendrycks2020AugMix_WRN'),              # C_l/C_h = 0.126
         'P3': ('mobilenetv2_x1_4', 'Hendrycks2020AugMix_WRN'),      # C_l/C_h = 0.543
         'P4': ('mobilenetv2_x0_5', 'Modas2021PRIMEResNet18'),       # C_l/C_h = 0.057
         'N1': ('resnet20', 'resnet56')}                             # same-family control pair
CNN_PAIRS = ('P1', 'P2', 'P4')             # the learned deep-gate baseline is trained for these pairs
N_SEEDS = 5
GRID = np.linspace(0, 1, 41)            # fraction of samples sent to the heavy backbone


def expand(F, mu=None, sd=None, quad=True):
    F = F.copy()
    for j in (1, 2, 3, 4, 8):           # positive, heavy-tailed descriptors -> log
        if j < F.shape[1]: F[:, j] = np.log1p(F[:, j])
    if mu is None: mu, sd = F.mean(0), F.std(0) + 1e-6
    Z = (F - mu) / sd
    if quad:
        iu = np.triu_indices(Z.shape[1])
        Z = np.concatenate([Z, (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]], 1)
    return Z, mu, sd


def fit_lr(Z, t, C=0.3):
    return LogisticRegression(C=C, max_iter=3000).fit(Z, t)


def curve(score, al, ah, cl, ch, cg=0.0, grid=GRID):
    """route the top-q fraction (by score) to the heavy model; cost normalised by C_heavy"""
    order = np.argsort(-score, kind='stable'); n = len(score)
    acc, cost = [], []
    for q in grid:
        k = int(round(q * n)); heavy = np.zeros(n, bool); heavy[order[:k]] = True
        acc.append(np.where(heavy, ah, al).mean())
        cost.append((cg + (1 - q) * cl + q * ch) / ch)
    return np.array(cost), np.array(acc)


def cascade_curve(conf, al, ah, cl, ch, grid=GRID):
    order = np.argsort(conf, kind='stable'); n = len(conf)   # least confident escalate first
    acc, cost = [], []
    for q in grid:
        k = int(round(q * n)); esc = np.zeros(n, bool); esc[order[:k]] = True
        acc.append(np.where(esc, ah, al).mean()); cost.append((cl + q * ch) / ch)
    return np.array(cost), np.array(acc)


def triage_curve(s0, sf, conf, al, ah, cl, ch, cg, va, te, budgets):
    """ZOCE-T.  Plan parameters (bypass share a0, verify share q) are chosen on the validation
    split for each budget and then frozen; the curve reports the held-out test split."""
    cand = []
    for a0 in np.arange(0, 0.55, 0.05):
        t0 = np.quantile(s0[va], 1 - a0) if a0 > 0 else np.inf
        direct = s0 > t0; rest = ~direct
        for q in np.linspace(0, 1, 41):
            t1 = np.quantile(sf[va & rest], 1 - q) if 0 < q < 1 else (np.inf if q == 0 else -np.inf)
            heavy = direct | (rest & (sf > t1))
            r = []
            for m in (va, te):
                r += [(cg + rest[m].mean() * cl + heavy[m].mean() * ch) / ch, np.where(heavy, ah, al)[m].mean()]
            cand.append(r + [direct[te].mean()])
    cand = np.array(cand); cost, acc, byp = [], [], []
    for b in budgets:
        ok = np.nonzero(cand[:, 0] <= b)[0]
        if len(ok) == 0: continue
        j = ok[np.argmax(cand[ok, 1])]
        cost.append(cand[j, 2]); acc.append(cand[j, 3]); byp.append(cand[j, 4])
    return np.array(cost), np.array(acc), np.array(byp)


def acc_at(cost, acc, b):
    """best accuracy achievable with average cost <= b (interpolated, monotone envelope)"""
    o = np.argsort(cost); c, a = cost[o], np.maximum.accumulate(acc[o])
    return float(np.interp(b, c, a)) if b >= c[0] else float('nan')


def cost_for(cost, acc, target):
    o = np.argsort(cost); c, a = cost[o], np.maximum.accumulate(acc[o])
    i = np.nonzero(a >= target)[0]
    if len(i) == 0: return float('nan')
    i = i[0]
    if i == 0: return float(c[0])
    return float(c[i - 1] + (target - a[i - 1]) / (a[i] - a[i - 1] + 1e-12) * (c[i] - c[i - 1]))


def train_cnn(x, al, ah, tr, seed):
    import torch, torch.nn as nn
    from thop import profile
    torch.manual_seed(seed); torch.set_num_threads(2)
    net = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
                        nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                        nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2))
    X = torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255).sub(0.5).div(0.25)
    T = torch.from_numpy(np.stack([al, ah], 1).astype(np.float32))
    idx = np.nonzero(tr)[0]; ep = 8
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=ep * (len(idx) // 128 + 1))
    rng = np.random.default_rng(seed)
    for e in range(ep):
        net.train(); rng.shuffle(idx)
        for b in range(0, len(idx), 128):
            j = idx[b:b + 128]; xb = X[j]
            if rng.random() < 0.5: xb = xb.flip(3)
            loss = nn.functional.binary_cross_entropy_with_logits(net(xb), T[j])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad():
        p = torch.cat([torch.sigmoid(net(X[b:b + 2000])) for b in range(0, len(X), 2000)]).numpy()
    fl, _ = profile(net, inputs=(torch.zeros(1, 3, 32, 32),), verbose=False)
    return p[:, 1] - p[:, 0], float(fl)


def main():
    st = np.load(os.path.join(OUT, 'stream.npz'))
    x, idx, cid, sev = st['x'], st['idx'], st['cid'], st['sev']
    ms = json.load(open(os.path.join(OUT, 'model_stats.json')))
    fF = os.path.join(OUT, 'features.npy')
    if not os.path.exists(fF): np.save(fF, zf.features(x))
    F = np.load(fF)
    # v10: full gate count (standardization, logs, products, both heads, sigmoids); the saved results in
    # results/ were produced with the earlier count (FULL_OPS + 187). The difference is below 400 ops,
    # i.e. about 1e-6 of C_h, and does not change any reported number.
    GATE_OPS = zf.FULL_OPS + zf.GATE_HEAD_OPS
    V1_OPS = zf.V1_OPS + zf.V1_GATE_OPS
    res = {'grid': GRID.tolist(), 'ops': {'proposed': GATE_OPS, 'v1': V1_OPS, 'per_feature': zf.OPS}}
    use_cnn = '--no-cnn' not in sys.argv
    for pk, (ln, hn) in PAIRS.items():
        al = np.load(os.path.join(OUT, 'preds_%s.npz' % ln))['correct'].astype(np.float32)
        conf = np.load(os.path.join(OUT, 'preds_%s.npz' % ln))['conf']
        ah = np.load(os.path.join(OUT, 'preds_%s.npz' % hn))['correct'].astype(np.float32)
        cl, ch = ms[ln]['flops'], ms[hn]['flops']
        R = {'light': ln, 'heavy': hn, 'cl': cl, 'ch': ch, 'seeds': []}
        for s in range(N_SEEDS):
            perm = np.random.default_rng(100 + s).permutation(10000)
            role = np.zeros(10000, int); role[perm[4000:5000]] = 1; role[perm[5000:]] = 2
            tr, va, te = role[idx] == 0, role[idx] == 1, role[idx] == 2
            out = {}
            # ---------- gates
            Zq, mu, sd = expand(F[tr]); Zq_all = expand(F, mu, sd)[0]
            Zl, mu1, sd1 = expand(F[tr], quad=False); Zl_all = expand(F, mu1, sd1, quad=False)[0]
            Z3, mu3, sd3 = expand(F[tr][:, :3], quad=False); Z3_all = expand(F[:, :3], mu3, sd3, quad=False)[0]
            sc = {}
            pl = fit_lr(Zq, al[tr]).predict_proba(Zq_all)[:, 1]
            ph = fit_lr(Zq, ah[tr]).predict_proba(Zq_all)[:, 1]
            sc['proposed'] = ph - pl
            sc['v1'] = -fit_lr(Z3, al[tr]).predict_proba(Z3_all)[:, 1]            # hard = not easy
            # ablations
            sc['abl_v1feat_gain'] = (fit_lr(Z3, ah[tr]).predict_proba(Z3_all)[:, 1]
                                     - fit_lr(Z3, al[tr]).predict_proba(Z3_all)[:, 1])
            sc['abl_full_easy'] = -pl
            sc['abl_full_gain_lin'] = (fit_lr(Zl, ah[tr]).predict_proba(Zl_all)[:, 1]
                                       - fit_lr(Zl, al[tr]).predict_proba(Zl_all)[:, 1])
            rng = np.random.default_rng(s); sc['blind'] = rng.random(len(al))
            sc['oracle'] = (ah - al) + 1e-3 * rng.random(len(al))
            cond = (cid + 1) * 6 + sev; gm = np.zeros(cond.max() + 1)   # knows the true corruption+severity
            for c_ in np.unique(cond): gm[c_] = (ah - al)[tr & (cond == c_)].mean()
            sc['cond_oracle'] = gm[cond] + 1e-6 * rng.random(len(al))
            cg = {'cond_oracle': 0, 'proposed': GATE_OPS, 'v1': V1_OPS, 'abl_v1feat_gain': V1_OPS + 4,
                  'abl_full_easy': GATE_OPS, 'abl_full_gain_lin': zf.FULL_OPS + 22, 'blind': 0, 'oracle': 0}
            if use_cnn and pk in CNN_PAIRS:
                sc['cnn'], cg['cnn'] = train_cnn(x, al, ah, tr, s)
            ben = (ah > al)                                  # samples where escalation pays off
            base, top = al[te].mean(), ah[te].mean()
            for k, v in sc.items():
                c, a = curve(v[te], al[te], ah[te], cl, ch, cg[k])
                out[k] = dict(cost=c.tolist(), acc=a.tolist(), gate_ops=cg[k],
                              auc_gain=float(roc_auc_score(ben[te], v[te])),
                              acc50=acc_at(c, a, 0.5), acc30=acc_at(c, a, 0.3),
                              cost_1pt=cost_for(c, a, top - 0.01),
                              aurc=float(np.sum((a[1:] + a[:-1]) / 2 * np.diff(GRID))))
            c, a = cascade_curve(conf[te], al[te], ah[te], cl, ch)
            lc = np.log(conf / (1 - conf + 1e-6))[:, None]
            Zf = np.column_stack([Zq_all, lc, lc ** 2, Zq_all[:, :10] * lc])
            sf = fit_lr(Zf[tr], ah[tr]).predict_proba(Zf)[:, 1] - fit_lr(Zf[tr], al[tr]).predict_proba(Zf)[:, 1]
            ct, at, bt = triage_curve(sc['proposed'], sf, conf, al, ah, cl, ch, GATE_OPS + zf.VERIFY_EXTRA_OPS, va, te,
                                      np.linspace((cl + GATE_OPS) / ch, 1 + cl / ch, 41))
            out['triage'] = dict(cost=ct.tolist(), acc=at.tolist(), bypass=bt.tolist(), gate_ops=GATE_OPS + zf.VERIFY_EXTRA_OPS,
                                 auc_gain=float(roc_auc_score(ben[te], sf[te])),
                                 acc50=acc_at(ct, at, 0.5), acc30=acc_at(ct, at, 0.3),
                                 cost_1pt=cost_for(ct, at, top - 0.01))
            out['cascade'] = dict(cost=c.tolist(), acc=a.tolist(), gate_ops=cl,
                                  auc_gain=float(roc_auc_score(ben[te], -conf[te])),
                                  acc50=acc_at(c, a, 0.5), acc30=acc_at(c, a, 0.3),
                                  cost_1pt=cost_for(c, a, top - 0.01))
            out['_light_acc'], out['_heavy_acc'] = float(base), float(top)
            # ---------- per-severity behaviour of the proposed gate at 50 % heavy share
            thr = np.quantile(sc['proposed'][va], 0.5)
            hv = sc['proposed'] > thr; out['_thr50'] = float(thr); hv1 = sc['v1'] > np.quantile(sc['v1'][va], 0.5)
            bysev = []
            for sv in range(6):
                m = te & (sev == sv)
                bysev.append(dict(sev=sv, n=int(m.sum()), light=float(al[m].mean()), heavy=float(ah[m].mean()),
                                  gain=float((ah[m] - al[m]).mean()), share_prop=float(hv[m].mean()),
                                  share_v1=float(hv1[m].mean()),
                                  acc_prop=float(np.where(hv[m], ah[m], al[m]).mean()),
                                  acc_v1=float(np.where(hv1[m], ah[m], al[m]).mean())))
            out['_bysev'] = bysev
            # clean-only sub-stream (honest sanity check)
            m = te & (cid < 0)
            out['_clean_auc'] = {k: float(roc_auc_score(ben[m], sc[k][m])) for k in ('proposed', 'v1')}
            out['_clean_auc']['cascade'] = float(roc_auc_score(ben[m], -conf[m]))
            R['seeds'].append(out)
            if s == 0:
                np.savez_compressed(os.path.join(OUT, 'scores_%s.npz' % pk), prop=sc['proposed'], v1=sc['v1'],
                                    role=role[idx], al=al, ah=ah, conf=conf)
            print(pk, s, {k: (round(v['auc_gain'], 3), round(v['acc50'], 4), round(v['cost_1pt'], 3))
                          for k, v in out.items() if not k.startswith('_')}, flush=True)
        res[pk] = R
        json.dump(res, open(os.path.join(OUT, 'gate_results.json'), 'w'))


if __name__ == '__main__':
    main()
