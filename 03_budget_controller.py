"""Step 3: telemetry-driven online budget controller (primal-dual threshold) on a
non-stationary request stream built from the held-out test split of pair P1.

Stream: T requests in four phases whose corruption mix changes (clean-dominated ->
degraded-dominated -> mixed -> clean), i.e. the score distribution drifts.
Budget: B(t) = C_l + (C_h - C_l) * min(rho_E(t), rho_T(t)) from a SYNTHETIC telemetry
trace (battery derating + thermal derating); the trace is an assumption of the simulation,
not a hardware measurement.

Controllers (all use the SAME proposed gain score, so only the control law differs)
  static  : one threshold calibrated offline for the mean budget
  lookup  : offline budget->threshold table tau(B) from the validation split (telemetry-aware
            but distribution-blind; this is the open-loop rule of version 1)
  dual    : proposed  lam <- [lam + mu * (c_t - B_t)/C_h]_+ , route heavy iff score > lam
"""
import os, json
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, 'results')
T, MU, SEEDS = 20000, 0.02, 20


def telemetry(T):
    t = np.arange(T) / T
    rho_e = np.where(t < 0.75, 0.70, 0.70 - 1.6 * (t - 0.75))            # battery sag in the last quarter
    temp = 55 + 25 * np.exp(-((t - 0.40) / 0.10) ** 2)                    # die-temperature excursion (deg C)
    rho_t = np.clip((85.0 - temp) / (85.0 - 60.0), 0.05, 1.0)             # linear derating 60..85 C
    return rho_e, temp, rho_t, np.minimum(rho_e, rho_t)


def main():
    z = np.load(os.path.join(OUT, 'scores_P1.npz')); st = np.load(os.path.join(OUT, 'stream.npz'))
    ms = json.load(open(os.path.join(OUT, 'model_stats.json')))
    cl = ms['shufflenetv2_x0_5']['flops']; ch = ms['Hendrycks2020AugMix_WRN']['flops']
    s, al, ah, role, sev = z['prop'], z['al'], z['ah'], z['role'], st['sev']
    va = np.nonzero(role == 1)[0]
    pools = [np.nonzero((role == 2) & (sev == k))[0] for k in range(6)]
    # phase-wise severity mix (probabilities of severity 0..5)
    mix = [(0.00, [.70, .15, .10, .05, 0, 0]), (0.25, [.05, .05, .10, .20, .30, .30]),
           (0.55, [.30, .15, .15, .15, .15, .10]), (0.80, [.70, .15, .10, .05, 0, 0])]
    rho_e, temp, rho_t, rho = telemetry(T)
    B = (cl + (ch - cl) * rho) / ch                       # normalised budget per request
    qs = np.linspace(0, 1, 201); tab = np.quantile(s[va], 1 - qs)   # heavy share q -> threshold
    share_of = lambda b: np.clip((b * ch - cl) / (ch - cl), 0, 1)
    R = {k: dict(acc=[], viol=[], maxrun=[]) for k in ('static', 'lookup', 'dual')}
    keep = {}
    for sd in range(SEEDS):
        rng = np.random.default_rng(sd)
        ph = np.searchsorted([m[0] for m in mix], np.arange(T) / T, side='right') - 1
        ids = np.array([rng.choice(pools[rng.choice(6, p=mix[p][1])]) for p in ph])
        sc, a_l, a_h = s[ids], al[ids], ah[ids]
        heavy = {}
        heavy['static'] = sc > np.interp(share_of(B.mean()), qs, tab)
        heavy['lookup'] = sc > np.interp(share_of(B), qs, tab)
        lam = float(np.interp(share_of(B[0]), qs, tab)); h = np.zeros(T, bool); lams = np.zeros(T)
        for t in range(T):
            h[t] = sc[t] > lam; lams[t] = lam
            c = (ch if h[t] else cl) / ch
            lam = max(0.0, lam + MU * (c - B[t]))
        heavy['dual'] = h
        for k, hv in heavy.items():
            c = np.where(hv, ch, cl) / ch
            R[k]['acc'].append(float(np.where(hv, a_h, a_l).mean()))
            R[k]['viol'].append(float((c - B).sum() / B.sum()))            # relative cumulative overrun
            w = 500; run = np.convolve(c - B, np.ones(w) / w, 'valid')
            R[k]['maxrun'].append(float(run.max()))
            if sd == 0:
                keep[k] = np.convolve(c, np.ones(w) / w, 'same').tolist()
        if sd == 0:
            keep['lam'] = lams.tolist()
            keep['bound'] = float((1.0 + MU) / MU)        # Proposition 3: sum(c_t - B_t)/C_h <= (s_max + mu)/mu with s_max = 1
            keep['dual_cum'] = float((np.where(h, ch, cl) / ch - B).sum())
            keep['ph_acc'] = {k: [float(np.where(heavy[k], a_h, a_l)[ph == p].mean()) for p in range(4)] for k in heavy}
            keep['ph_cost'] = {k: [float((np.where(heavy[k], ch, cl) / ch)[ph == p].mean()) for p in range(4)] for k in heavy}
            keep['ph_B'] = [float(B[ph == p].mean()) for p in range(4)]
    summ = {k: {m: [float(np.mean(v)), float(np.std(v))] for m, v in d.items()} for k, d in R.items()}
    json.dump(dict(T=T, mu=MU, B=B.tolist(), temp=temp.tolist(), rho_e=rho_e.tolist(), rho_t=rho_t.tolist(),
                   trace=keep, summary=summ, phases=[m[0] for m in mix]),
              open(os.path.join(OUT, 'controller_results.json'), 'w'))
    for k, v in summ.items(): print(k, v)
    print('dual cumulative overrun (requests of heavy-cost): %.2f  <= bound %.1f' % (keep['dual_cum'], keep['bound']))
    print('phase B', keep['ph_B']); print('phase cost', keep['ph_cost']); print('phase acc', keep['ph_acc'])


if __name__ == '__main__':
    main()
