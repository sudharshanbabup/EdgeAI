"""Step 4: aggregate results over the 5 splits, write
   matlab/data/*.json   (panel descriptions for matlab_fig_1.m)
   results/summary.json and ../latex_4/results_macros.tex (every number quoted in the paper)  [v12: writes to ../latex_12]."""
import os, json, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, 'results')
MD = os.path.join(ROOT, 'matlab', 'data'); os.makedirs(MD, exist_ok=True)
TEX = os.path.join(ROOT, '..', 'latex_12'); os.makedirs(TEX, exist_ok=True)
r = json.load(open(os.path.join(OUT, 'gate_results.json')))
ms = json.load(open(os.path.join(OUT, 'model_stats.json')))
ct = json.load(open(os.path.join(OUT, 'controller_results.json')))
lat = json.load(open(os.path.join(OUT, 'latency.json'))) if os.path.exists(os.path.join(OUT, 'latency.json')) else None
NICE = {'shufflenetv2_x0_5': 'ShuffleNetV2-0.5', 'mobilenetv2_x0_5': 'MobileNetV2-0.5', 'resnet20': 'ResNet-20',
        'resnet56': 'ResNet-56', 'mobilenetv2_x1_4': 'MobileNetV2-1.4',
        'Hendrycks2020AugMix_WRN': 'AugMix-WRN-40-2', 'Modas2021PRIMEResNet18': 'PRIME-ResNet-18'}
# one colour / marker / line per scheme, reused in every figure
PAL = {'triage': dict(label='ZOCE T (proposed)', color=[0.000, 0.447, 0.698], marker='o', line='-', proposed=True),
       'proposed': dict(label='ZOCE R (proposed)', color=[0.337, 0.706, 0.914], marker='s', line='-', proposed=True),
       'cascade': dict(label='Confidence cascade', color=[0.835, 0.369, 0.000], marker='d', line='--'),
       'cnn': dict(label='CNN gate', color=[0.000, 0.620, 0.451], marker='^', line='-.'),
       'v1': dict(label='Version 1 gate', color=[0.482, 0.196, 0.580], marker='v', line=':'),
       'blind': dict(label='Data blind', color=[0.350, 0.350, 0.350], marker='p', line=':')}
json.dump(PAL, open(os.path.join(ROOT, 'matlab', 'palette.json'), 'w'), indent=1)


def env(c, a, b):
    c, a = np.array(c), np.array(a); o = np.argsort(c); c, a = c[o], np.maximum.accumulate(a[o])
    return np.where(b >= c[0] - 1e-9, np.interp(b, c, a), np.nan)


def agg(pk, k, m):
    v = [s[k][m] for s in r[pk]['seeds'] if k in s]
    return (float(np.nanmean(v)), float(np.nanstd(v))) if v else (float('nan'), float('nan'))


def acc_b(pk, k, b):
    v = [float(env(s[k]['cost'], s[k]['acc'], np.array([b]))[0]) for s in r[pk]['seeds'] if k in s]
    return (float(np.mean(v)), float(np.std(v))) if v and not np.isnan(v).any() else (float('nan'), float('nan'))


S = {'ops': r['ops'], 'pairs': {}}
# ---------------- Fig. 2: accuracy vs. average compute
for pk, tag in (('P2', 'a'), ('P4', 'b')):
    R = r[pk]; b = np.round(np.arange(0.15, 1.001, 0.085), 3)
    ser = []
    for k in ('triage', 'proposed', 'cascade', 'cnn', 'v1', 'blind'):
        if k not in R['seeds'][0]: continue
        y = np.mean([env(s[k]['cost'], s[k]['acc'], b) for s in R['seeds']], 0) * 100
        ok = ~np.isnan(y)
        ser.append(dict(PAL[k], x=b[ok].tolist(), y=np.round(y[ok], 3).tolist()))
    json.dump(dict(name='fig2%s_frontier_%s' % (tag, pk), layout='pair',
                   xlabel='Average compute per input (normalized to heavy model)', ylabel='Stream accuracy (%)',
                   xlim=[0.12, 1.03], xticks=[0.2, 0.4, 0.6, 0.8, 1.0], xtickformat='%.1f', legend='southeast',
                   series=ser), open(os.path.join(MD, 'fig2%s_frontier_%s.json' % (tag, pk)), 'w'), indent=1)
# ---------------- Fig. 3: what the gate does per severity (P2, 50 % heavy share)
pk = 'P2'; bs = r[pk]['seeds']
g = lambda m: [float(np.mean([s['_bysev'][j][m] for s in bs])) * 100 for j in range(6)]
S['bysev'] = {m: g(m) for m in ('light', 'heavy', 'gain', 'share_prop', 'share_v1', 'acc_prop', 'acc_v1')}
json.dump(dict(name='fig3_severity_share', layout='col', xlabel='Corruption severity (0 = clean)',
               ylabel='Share sent to heavy model (%)', xlim=[-0.3, 5.3], ylim=[0, 100],
               xticks=[0, 1, 2, 3, 4, 5], legend='northwest',
               series=[dict(PAL['proposed'], x=list(range(6)), y=g('share_prop')),
                       dict(PAL['v1'], x=list(range(6)), y=g('share_v1')),
                       dict(label='Measured gain (pts)', x=list(range(6)), y=g('gain'), color=[0.902, 0.624, 0.000],
                            marker='h', line='--')]),
          open(os.path.join(MD, 'fig3_severity_share.json'), 'w'), indent=1)
# ---------------- Fig. 5 + descriptor table: per-family behaviour and single-descriptor AUC (P2, split 0)
from sklearn.metrics import roc_auc_score
import zoce_features as zf
st = np.load(os.path.join(OUT, 'stream.npz')); z2 = np.load(os.path.join(OUT, 'scores_P2.npz'))
F = np.load(os.path.join(OUT, 'features.npy')); corr = [str(c) for c in st['corr']]
FAM = {'Noise': ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise'],
       'Blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur'],
       'Weather': ['snow', 'frost', 'fog', 'brightness', 'spatter'],
       'Digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'saturate']}
te2, va2 = z2['role'] == 2, z2['role'] == 1
hv = z2['prop'] > np.quantile(z2['prop'][va2], 0.5)
fam = {}
for nm, members in [('Clean', None)] + list(FAM.items()):
    m = te2 & ((st['cid'] < 0) if members is None else np.isin(st['cid'], [corr.index(c) for c in members]))
    fam[nm] = dict(share=float(100 * hv[m].mean()), gain=float(100 * (z2['ah'][m] - z2['al'][m]).mean()),
                   light=float(100 * z2['al'][m].mean()), heavy=float(100 * z2['ah'][m].mean()))
S['family'] = fam
xs = np.arange(5)
json.dump(dict(name='fig5_family_share', layout='col', xlabel='Input condition', ylabel='Percent',
               xlim=[-0.6, 4.6], ylim=[0, 100], xticks=xs.tolist(), xticklabels=list(fam), legend='northeast',
               series=[dict(kind='bar', label='Share sent to heavy by ZOCE R (%)', color=PAL['proposed']['color'], width=0.36,
                            x=(xs - 0.19).tolist(), y=[fam[k]['share'] for k in fam]),
                       dict(kind='bar', label='Measured gain of heavy model (pts)', color=[0.902, 0.624, 0.000], width=0.36,
                            x=(xs + 0.19).tolist(), y=[fam[k]['gain'] for k in fam])]),
          open(os.path.join(MD, 'fig5_family_share.json'), 'w'), indent=1)
ben2 = z2['ah'] > z2['al']
DESC = [('entropy', 'Histogram entropy $H$', 'scene content'), ('variance', 'Variance $\\sigma^2$', 'contrast'),
        ('grad_density', 'Gradient density $G$', 'texture'), ('noise_sigma', 'Noise level $\\hat\\sigma_n$', 'sensor noise'),
        ('lap_energy', 'Laplacian energy $E_\\Delta$', 'blur'), ('grad_tail', 'Gradient tail $K$', 'noise vs.\\ edges'),
        ('scale_ratio', 'Scale ratio $R_s$', 'noise vs.\\ blur'), ('brightness', 'Brightness $\\mu_L$', 'exposure'),
        ('chroma', 'Chroma spread', 'fog, desaturation'), ('block_ratio', 'Block ratio', 'pixelation, JPEG')]
rows = []; S['desc_auc'] = {}
for j, (k, nm, what) in enumerate(DESC):
    a = roc_auc_score(ben2[te2], F[te2, j]); a = max(a, 1 - a); S['desc_auc'][k] = float(a)
    rows.append('%s & %s & %d & %.3f \\\\' % (nm, what, zf.OPS[k], a))
    if j == 2: rows.append('\\midrule')
open(os.path.join(TEX, 'table_desc_rows.tex'), 'w').write('\n'.join(rows) + '\n')
# ---------------- Fig. 4: budget controller trace
T = ct['T']; W_ = 500; ix = np.arange(W_ // 2, T - W_ // 2 + 1, 100)
tr = ct['trace']
Bw = np.convolve(np.array(ct['B']), np.ones(W_) / W_, 'same')     # budget averaged over the same window
xs_ = (ix / 1000).tolist()
ln = lambda k: np.array(tr[k])[ix].tolist()
json.dump(dict(name='fig4_budget_controller', layout='col', xlabel='Request index (x1000)',
               ylabel='Compute per input (normalized)', xlim=[0, 20], ylim=[0, 1.1],
               yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickformat='%.1f', xticks=[0, 5, 10, 15, 20],
               legend='north', legend_cols=2, lock_ylim=True,
               series=[dict(label='Static threshold', color=PAL['v1']['color'], marker='none', line=':', x=xs_, y=ln('static')),
                       dict(label='Offline table', color=PAL['cascade']['color'], marker='none', line='--', x=xs_, y=ln('lookup')),
                       dict(label='ZOCE dual (proposed)', color=PAL['triage']['color'], marker='none', line='-', proposed=True,
                            x=xs_, y=ln('dual')),
                       dict(kind='ref', label='Budget {\\itB_t}', line='-.', color=[0, 0, 0], x=xs_, y=Bw[ix].tolist()),
                       dict(kind='vline', x=5), dict(kind='vline', x=11), dict(kind='vline', x=16)],
               annotations=[dict(x=2.5, y=0.05, text='clean'), dict(x=8, y=0.05, text='degraded'),
                            dict(x=13.5, y=0.05, text='mixed'), dict(x=18, y=0.05, text='clean')]),
          open(os.path.join(MD, 'fig4_budget_controller.json'), 'w'), indent=1)

# ---------------- tables / macros
M = []
def mac(name, val, fmt='%.1f'):
    M.append('\\newcommand{\\%s}{%s}' % (name, (fmt % val) if not isinstance(val, str) else val))

for pk in ('P1', 'P2', 'P3', 'P4', 'N1'):
    R = r[pk]; d = dict(light=NICE[R['light']], heavy=NICE[R['heavy']], ratio=R['cl'] / R['ch'],
                        acc_l=float(np.mean([s['_light_acc'] for s in R['seeds']])),
                        acc_h=float(np.mean([s['_heavy_acc'] for s in R['seeds']])), m={})
    bmid = 0.5 if pk != 'P3' else 0.8
    for k in ('blind', 'v1', 'cnn', 'proposed', 'cond_oracle', 'cascade', 'triage', 'oracle', 'abl_v1feat_gain',
              'abl_full_easy', 'abl_full_gain_lin'):
        if k not in R['seeds'][0]: continue
        d['m'][k] = dict(auc=agg(pk, k, 'auc_gain'), acc_mid=acc_b(pk, k, bmid), acc30=acc_b(pk, k, 0.3),
                         cost1=agg(pk, k, 'cost_1pt'), ops=R['seeds'][0][k]['gate_ops'])
    d['bmid'] = bmid
    d['clean_auc'] = {k: float(np.mean([s['_clean_auc'][k] for s in R['seeds']])) for k in ('proposed', 'v1', 'cascade')}
    S['pairs'][pk] = d
S['controller'] = ct['summary']; S['controller_bound'] = tr['bound']; S['controller_cum'] = tr['dual_cum']
S['controller_phase'] = dict(B=tr['ph_B'], cost=tr['ph_cost'], acc=tr['ph_acc'])
S['models'] = {NICE[k]: v for k, v in ms.items()}
if lat: S['latency'] = lat
json.dump(S, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)

# main results table rows (LaTeX)
rows = []
names = [('blind', 'Data blind switching'), ('v1', 'Version 1 gate (3 statistics)'), ('cnn', 'Learned CNN gate'),
         ('proposed', '\\textbf{ZOCE R (proposed)}'), ('cascade', 'Confidence cascade'),
         ('triage', '\\textbf{ZOCE T (proposed)}')]
def cell(v, best, f='%.1f', scale=100.0):
    if np.isnan(v): return 'n/a'
    t = f % (v * scale); return '\\textbf{%s}' % t if best else t
for k, nm in names:
    row = [nm]
    for pk in ('P1', 'P2', 'P4', 'P3', 'N1'):
        P = S['pairs'][pk]['m']
        if k not in P: row += ['n/a', 'n/a', 'n/a']; continue
        pre = [q for q in ('blind', 'v1', 'cnn', 'proposed', 'cascade', 'triage') if q in P]
        ba = np.nanmax([round(100 * P[q]['acc_mid'][0], 1) for q in pre]); bc = np.nanmin([round(P[q]['cost1'][0], 2) for q in pre])  # v6: bold by displayed value
        row += [cell(P[k]['auc'][0], False, '%.2f', 1.0), cell(P[k]['acc_mid'][0], round(100 * P[k]['acc_mid'][0], 1) == ba),
                cell(P[k]['cost1'][0], round(P[k]['cost1'][0], 2) == bc, '%.2f', 1.0)]
    rows.append(' & '.join(row) + ' \\\\')
open(os.path.join(TEX, 'table_main_rows.tex'), 'w').write('\n'.join(rows) + '\n')
abl = [('v1', '3 stat. (V1), easy label, linear'), ('abl_v1feat_gain', '3 stat. (V1), gain label, linear'),
       ('abl_full_gain_lin', '10 descr., gain label, linear'), ('abl_full_easy', '10 descr., easy label, quadratic'),
       ('proposed', '\\textbf{10 descr., gain label, quadratic}'), ('cond_oracle', 'Condition oracle (true corruption)')]
rows = []
for k, nm in abl:
    rows.append(' & '.join([nm] + ['%.3f' % S['pairs'][pk]['m'][k]['auc'][0] for pk in ('P2', 'P4', 'N1')]
                           + ['%.1f' % (100 * S['pairs']['P2']['m'][k]['acc_mid'][0])]) + ' \\\\')
open(os.path.join(TEX, 'table_ablation_rows.tex'), 'w').write('\n'.join(rows) + '\n')
rows = []
for k in ('shufflenetv2_x0_5', 'mobilenetv2_x0_5', 'resnet20', 'mobilenetv2_x1_4', 'resnet56',
          'Hendrycks2020AugMix_WRN', 'Modas2021PRIMEResNet18'):
    v = ms[k]; l = (lat or {}).get(k, v['lat_ms'])
    rows.append('%s & %s & %.1f & %.2f & %.1f & %.1f & %.2f \\\\' % (
        NICE[k], 'robust' if k[0] in 'HM' else 'standard', v['flops'] / 1e6, v['params'] / 1e6,
        100 * v['acc_clean'], 100 * v['acc_stream'], l))
open(os.path.join(TEX, 'table_models_rows.tex'), 'w').write('\n'.join(rows) + '\n')

P = S['pairs']
mac('opsFull', '{:,}'.format(r['ops']['proposed']).replace(',', '\\,')); mac('opsVone', '%d' % r['ops']['v1'])  # v6: thin-space thousands
for pk in P:
    for k, v in P[pk]['m'].items():
        kk = {'blind': 'Blind', 'v1': 'Vone', 'cnn': 'Cnn', 'proposed': 'R', 'cascade': 'Casc', 'triage': 'T',
              'cond_oracle': 'Cond', 'oracle': 'Orc'}.get(k)
        if not kk: continue
        tag = {'P1': 'one', 'P2': 'two', 'P3': 'three', 'P4': 'four', 'N1': 'ctrl'}[pk]
        mac('auc%s%s' % (tag, kk), v['auc'][0], '%.3f'); mac('acc%s%s' % (tag, kk), 100 * v['acc_mid'][0])
        mac('accLow%s%s' % (tag, kk), 100 * v['acc30'][0]); mac('cost%s%s' % (tag, kk), v['cost1'][0], '%.2f')
for nm, k in (('Dual', 'dual'), ('Look', 'lookup'), ('Stat', 'static')):
    mac('ctlAcc' + nm, 100 * ct['summary'][k]['acc'][0]); mac('ctlViol' + nm, 100 * ct['summary'][k]['viol'][0], '%+.2f')
    mac('ctlPeak' + nm, ct['summary'][k]['maxrun'][0], '%.3f')
mac('ctlCum', tr['dual_cum'], '%.1f'); mac('ctlBound', tr['bound'], '%.0f')
if lat: mac('latFeat', lat['zoce_c_us'], '%.1f'); mac('latFeatMs', lat['zoce_c_us'] / 1e3, '%.3f')
# v6: gains are computed from the ROUNDED table entries so that text and Table III agree exactly
g_ = lambda pk, k: round(100 * (P[pk]['m'][k]['acc_mid'][0]), 1)
mac('gainRBlindOne', g_('P1', 'proposed') - g_('P1', 'blind')); mac('gainRBlindThree', g_('P3', 'proposed') - g_('P3', 'blind'))
mac('gainRBlindTwo', g_('P2', 'proposed') - g_('P2', 'blind')); mac('gainRBlindFour', g_('P4', 'proposed') - g_('P4', 'blind'))
mac('accGainBest', max(g_(pk, 'proposed') - g_(pk, 'blind') for pk in ('P1', 'P2', 'P3', 'P4')))
mac('gainTCascFour', g_('P4', 'triage') - g_('P4', 'cascade'))
mac('gapCondMax', max(g_(pk, 'cond_oracle') - g_(pk, 'proposed') for pk in ('P1', 'P2', 'P3', 'P4')), '%.2f')
mac('shareClean', S['bysev']['share_prop'][0], '%.0f'); mac('shareSevFive', S['bysev']['share_prop'][5], '%.0f')
mac('gainClean', S['bysev']['gain'][0]); mac('gainSevFive', S['bysev']['gain'][5])
mac('aucCtrlEasy', P['N1']['m']['abl_full_easy']['auc'][0], '%.3f')
mac('ctlUnderLook', -100 * ct['summary']['lookup']['viol'][0], '%.0f')
mac('ovhLight', 100 * r['ops']['proposed'] / ms['shufflenetv2_x0_5']['flops'], '%.2f')  # 'flops' from thop = MACs
import zoce_features as _zf; mac('opsLumGate', '%d' % (_zf.OPS['luminance'] + _zf.GATE_HEAD_OPS))
mac('ovhHeavy', 100 * r['ops']['proposed'] / ms['Hendrycks2020AugMix_WRN']['flops'], '%.3f')
mac('shareNoise', fam['Noise']['share'], '%.0f'); mac('shareBlur', fam['Blur']['share'], '%.0f')
mac('shareCleanFam', fam['Clean']['share'], '%.0f'); mac('lightNoise', fam['Noise']['light']); mac('gainNoise', fam['Noise']['gain'])
mac('aucDescBest', max(S['desc_auc'].values()), '%.3f')
A_ = lambda k: P['P2']['m'][k]['auc'][0]
mac('aucAblVoneGain', A_('abl_v1feat_gain'), '%.3f'); mac('aucAblLin', A_('abl_full_gain_lin'), '%.3f')
mac('aucAblQuadStep', A_('proposed') - A_('abl_full_gain_lin'), '%.3f')
bb = np.arange(0.2, 1.001, 0.05)
cur = lambda pk, k: np.mean([env(s_[k]['cost'], s_[k]['acc'], bb) for s_ in r[pk]['seeds']], 0) * 100
mac('maxCascLead', max(np.nanmax(cur(pk, 'cascade') - cur(pk, 'proposed')) for pk in ('P2', 'P4')))
mac('maxRLeadThree', np.nanmax(cur('P3', 'proposed') - cur('P3', 'cascade')))
mac('gapOneCasc', g_('P1', 'cascade') - g_('P1', 'proposed'))
mac('tSlack', max(max(np.nanmax(cur(pk, 'proposed') - cur(pk, 'triage')), np.nanmax(cur(pk, 'cascade') - cur(pk, 'triage')))
                  for pk in ('P1', 'P2', 'P3', 'P4')), '%.2f')
mac('maxTLead', max(np.nanmax(cur(pk, 'triage') - cur(pk, 'cascade')) for pk in ('P2', 'P4')))
for pk in ('P2', 'P4'):
    d_ = cur(pk, 'proposed') - cur(pk, 'cascade'); print(pk, 'ZOCE-R ahead of cascade for B in', bb[d_ > 0].round(2).tolist())
mac('aucCleanR', P['P2']['clean_auc']['proposed'], '%.2f')
# dominance check quoted in the text: ZOCE-T >= max(ZOCE-R, cascade) on the common budget grid
for pk in ('P1', 'P2', 'P3', 'P4', 'N1'):
    bb = np.arange(0.2, 1.001, 0.05); R_ = r[pk]['seeds']
    f = lambda k: np.mean([env(s[k]['cost'], s[k]['acc'], bb) for s in R_], 0)
    t_, p_, c_ = f('triage'), f('proposed'), f('cascade')
    print(pk, 'min(T - R) %.4f  min(T - cascade) %.4f  max(R - cascade) %.4f' % (
        np.nanmin(t_ - p_), np.nanmin(t_ - c_), np.nanmax(p_ - c_)), 'R>cascade for B <=',
        [float(b) for b in bb[(p_ > c_)]][-1:] )
# Proposition 3 check: bypass share chosen by ZOCE T (validation selected), averaged over the budget grid
for pk, tg in (('P1', 'One'), ('P4', 'Four'), ('P2', 'Two'), ('P3', 'Three'), ('N1', 'Ctrl')):
    mac('byp' + tg, 100 * np.mean([np.mean(s_['triage']['bypass']) for s_ in r[pk]['seeds']]))
open(os.path.join(TEX, 'results_macros.tex'), 'w').write('\n'.join(M) + '\n')
print(json.dumps({pk: {k: [round(v['auc'][0], 3), round(100 * v['acc_mid'][0], 1), round(100 * v['acc30'][0], 1), round(v['cost1'][0], 2)]
                       for k, v in P[pk]['m'].items()} for pk in P}, indent=0))
print('family', fam); print('desc auc', S['desc_auc'])
print('bysev', {k: np.round(v, 1).tolist() for k, v in S['bysev'].items()})
