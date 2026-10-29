"""Zero-overhead complexity/quality descriptors computed on the RAW uint8 input (no DNN).

All ten descriptors use only adds, subtractions, absolute values and one 32-bin histogram on
the 32x32 luminance plane, i.e. O(hw) work.  OPS is an explicit arithmetic-operation count
(1 op = one add/sub/abs/compare/mul; log, exp, division = 4 ops; counted pessimistically) used for the overhead numbers.
"""
import numpy as np

NAMES = ['entropy', 'variance', 'grad_density',            # the three descriptors of version 1
         'noise_sigma', 'lap_energy', 'grad_tail', 'scale_ratio',
         'brightness', 'chroma', 'block_ratio']


def features(x):
    """x: (N,32,32,3) uint8  ->  (N,10) float32"""
    x = x.astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    L = (r + 2 * g + b) / 4.0                                   # shift-add luminance
    n = len(L)
    # 1) 32-bin histogram entropy
    q = (L // 8).astype(np.int64).reshape(n, -1)
    H = np.zeros(n, np.float32)
    for k in range(32):
        p = (q == k).mean(1)
        H -= np.where(p > 0, p * np.log2(p + 1e-7), 0).astype(np.float32)
    # 2) variance, 8) brightness
    mu = L.mean((1, 2)); var = L.var((1, 2))
    # 3) gradient density (forward differences)
    dx = L[:, :, 1:] - L[:, :, :-1]; dy = L[:, 1:, :] - L[:, :-1, :]
    gd = np.abs(dx).mean((1, 2)) + np.abs(dy).mean((1, 2))
    # 4) noise level: Immerkaer mask [1 -2 1; -2 4 -2; 1 -2 1] (cancels image structure)
    c = L[:, 1:-1, 1:-1]
    M = (L[:, :-2, :-2] + L[:, :-2, 2:] + L[:, 2:, :-2] + L[:, 2:, 2:]
         - 2 * (L[:, :-2, 1:-1] + L[:, 2:, 1:-1] + L[:, 1:-1, :-2] + L[:, 1:-1, 2:]) + 4 * c)
    sig = np.sqrt(np.pi / 2) / 6.0 * np.abs(M).mean((1, 2))
    # 5) Laplacian energy (sharpness / blur indicator)
    lap = (L[:, :-2, 1:-1] + L[:, 2:, 1:-1] + L[:, 1:-1, :-2] + L[:, 1:-1, 2:] - 4 * c)
    le = np.abs(lap).mean((1, 2))
    # 6) gradient tail weight: E[d^2] / E[|d|]^2 (sparse edges -> large, noise -> ~pi/2)
    gt = (dx ** 2).mean((1, 2)) / (np.abs(dx).mean((1, 2)) ** 2 + 1e-3)
    # 7) two-scale ratio: gradient density after 2x2 box pooling vs. full resolution
    P = (L[:, 0::2, 0::2] + L[:, 1::2, 0::2] + L[:, 0::2, 1::2] + L[:, 1::2, 1::2]) / 4.0
    g2 = np.abs(P[:, :, 1:] - P[:, :, :-1]).mean((1, 2)) + np.abs(P[:, 1:, :] - P[:, :-1, :]).mean((1, 2))
    sr = g2 / (gd + 1e-3)
    # 9) chroma spread
    ch = (np.abs(r - g) + np.abs(b - g)).mean((1, 2))
    # 10) block ratio: horizontal differences across even vs. odd columns (pixelation/JPEG grid)
    br = np.abs(dx[:, :, 1::2]).mean((1, 2)) / (np.abs(dx[:, :, 0::2]).mean((1, 2)) + 1e-3)
    return np.stack([H, var, gd, sig, le, gt, sr, mu, ch, br], 1).astype(np.float32)


# explicit operation count per descriptor for a 32x32 input (hw = 1024)
hw = 1024
OPS = {'luminance': 3 * hw, 'entropy': 2 * hw + 7 * 32,   # v10: per bin p = count * (1/n) (1), log2 p (4), p * log2 p (1), accumulate (1) = 7 ops
       'variance': 3 * hw, 'grad_density': 6 * hw,
       'noise_sigma': 10 * hw, 'lap_energy': 6 * hw, 'grad_tail': 2 * hw, 'scale_ratio': 3 * hw + 6 * hw // 4,
       'brightness': 0, 'chroma': 5 * hw, 'block_ratio': hw}
V1_OPS = OPS['luminance'] + OPS['entropy'] + OPS['variance'] + OPS['grad_density']
FULL_OPS = sum(OPS.values())


# v10: operation count of the gate itself (same counting rule as OPS)
#   standardize 10 descriptors (sub + mul)        20
#   log1p on 5 heavy tailed descriptors (4 each)  20
#   55 quadratic products                          55
#   two logistic heads, 65 mul + 65 add each      260
#   two sigmoids (exp 4 + add 1 + div 4)           18
#   difference of the two heads                     1
GATE_HEAD_OPS = 20 + 20 + 55 + 260 + 18 + 1          # = 374
#   ZOCE T verification: logit(kappa) (div 4 + log 4 + sub 1) + square 1 + 10 products
#   + 12 extra weights per head (12 mul + 12 add) x 2                       = 68
VERIFY_EXTRA_OPS = 9 + 1 + 10 + 48
#   version 1 gate: standardize 3 (6) + log1p on 2 (8) + 3 mul + 3 add (6) + sigmoid (9) = 29
V1_GATE_OPS = 29
