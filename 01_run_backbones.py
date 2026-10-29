"""Step 1: run the frozen, pre-trained CIFAR-100 backbones on the evaluation stream.

Stream = CIFAR-100 test set (clean) + CIFAR-100-C (Hendrycks & Dietterich), i.e. every
one of the 10,000 test images appears once clean and K_DRAWS times under a randomly drawn
(corruption type, severity) pair out of the 19 x 5 conditions of CIFAR-100-C.
Backbones are the public checkpoints of github.com/chenyaofo/pytorch-cifar-models
(used strictly as black boxes: no fine-tuning, no architectural change).

Outputs  results/stream.npz   : images (uint8), labels, image index, corruption id, severity
         results/preds_<m>.npz: correct (0/1), max-softmax confidence, predicted class
         results/model_stats.json: params, FLOPs (thop), measured CPU latency
"""
import os, sys, json, time, pickle, tarfile
import numpy as np, torch
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('ZOCE_DATA', os.path.join(ROOT, '..', 'data'))
MODELS = os.environ.get('ZOCE_MODELS', os.path.join(ROOT, '..', 'models'))
PCM = os.environ.get('ZOCE_PCM', os.path.join(ROOT, '..', 'pcm'))
OUT = os.path.join(ROOT, 'results'); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, PCM)
import pytorch_cifar_models as pcm
from thop import profile

K_DRAWS, SEED = 6, 2026
CORR = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'spatter', 'saturate']
CKPT = {'shufflenetv2_x0_5': 'cifar100_shufflenetv2_x0_5-1977720f.pt',
        'mobilenetv2_x0_5': 'cifar100_mobilenetv2_x0_5-9f915757.pt',
        'resnet20': 'cifar100_resnet20-23dac2f1.pt',
        'resnet56': 'cifar100_resnet56-f2eff4c8.pt',
        'mobilenetv2_x1_4': 'cifar100_mobilenetv2_x1_4-8a269f5e.pt'}
# corruption-robust heavy experts: public RobustBench (CIFAR-100, corruptions) checkpoints.
# These models normalise internally and take inputs in [0,1].
ROBUST = ['Hendrycks2020AugMix_WRN', 'Modas2021PRIMEResNet18']
RB_DIR = os.environ.get('ZOCE_RB', os.path.join(ROOT, '..', 'rb'))
MEAN = np.array([0.5070, 0.4865, 0.4409], dtype=np.float32)
STD = np.array([0.2673, 0.2564, 0.2761], dtype=np.float32)


def build_stream():
    f = os.path.join(OUT, 'stream.npz')
    if os.path.exists(f):
        return np.load(f)
    with tarfile.open(os.path.join(DATA, 'cifar-100-python.tar.gz')) as t:
        d = pickle.load(t.extractfile('cifar-100-python/test'), encoding='latin1')
    x0 = d['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
    y0 = np.array(d['fine_labels'], dtype=np.int64)
    yc = np.load(os.path.join(DATA, 'CIFAR-100-C', 'labels.npy'))
    assert np.array_equal(yc[:10000], y0), 'CIFAR-100-C ordering differs from the test set'
    rng = np.random.default_rng(SEED)
    cid = rng.integers(0, len(CORR), size=(10000, K_DRAWS))
    sev = rng.integers(1, 6, size=(10000, K_DRAWS))
    X = [x0]; Y = [y0]; I = [np.arange(10000)]; C = [np.full(10000, -1)]; S = [np.zeros(10000, int)]
    for c, name in enumerate(CORR):
        arr = np.load(os.path.join(DATA, 'CIFAR-100-C', name + '.npy'), mmap_mode='r')
        ii, kk = np.nonzero(cid == c)
        ss = sev[ii, kk]
        X.append(np.stack([arr[(s - 1) * 10000 + i] for i, s in zip(ii, ss)]))
        Y.append(y0[ii]); I.append(ii); C.append(np.full(len(ii), c)); S.append(ss)
    np.savez_compressed(f, x=np.concatenate(X), y=np.concatenate(Y), idx=np.concatenate(I),
                        cid=np.concatenate(C), sev=np.concatenate(S), corr=np.array(CORR))
    return np.load(f)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    st = build_stream()
    x, y = st['x'], st['y']
    print('stream', x.shape, 'clean share %.3f' % np.mean(st['cid'] < 0), flush=True)
    stats = {}
    f = os.path.join(OUT, 'model_stats.json')
    if os.path.exists(f): stats = json.load(open(f))
    for name in list(CKPT) + ROBUST:
        if name in stats and os.path.exists(os.path.join(OUT, 'preds_%s.npz' % name)): continue
        if name in CKPT:
            net = getattr(pcm, 'cifar100_' + name)(pretrained=False)
            net.load_state_dict(torch.load(os.path.join(MODELS, CKPT[name]), map_location='cpu'))
            mean, std = MEAN, STD
        else:
            from robustbench.utils import load_model
            net = load_model(name, dataset='cifar100', threat_model='corruptions', model_dir=RB_DIR)
            mean, std = np.zeros(3, np.float32), np.ones(3, np.float32)
        net.eval()
        if hasattr(net, 'convert_to_inference_model'):
            net.convert_to_inference_model()          # RepVGG re-parameterised deploy form
        import copy
        fl, pa = profile(copy.deepcopy(net), inputs=(torch.zeros(1, 3, 32, 32),), verbose=False)
        one = torch.zeros(1, 3, 32, 32)
        torch.set_num_threads(1)
        for _ in range(20): net(one)
        t = []
        for _ in range(200):
            t0 = time.perf_counter(); net(one); t.append(time.perf_counter() - t0)
        torch.set_num_threads(2)
        stats[name] = dict(flops=float(fl), params=float(pa), lat_ms=float(np.median(t) * 1e3))
        fo = os.path.join(OUT, 'preds_%s.npz' % name)
        if not os.path.exists(fo):
            cor, conf, pred = [], [], []
            for b in range(0, len(x), 500):
                xb = (x[b:b + 500].astype(np.float32) / 255. - mean) / std
                p = torch.softmax(net(torch.from_numpy(xb).permute(0, 3, 1, 2).contiguous()), 1)
                c, k = p.max(1)
                conf.append(c.numpy()); pred.append(k.numpy())
            pred = np.concatenate(pred); conf = np.concatenate(conf)
            np.savez_compressed(fo, correct=(pred == y), conf=conf, pred=pred)
        r = np.load(fo)
        stats[name]['acc_clean'] = float(r['correct'][st['cid'] < 0].mean())
        stats[name]['acc_stream'] = float(r['correct'].mean())
        print(name, stats[name], flush=True)
        json.dump(stats, open(os.path.join(OUT, 'model_stats.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
