"""Step 5 (run on an idle machine): single-thread, batch-1 host latency of every backbone
(PyTorch, median of 300 runs) and of the C implementation of the ZOCE descriptors."""
import os, sys, json, time, subprocess, importlib.util
import numpy as np, torch
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, 'results')
spec = importlib.util.spec_from_file_location('rb', os.path.join(ROOT, '01_run_backbones.py'))
rb = importlib.util.module_from_spec(spec); spec.loader.exec_module(rb)
torch.set_num_threads(1); lat = {}
one = torch.zeros(1, 3, 32, 32)
with torch.no_grad():
    for name in list(rb.CKPT) + rb.ROBUST:
        if name in rb.CKPT:
            net = getattr(rb.pcm, 'cifar100_' + name)(pretrained=False)
            net.load_state_dict(torch.load(os.path.join(rb.MODELS, rb.CKPT[name]), map_location='cpu'))
        else:
            from robustbench.utils import load_model
            net = load_model(name, dataset='cifar100', threat_model='corruptions', model_dir=rb.RB_DIR)
        net.eval()
        for _ in range(30): net(one)
        t = []
        for _ in range(300):
            t0 = time.perf_counter(); net(one); t.append(time.perf_counter() - t0)
        lat[name] = float(np.median(t) * 1e3); print(name, '%.2f ms' % lat[name], flush=True)
exe = os.path.join(ROOT, 'zoce_features')
subprocess.run(['gcc', '-O2', '-o', exe, os.path.join(ROOT, 'zoce_features.c'), '-lm'], check=True)
x = np.load(os.path.join(OUT, 'stream.npz'))['x'][:10000]; x.tofile('/tmp/zoce_x.u8')
us = min(float(subprocess.run([exe, '/tmp/zoce_x.u8', '10000', '/tmp/zoce_f.f32'], capture_output=True, text=True).stdout)
         for _ in range(3))
import zoce_features as zf
err = np.abs(np.fromfile('/tmp/zoce_f.f32', np.float32).reshape(-1, 10) - zf.features(x)).max()
lat['zoce_c_us'] = us; lat['c_vs_numpy_max_abs_err'] = float(err)
print('ZOCE descriptors (C): %.2f us/image, parity error %.2e' % (us, err))
json.dump(lat, open(os.path.join(OUT, 'latency.json'), 'w'), indent=1)
