import json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adapter import Adapter
r = np.random.default_rng(42)
w = r.normal(0, 0.1, (32, 24))
target = w + r.normal(0, 0.2, (32, 4)) @ r.normal(0, 0.2, (4, 24))
x = r.normal(size=(900, 32))
y = x @ target + r.normal(0, 0.02, (900, 24))
xt, xv, xh = (x[:600], x[600:750], x[750:])
yt, yv, yh = (y[:600], y[600:750], y[750:])
rows = [{'method': 'Frozen base', 'parameters': 0, 'holdout_mse': float(np.mean((xh @ w - yh) ** 2))}]
traces = {}
for rank in [1, 4, 8, None]:
    a = Adapter(w, rank, rank) if rank else None
    p = [a.a, a.b] if a else [w.copy()]
    m = [np.zeros_like(z) for z in p]
    v = [z.copy() for z in m]
    best = float('inf')
    saved = None
    curve = []
    for step in range(1, 801):
        if a:
            loss, ga, gb = a.loss_grad(xt, yt)
            gs = [ga, gb]
        else:
            e = xt @ p[0] - yt
            loss = float(np.mean(e ** 2))
            gs = [2 * xt.T @ e / e.size]
        for i, (z, g) in enumerate(zip(p, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            z -= 0.01 * (m[i] / (1 - 0.9 ** step)) / (np.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-08)
        validation = float(np.mean(((a.predict(xv) if a else xv @ p[0]) - yv) ** 2))
        curve.append(validation)
        if validation < best:
            best = validation
            saved = [z.copy() for z in p]
            best_step = step
    for z, best_z in zip(p, saved):
        z[:] = best_z
    name = f'LoRA rank {rank}' if rank else 'Full update'
    pred = a.predict(xh) if a else xh @ p[0]
    rows.append({'method': name, 'parameters': sum((z.size for z in p)), 'selected_step': best_step, 'validation_mse': best, 'holdout_mse': float(np.mean((pred - yh) ** 2))})
    traces[name] = curve
    if rank == 4:
        a.save('adapter.npz')
        np.save('base.npy', w)
        np.save('holdout_inputs.npy', xh)
Path('figures').mkdir(exist_ok=True)
Path('metrics.json').write_text(json.dumps({'seed': 42, 'fixture': 'Synthetic noisy linear transfer task, true update rank 4; not an LLM benchmark', 'train': 600, 'validation': 150, 'holdout': 150, 'optimizer': 'Adam, lr .01, 800 steps; minimum validation checkpoint', 'results': rows}, indent=2) + '\n')
for n, c in traces.items():
    plt.plot(c, label=n)
plt.yscale('log')
plt.xlabel('Step')
plt.ylabel('Validation MSE')
plt.legend()
plt.tight_layout()
plt.savefig('figures/learning-curves.png', dpi=150)
plt.close()
plt.bar([z['method'] for z in rows], [z['holdout_mse'] for z in rows], color='#315b7d')
plt.yscale('log')
plt.ylabel('Holdout MSE')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('figures/holdout.png', dpi=150)
print(rows)
with open('learning_curves.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['step'] + list(traces))
    writer.writerows(([i + 1] + [traces[n][i] for n in traces] for i in range(800)))
