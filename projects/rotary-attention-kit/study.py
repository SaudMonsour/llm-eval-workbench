import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from attention import rope, attention
r = np.random.default_rng(42)
q, k, v = [r.normal(size=(4, 64, 32)) for _ in range(3)]
p = np.arange(64)
qr, kr = (rope(q, p), rope(k, p))
y, a = attention(qr, kr, v, p, p)
shift = 137
ys, _ = attention(rope(q, p + shift), rope(k, p + shift), v, p + shift, p + shift)
yinc = []
for t in range(64):
    z, _ = attention(qr[:, t:t + 1], kr[:, :t + 1], v[:, :t + 1], p[t:t + 1], p[:t + 1])
    yinc.append(z)
err = float(np.max(np.abs(y - np.concatenate(yinc, axis=1))))
result = {'seed': 42, 'fixture': 'Synthetic Gaussian Q/K/V; no trained language model', 'heads': 4, 'tokens': 64, 'head_dim': 32, 'incremental_max_abs_error': err, 'joint_shift_max_abs_error': float(np.max(np.abs(y - ys))), 'rotation_norm_max_abs_error': float(np.max(np.abs(np.linalg.norm(q, axis=-1) - np.linalg.norm(qr, axis=-1)))), 'future_attention_max': float(a[:, np.triu_indices(64, 1)[0], np.triu_indices(64, 1)[1]].max())}
Path('figures').mkdir(exist_ok=True)
Path('metrics.json').write_text(json.dumps(result, indent=2) + '\n')
plt.imshow(a[0], origin='lower', aspect='auto', cmap='viridis')
plt.xlabel('Key position')
plt.ylabel('Query position')
plt.title('Causal RoPE attention, head 0')
plt.colorbar(label='Probability')
plt.tight_layout()
plt.savefig('figures/attention.png', dpi=150)
plt.close()
off = np.arange(64)
dots = [float(rope(q[:, :1], [0])[0, 0] @ rope(k[:, :1], [int(i)])[0, 0]) for i in off]
plt.plot(off, dots)
plt.xlabel('Relative position')
plt.ylabel('Rotated Q · K')
plt.title('Position changes a fixed query/key score')
plt.tight_layout()
plt.savefig('figures/relative-position.png', dpi=150)
print(result)
