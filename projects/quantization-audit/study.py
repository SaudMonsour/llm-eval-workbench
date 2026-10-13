import json, csv
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from quantize import quantize, pack4, unpack4
x, y = load_digits(return_X_y=True)
xt, xh, yt, yh = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
s = StandardScaler().fit(xt)
xt = s.transform(xt)
xh = s.transform(xh)
m = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(xt, yt)
w = m.coef_.T.astype(np.float32)
b = m.intercept_.astype(np.float32)
reference = xh @ w + b
errors = []
rows = []
Path('figures').mkdir(exist_ok=True)
for name, bits, channel in [('FP32', None, False), ('INT8 tensor', 8, False), ('INT8 channel', 8, True), ('INT4 tensor', 4, False), ('INT4 channel', 4, True)]:
    if bits:
        q, scale = quantize(w, bits, channel)
        packed = pack4(q) if bits == 4 else q
        dq = unpack4(packed, w.shape) if bits == 4 else q
        ww = dq * scale
        size = packed.nbytes + scale.nbytes + b.nbytes
    else:
        ww = w
        size = w.nbytes + b.nbytes
    logits = xh @ ww + b
    prob = np.exp(logits - logits.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    errors.extend(({'holdout_index': i, 'method': name, 'true_class': int(yh[i]), 'predicted_class': int(prob[i].argmax()), 'reference_class': int(reference[i].argmax()), 'correct': bool(prob[i].argmax() == yh[i])} for i in range(len(yh))))
    rows.append({'method': name, 'accuracy': float(accuracy_score(yh, prob.argmax(1))), 'log_loss': float(log_loss(yh, prob)), 'parameter_bytes': int(size), 'logit_rmse': float(np.sqrt(np.mean((logits - reference) ** 2))), 'prediction_flips': int(np.sum(logits.argmax(1) != reference.argmax(1)))})
with open('error_analysis.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=list(errors[0]))
    writer.writeheader()
    writer.writerows(errors)
np.savez('weights.npz', weight=w, bias=b, mean=s.mean_, scale=s.scale_)
Path('metrics.json').write_text(json.dumps({'seed': 42, 'train': len(xt), 'holdout': len(xh), 'data_source': 'scikit-learn bundled UCI Optical Recognition of Handwritten Digits', 'model': 'Multinomial logistic regression; not an LLM', 'quantization': 'Weight only, dequantized float matrix multiplication; no GPU speed claim', 'results': rows}, indent=2) + '\n')
for key, title in [('accuracy', 'Holdout accuracy'), ('parameter_bytes', 'Weight, scale and bias storage (bytes)')]:
    plt.figure(figsize=(8, 4))
    plt.bar([r['method'] for r in rows], [r[key] for r in rows], color='#315b7d')
    plt.ylabel(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig('figures/' + key + '.png', dpi=150)
    plt.close()
print(rows)
