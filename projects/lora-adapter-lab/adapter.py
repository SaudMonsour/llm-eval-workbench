"""Low-rank adaptation of a frozen linear map, with explicit NumPy gradients."""
import numpy as np

class Adapter:

    def __init__(self, weight, rank=4, alpha=4.0, seed=42):
        w = np.asarray(weight, dtype=np.float64)
        if w.ndim != 2 or not np.isfinite(w).all() or (not 1 <= rank <= min(w.shape)) or (not np.isfinite(alpha)) or (alpha <= 0):
            raise ValueError('Invalid matrix, rank or alpha')
        self.weight = w.copy()
        self.weight.flags.writeable = False
        self.factor = alpha / rank
        self.a = np.random.default_rng(seed).normal(0, 0.02, size=(w.shape[0], rank))
        self.b = np.zeros((rank, w.shape[1]))

    def merged(self):
        return self.weight + self.factor * self.a @ self.b

    def predict(self, x):
        return np.asarray(x) @ self.merged()

    def loss_grad(self, x, y):
        x, y = (np.asarray(x), np.asarray(y))
        if x.ndim != 2 or y.shape != (len(x), self.weight.shape[1]) or x.shape[1] != self.weight.shape[0] or (len(x) == 0) or (not np.isfinite(x).all()) or (not np.isfinite(y).all()):
            raise ValueError('Invalid training arrays')
        e = self.predict(x) - y
        g = 2 * x.T @ e / e.size
        return (float(np.mean(e ** 2)), self.factor * g @ self.b.T, self.factor * self.a.T @ g)

    def save(self, path):
        np.savez(path, a=self.a, b=self.b, factor=self.factor)

    def load(self, path):
        with np.load(path, allow_pickle=False) as d:
            if d['a'].shape != self.a.shape or d['b'].shape != self.b.shape or (not all((np.isfinite(d[k]).all() for k in ('a', 'b', 'factor')))):
                raise ValueError('Incompatible adapter')
            self.a = d['a'].copy()
            self.b = d['b'].copy()
            self.factor = float(d['factor'])
