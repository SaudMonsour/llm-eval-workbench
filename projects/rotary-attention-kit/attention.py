"""NumPy reference attention with adjacent-pair rotary positions."""
import numpy as np

def rope(x, positions, base=10000.0):
    x = np.asarray(x)
    positions = np.asarray(positions)
    if x.ndim != 3 or x.shape[-1] % 2 or positions.shape != (x.shape[1],):
        raise ValueError('Expected [heads,tokens,even_dim] and token positions')
    if not np.isfinite(base) or base <= 1 or (not np.isfinite(x).all()) or (not np.isfinite(positions).all()):
        raise ValueError('Finite inputs and base > 1 required')
    theta = positions[:, None] * base ** (-np.arange(0, x.shape[-1], 2) / x.shape[-1])
    out = np.empty_like(x, dtype=np.result_type(x, np.float64))
    a, b = (x[..., 0::2], x[..., 1::2])
    out[..., 0::2] = a * np.cos(theta) - b * np.sin(theta)
    out[..., 1::2] = a * np.sin(theta) + b * np.cos(theta)
    return out

def attention(q, k, v, q_positions, k_positions, causal=True):
    q, k, v = map(np.asarray, (q, k, v))
    if q.ndim != 3 or min(q.shape) == 0 or min(k.shape) == 0 or (k.ndim != 3) or (v.shape != k.shape) or (q.shape[0] != k.shape[0]) or (q.shape[2] != k.shape[2]):
        raise ValueError('Incompatible attention shapes')
    qp, kp = (np.asarray(q_positions), np.asarray(k_positions))
    if qp.shape != (q.shape[1],) or kp.shape != (k.shape[1],):
        raise ValueError('Position shape mismatch')
    if not all((np.isfinite(x).all() for x in (q, k, v, qp, kp))):
        raise ValueError('Nonfinite input')
    score = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
    allowed = kp[None, :] <= qp[:, None] if causal else np.ones(score.shape[1:], bool)
    if not allowed.any(axis=-1).all():
        raise ValueError('A query has no visible keys')
    score = np.where(allowed, score, -np.inf)
    prob = np.exp(score - score.max(axis=-1, keepdims=True))
    prob /= prob.sum(axis=-1, keepdims=True)
    return (prob @ v, prob)
