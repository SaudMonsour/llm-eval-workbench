"""Symmetric signed quantization and real packed int4 storage."""
import numpy as np

def quantize(w, bits=8, per_channel=True):
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2 or not np.isfinite(w).all() or bits not in (4, 8):
        raise ValueError('Finite matrix; bits must be 4 or 8')
    limit = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(w), axis=0, keepdims=True) if per_channel else np.array([[np.max(np.abs(w))]])
    scale = np.where(scale == 0, 1, scale / limit).astype(np.float32)
    return (np.clip(np.rint(w / scale), -limit, limit).astype(np.int8), scale)

def pack4(q):
    q = np.asarray(q)
    if q.dtype != np.int8 or np.any(q < -7) or np.any(q > 7):
        raise ValueError('Expected symmetric int4 in int8 array')
    a = (q.ravel().astype(np.int16) & 15).astype(np.uint8)
    if len(a) % 2:
        a = np.append(a, np.uint8(0))
    return a[0::2] | a[1::2] << 4

def unpack4(p, shape):
    p = np.asarray(p, dtype=np.uint8)
    n = int(np.prod(shape))
    if n < 1 or len(p) != (n + 1) // 2:
        raise ValueError('Packed length mismatch')
    a = np.empty(len(p) * 2, np.int8)
    a[0::2] = p & 15
    a[1::2] = p >> 4
    a = np.where(a >= 8, a - 16, a).astype(np.int8)
    return a[:n].reshape(shape)
