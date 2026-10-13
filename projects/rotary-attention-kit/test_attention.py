import unittest
import numpy as np
from attention import rope, attention

class Tests(unittest.TestCase):

    def setUp(self):
        r = np.random.default_rng(7)
        self.q, self.k, self.v = [r.normal(size=(2, 8, 6)) for _ in range(3)]
        self.p = np.arange(8)

    def test_norm(self):
        np.testing.assert_allclose(np.linalg.norm(rope(self.q, self.p), axis=-1), np.linalg.norm(self.q, axis=-1), atol=1e-12)

    def test_relative_shift(self):
        a = rope(self.q, self.p) @ rope(self.k, self.p).transpose(0, 2, 1)
        b = rope(self.q, self.p + 53) @ rope(self.k, self.p + 53).transpose(0, 2, 1)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_causality(self):
        y, _ = attention(self.q, self.k, self.v, self.p, self.p)
        v = self.v.copy()
        v[:, 5:] = 999
        y2, _ = attention(self.q, self.k, v, self.p, self.p)
        np.testing.assert_allclose(y[:, :5], y2[:, :5])

    def test_incremental(self):
        y, _ = attention(self.q, self.k, self.v, self.p, self.p)
        rows = [attention(self.q[:, i:i + 1], self.k[:, :i + 1], self.v[:, :i + 1], self.p[i:i + 1], self.p[:i + 1])[0] for i in range(8)]
        np.testing.assert_allclose(y, np.concatenate(rows, axis=1), atol=1e-12)

    def test_bad_shape(self):
        with self.assertRaises(ValueError):
            rope(np.ones((1, 2, 3)), [0, 1])

    def test_empty_visible(self):
        with self.assertRaises(ValueError):
            attention(self.q, self.k, self.v, self.p - 100, self.p)
if __name__ == '__main__':
    unittest.main()
