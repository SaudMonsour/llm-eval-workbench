import tempfile
import unittest
import numpy as np
from adapter import Adapter

class Tests(unittest.TestCase):

    def setUp(self):
        r = np.random.default_rng(4)
        self.w = r.normal(size=(5, 3))
        self.x = r.normal(size=(7, 5))
        self.y = r.normal(size=(7, 3))
        self.a = Adapter(self.w, 2, 2)

    def test_zero_initialization(self):
        np.testing.assert_array_equal(self.a.predict(self.x), self.x @ self.w)

    def test_gradients(self):
        self.a.b[:] = 0.1
        _, ga, gb = self.a.loss_grad(self.x, self.y)
        for p, g in [(self.a.a, ga), (self.a.b, gb)]:
            for idx in np.ndindex(p.shape):
                old = p[idx]
                p[idx] = old + 1e-06
                plus = self.a.loss_grad(self.x, self.y)[0]
                p[idx] = old - 1e-06
                minus = self.a.loss_grad(self.x, self.y)[0]
                p[idx] = old
                self.assertAlmostEqual(g[idx], (plus - minus) / 2e-06, places=6)

    def test_frozen(self):
        with self.assertRaises(ValueError):
            self.a.weight[0, 0] = 9

    def test_merge(self):
        self.a.b[:] = 0.2
        np.testing.assert_allclose(self.a.predict(self.x), self.x @ self.a.weight + self.a.factor * (self.x @ self.a.a) @ self.a.b)

    def test_roundtrip(self):
        self.a.b[:] = 0.2
        with tempfile.TemporaryDirectory() as d:
            self.a.save(d + '/a.npz')
            b = Adapter(self.w, 2, 2)
            b.load(d + '/a.npz')
            np.testing.assert_array_equal(b.predict(self.x), self.a.predict(self.x))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            Adapter(self.w, 7)
if __name__ == '__main__':
    unittest.main()
