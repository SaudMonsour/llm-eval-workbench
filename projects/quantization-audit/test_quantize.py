import unittest
import numpy as np
from quantize import quantize, pack4, unpack4

class Tests(unittest.TestCase):

    def test_packing(self):
        a = np.arange(-7, 8, dtype=np.int8).reshape(3, 5)
        np.testing.assert_array_equal(a, unpack4(pack4(a), a.shape))
        self.assertEqual(pack4(a).nbytes, 8)

    def test_zero(self):
        q, s = quantize(np.zeros((3, 2)), 4)
        np.testing.assert_array_equal(q * s, 0)
        self.assertTrue(np.isfinite(s).all())

    def test_error_bound(self):
        w = np.random.default_rng(1).normal(size=(20, 4))
        q, s = quantize(w, 8)
        self.assertTrue(np.all(np.abs(q * s - w) <= s / 2 + 1e-06))

    def test_channel_outlier(self):
        w = np.array([[1.0, 100.0], [0.5, 50.0]])
        qt, st = quantize(w, 4, False)
        qc, sc = quantize(w, 4, True)
        self.assertLess(np.linalg.norm((qc * sc - w)[:, 0]), np.linalg.norm((qt * st - w)[:, 0]))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            quantize([[float('nan')]], 8)

    def test_pack_invalid(self):
        with self.assertRaises(ValueError):
            pack4(np.array([8], np.int8))
if __name__ == '__main__':
    unittest.main()
