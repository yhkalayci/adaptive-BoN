import unittest
import numpy as np
from online_early_start import small_start_statistics,average_from
from online_smoothed_spacing import statistics,smooth_numerator


class EarlyStartTests(unittest.TestCase):
    def test_reproduces_old_after_four(self):
        r=np.random.default_rng(15).normal(size=40)
        a=small_start_statistics(r);b=statistics(r)['odds']
        np.testing.assert_allclose(a[3:],b[3:])
        np.testing.assert_allclose(average_from(a,4),smooth_numerator(b,'mean'))

    def test_prefix_only(self):
        r=np.array([1.,2.,3.,4.,5.]);a=small_start_statistics(r)
        r[3:]+=100
        np.testing.assert_array_equal(a[:3],small_start_statistics(r)[:3])

    def test_constant_rewards(self):
        np.testing.assert_array_equal(small_start_statistics(np.ones(10)),np.zeros(10))


if __name__=='__main__':unittest.main()
