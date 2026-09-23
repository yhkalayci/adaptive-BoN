import unittest
import numpy as np
from training_free_lipschitz import gains


class LipschitzTests(unittest.TestCase):
    def test_prefix_measurability(self):
        r=np.random.default_rng(87).normal(size=50)
        a=gains(r);r[20:]+=100
        np.testing.assert_array_equal(a[:,:20],gains(r)[:,:20])

    def test_translation_invariance_and_scale(self):
        r=np.random.default_rng(81).normal(size=50)
        a=gains(r)
        np.testing.assert_allclose(a,gains(r+50),atol=1e-14)
        np.testing.assert_allclose(a*3,gains(r*3),atol=1e-14)

    def test_ties_and_extreme_incumbent(self):
        np.testing.assert_array_equal(gains(np.ones(20))[:,3:],0.)
        r=np.arange(20,dtype=float)
        a=gains(r);r[-1]=1000.;b=gains(r)
        self.assertTrue(np.all(b[1::2,-1] < a[1::2,-1]))


if __name__=='__main__':
    unittest.main()
