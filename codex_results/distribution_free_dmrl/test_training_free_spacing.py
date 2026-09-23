import unittest
import numpy as np
from training_free_spacing import stopping_indices, evaluate, CONFIGS, matched_cost


class SpacingTests(unittest.TestCase):
    def test_exact_first_crossing(self):
        numerator = np.array([[[1.,1.,1.,1.,.1,.1,.1,.1]]])
        cost = np.arange(1,9, dtype=float)[None,None,:]
        self.assertEqual(stopping_indices(numerator,cost,[.1],4,'sequential').item(),4)
        self.assertEqual(stopping_indices(numerator,cost,[.1],4,'doubling').item(),7)

    def test_cap_and_future_independence(self):
        n = np.ones((1,1,20)); c = np.arange(1,21,dtype=float)[None,None,:]
        a = stopping_indices(n,c,[1.],4,'sequential')
        c[...,5:] *= 1000
        np.testing.assert_array_equal(a,stopping_indices(n,c,[1.],4,'sequential'))
        self.assertEqual(stopping_indices(n,np.ones_like(n),[0.],4,'sequential').item(),19)

    def test_evaluation_quality_cannot_affect_stop(self):
        rng = np.random.default_rng(80); cap = 32
        top = np.full((1,1,cap,33),-np.inf)
        rewards = rng.normal(size=cap)
        for j in range(cap):
            s = np.sort(rewards[:j+1])[::-1];top[0,0,j,:len(s)] = s
        ref = top[...,0].copy();cost = np.arange(1,cap+1)[None,None,:]*100.
        q = np.zeros_like(ref)
        a = evaluate(top,ref,cost,q,rates=np.array([1e-5,1e-4]))
        b = evaluate(top,ref,cost,q+1,rates=np.array([1e-5,1e-4]))
        np.testing.assert_array_equal(a[...,1:],b[...,1:])
        np.testing.assert_allclose(b[...,0]-a[...,0],1.)
        self.assertEqual(a.shape,(1,len(CONFIGS),2,3))

    def test_mixture(self):
        values = np.array([[.2,10],[.4,20],[.6,50]])
        self.assertAlmostEqual(matched_cost(values,.3),15.)
        self.assertIsNone(matched_cost(values,.7))


if __name__ == '__main__':
    unittest.main()
