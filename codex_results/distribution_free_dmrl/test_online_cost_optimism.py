import unittest
import numpy as np
from online_cost_optimism import effective_cost
from training_free_spacing import stopping_indices


class CostOptimismTests(unittest.TestCase):
    def test_zero_bonus_and_constant_cost(self):
        l=np.arange(1,30,dtype=float)
        np.testing.assert_array_equal(effective_cost(l,0),np.cumsum(l))
        np.testing.assert_array_equal(effective_cost(np.ones(30),2),np.arange(1,31))

    def test_prefix_only_and_positive(self):
        l=np.arange(1,30,dtype=float);a=effective_cost(l,2);l[10:]*=100
        np.testing.assert_array_equal(a[:10],effective_cost(l,2)[:10])
        self.assertTrue(np.all(a>0))
        self.assertTrue(np.all(a<=effective_cost(np.arange(1,30,dtype=float),1)))

    def test_optimism_cannot_stop_earlier(self):
        l=np.random.default_rng(43).lognormal(size=80)
        g=np.linspace(.1,.01,80)
        a=stopping_indices(g,effective_cost(l,0),[.002,.005],4,'sequential')
        b=stopping_indices(g,effective_cost(l,1),[.002,.005],4,'sequential')
        self.assertTrue(np.all(b>=a))


if __name__=='__main__':unittest.main()
