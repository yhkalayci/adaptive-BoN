import unittest
import numpy as np
from training_free_odds import prefix_statistics,first_stop,PRIMARY


class TrainingFreeTests(unittest.TestCase):
    def test_prefix_only(self):
        r=np.linspace(-2,2,40);a,u=prefix_statistics(r)
        r[20:]+=100;b,v=prefix_statistics(r)
        for name in a:
            np.testing.assert_array_equal(a[name][:20],b[name][:20])
            np.testing.assert_array_equal(u[name][:20],v[name][:20])

    def test_jensen_dominates_empirical_gain(self):
        r=np.random.default_rng(7).normal(size=80);g,u=prefix_statistics(r)
        for mass in ['rank','shape']:
            for ref in ['bound','moment']:
                self.assertTrue(np.all(g[f'{mass}_jensen_{ref}']+1e-14>=g[f'{mass}_empirical_{ref}']))
        self.assertTrue(np.all(u['rank_jensen_bound']<=u['rank_jensen_moment']+1e-14))

    def test_invariance_to_score_location(self):
        r=np.random.default_rng(8).normal(size=40);a,u=prefix_statistics(r);b,v=prefix_statistics(r+100)
        for name in a:
            np.testing.assert_allclose(a[name],b[name],atol=1e-14)
            np.testing.assert_allclose(u[name],v[name],atol=1e-14)

    def test_ties_and_targets(self):
        g,u=prefix_statistics(np.ones(16));lengths=np.ones(16)*100
        self.assertEqual(first_stop(g[PRIMARY],u[PRIMARY],lengths,price=1e-5),4)
        self.assertEqual(first_stop(g[PRIMARY],u[PRIMARY],lengths,target=.4),4)
        self.assertEqual(first_stop(g[PRIMARY],u[PRIMARY],lengths,target=.6),16)

    def test_future_cost_does_not_change_stopping(self):
        g,u=prefix_statistics(np.ones(16));lengths=np.ones(16)
        a=first_stop(g[PRIMARY],u[PRIMARY],lengths,price=1e-5)
        lengths[4:]=100000
        self.assertEqual(a,first_stop(g[PRIMARY],u[PRIMARY],lengths,price=1e-5))


if __name__=='__main__':unittest.main()
