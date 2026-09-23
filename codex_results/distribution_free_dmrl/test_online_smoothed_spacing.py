import unittest
import numpy as np
from online_smoothed_spacing import statistics, smooth_numerator
from training_free_odds import prefix_statistics, PRIMARY


class SmoothedTests(unittest.TestCase):
    def test_prefix_only(self):
        r=np.random.default_rng(61).normal(size=60)
        a=statistics(r);r[20:]+=100;b=statistics(r)
        for base in a:
            for mode in ('current','mean','recent_half'):
                np.testing.assert_array_equal(smooth_numerator(a[base],mode)[:20],
                                              smooth_numerator(b[base],mode)[:20])

    def test_reproduces_previous_odds(self):
        r=np.random.default_rng(68).normal(size=40)
        g,_=prefix_statistics(r)
        np.testing.assert_allclose(statistics(r)['odds'],g[PRIMARY]*np.arange(1,41),atol=1e-14)

    def test_averages_only_eligible_prefixes(self):
        v=np.arange(1,21,dtype=float)
        all_mean=smooth_numerator(v,'mean');recent=smooth_numerator(v,'recent_half')
        for n in range(4,21):
            self.assertAlmostEqual(all_mean[n-1],np.mean(v[3:n]))
            self.assertAlmostEqual(recent[n-1],np.mean(v[max(3,n//2):n]))

    def test_score_translation_invariance(self):
        r=np.random.default_rng(91).normal(size=40)
        a=statistics(r);b=statistics(r+100)
        for base in a:
            np.testing.assert_allclose(a[base],b[base],atol=1e-14)


if __name__=='__main__':unittest.main()
