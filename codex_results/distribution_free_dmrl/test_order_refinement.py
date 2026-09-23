import unittest

import numpy as np
from scipy.optimize import linprog

from evaluate import Pool, q99, sigmoid
from order_statistic_refinement import (RATES, WIDTHS, evaluate_reference,
    lower_hull, mix_values, prepare, reference_bias, target_mix, width_sequence)


class RefinementTests(unittest.TestCase):
    def test_mixture_uses_lower_envelope(self):
        q, c = np.array([0., .5, 1.]), np.array([0., .8, 1.])
        self.assertEqual(lower_hull(q, c), [0, 2])
        a, b, w, feasible = target_mix(q, c, .5)
        self.assertEqual((a,b,feasible), (0,2,True))
        self.assertAlmostEqual(w,.5)

    def test_mixture_matches_linear_program(self):
        rng = np.random.default_rng(6)
        for _ in range(20):
            q, c = rng.uniform(size=12), rng.uniform(size=12)
            target = float(rng.uniform(q.min(),q.max()))
            for exact in [False, True]:
                mixture = target_mix(q,c,target,exact=exact)
                actual = mix_values(np.stack([q,c],axis=1)[None,:,:],mixture)[0]
                options = (dict(A_eq=np.stack([np.ones(12),q]),b_eq=[1,target])
                           if exact else dict(A_eq=np.ones((1,12)),b_eq=[1],A_ub=-q[None,:],b_ub=[-target]))
                lp = linprog(c,bounds=(0,None),method='highs',**options)
                self.assertTrue(lp.success)
                self.assertAlmostEqual(actual[1],lp.fun,places=8)
                self.assertGreaterEqual(actual[0]+1e-10,target)

    def test_infeasible_targets_not_extrapolated(self):
        self.assertIsNone(target_mix([.2,.6],[1,2],.8,exact=True))
        self.assertIsNone(target_mix([.2,.6],[1,2],.1,exact=True))
        self.assertFalse(target_mix([.2,.6],[1,2],.8)[3])

    def test_bias_ignores_test_prompt_references(self):
        prefix = np.arange(4*2*40,dtype=float).reshape(4,2,40)
        truth = np.arange(4,dtype=float)
        b = reference_bias(prefix,truth,np.array([0,1]))
        truth[2:] += 1000
        prefix[2:] += 1000
        np.testing.assert_array_equal(b,reference_bias(prefix,truth,np.array([0,1])))

    def test_first_crossing_matches_direct_prefix_rule(self):
        rng = np.random.default_rng(20)
        rewards = rng.normal(size=40)
        lengths = rng.uniform(100,500,size=40)
        pool = Pool('p', rewards, lengths)
        top, refs, cost, quality, truth = prepare([pool],40,1,7)
        order = np.random.default_rng(7).permutation(40)
        actual = evaluate_reference(top,refs,cost,quality)
        for wi, width in enumerate(WIDTHS):
            ks, minimum = width_sequence(width,40)
            for ri, rate in enumerate(RATES):
                stop = 40
                for n in range(minimum,41):
                    scores = rewards[order[:n]]
                    u = np.sort(sigmoid(scores-q99(scores)))
                    k = ks[n-1]
                    gain = np.mean(u[-k:]-u[-k-1])/n
                    if gain <= rate*np.mean(lengths[order[:n]]):
                        stop = n
                        break
                self.assertEqual(actual[0,wi*len(RATES)+ri,2],stop)

    def test_online_decisions_ignore_evaluation_quality(self):
        pool = Pool('p',np.linspace(-2,2,40),np.ones(40)*100)
        top,refs,cost,quality,_ = prepare([pool],40,1,9)
        a = evaluate_reference(top,refs,cost,quality)
        b = evaluate_reference(top,refs,cost,np.ones_like(quality))
        np.testing.assert_array_equal(a[...,1:],b[...,1:])


if __name__ == '__main__':
    unittest.main()
