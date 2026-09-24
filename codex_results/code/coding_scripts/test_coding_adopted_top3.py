import unittest
import numpy as np
import coding_theory_bridge as bridge
from algorithm.adaptive_coding import AdaptiveCoding, CodingProfile


class AdoptedTopThreeTests(unittest.TestCase):
    def test_statistic_begins_at_four_not_three(self):
        profile=CodingProfile((0.,1.),(0.,1.),mean_length=10.)
        policy=AdaptiveCoding.from_paper_settings(profile,1e-9,multiplier=1,
                    cost_adjustment=2,minimum=4,cap=20)
        for u in (.1,.2,.8):
            result=policy.observe(u,10)
            self.assertIsNone(result.estimated_improvement)
        result=policy.observe(.3,10)
        self.assertAlmostEqual(result.residual_scale, ((.8-.2)+(.3-.2))/2)
        self.assertAlmostEqual(result.smoothed_residual_scale,result.residual_scale)

    def test_zero_cutoff_and_latched_minimum(self):
        profile=CodingProfile((0.,1.),(0.,1.),mean_length=10.)
        policy=AdaptiveCoding.from_paper_settings(profile,1e-6,multiplier=1,
                    cost_adjustment=0,minimum=8,cap=20)
        for j,u in enumerate((0.,0.,0.,0.,1.,0.,0.,0.),1):
            result=policy.observe(u,10)
            self.assertEqual(result.should_stop,j==8)
        self.assertGreater(result.estimated_improvement,result.estimated_next_cost)
        self.assertEqual(result.total_length,80)
        self.assertEqual(result.best_index,4)

    def test_matches_batched_full_history_rule(self):
        rng=np.random.default_rng(23)
        profile=CodingProfile((0.,1.),(0.,1.),mean_length=10.)
        for _ in range(10):
            values=np.round(rng.uniform(size=50),2)
            lengths=rng.integers(1,500,size=50)
            problem=bridge.old.CodingProblem(tuple(values),(False,)*50,tuple(lengths))
            batch=bridge.old.build_trajectory_batch({"p":problem},{"p":values},{"p":np.arange(50)[None,:]},width=2)
            positive=np.cumsum(values>0)[None,:]
            for price in (1e-6,1e-4,.01):
                rule=bridge.Rule(width=2,smoothing="mean",alpha=1,beta=2,minimum=16,cap=50,guard=False,latch=True)
                expected=bridge.stops(batch,positive,price,rule)[0]
                policy=AdaptiveCoding.from_paper_settings(profile,price,multiplier=1,
                    cost_adjustment=2,minimum=16,cap=50)
                for u,length in zip(values,lengths):
                    result=policy.observe(u,int(length))
                    if result.should_stop: break
                self.assertEqual(result.count,expected)
                self.assertEqual(result.total_length,sum(lengths[:expected]))
                self.assertEqual(result.best_index,int(np.argmax(values[:expected])))

    def test_invalid_statistic_start(self):
        profile=CodingProfile((0.,1.),(0.,1.),mean_length=10.)
        for value in (True,2,2.5,"four"):
            with self.assertRaises(ValueError):
                AdaptiveCoding(profile,1e-6,width=2,statistic_start=value)


if __name__=="__main__":
    unittest.main()
