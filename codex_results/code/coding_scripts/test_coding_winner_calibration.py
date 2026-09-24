import itertools
import unittest
import numpy as np
import coding_winner_calibration as c


class WinnerCalibrationTests(unittest.TestCase):
    def test_weights_equal_exhaustive_selection_with_ties(self):
        problem=c.old.CodingProblem((0.,1.,1.,3.),(False,True,False,True),(10,10,10,10))
        weights=c.winner_weights(problem,counts=(1,2,3,4))
        orders=np.array(list(itertools.permutations(range(4))))
        for n in range(1,5):
            prefix=orders[:,:n]
            winners=prefix[np.arange(len(prefix)),np.asarray(problem.rewards)[prefix].argmax(axis=1)]
            np.testing.assert_allclose(weights[n-1],np.bincount(winners,minlength=4)/len(orders))

    def test_fit_is_monotone_bounded_and_uniform_mean_calibrated(self):
        problems={str(i):c.old.CodingProblem((0.,1.,2.,3.),(False,i%2==0,False,True),(10,20,30,40)) for i in range(8)}
        weights={key:c.winner_weights(p,counts=(1,2,4)) for key,p in problems.items()}
        for mixture in c.MIXTURES:
            profile=c.fit_profile(problems,weights,mixture)
            self.assertTrue(np.all(np.diff(profile.probability_knots)>=0))
            self.assertTrue(np.all((np.asarray(profile.probability_knots)>=0)&(np.asarray(profile.probability_knots)<=1)))
            if mixture==0:
                predictions=c.old.calibrated_rewards(problems,profile)
                self.assertAlmostEqual(np.mean(list(predictions.values())),np.mean([p.correct for p in problems.values()]))

    def test_zero_selection_weights_are_supported(self):
        problem=c.old.CodingProblem((0.,1.,2.,3.),(False,False,False,True),(10,10,10,10))
        weights={"p":c.winner_weights(problem,counts=(4,))}
        profile=c.fit_profile({"p":problem},weights,1.)
        self.assertEqual(profile.reward_knots,(3.,))
        self.assertEqual(profile.probability_knots,(1.,))


if __name__=='__main__':
    unittest.main()
