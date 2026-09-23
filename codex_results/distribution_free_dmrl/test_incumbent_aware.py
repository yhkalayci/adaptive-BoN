import unittest
import numpy as np

from evaluate import Pool
from order_statistic_refinement import prepare, evaluate_reference
from incumbent_aware_study import evaluate, tail_numerators


class IncumbentTests(unittest.TestCase):
    def test_spacing_reproduces_previous_policy(self):
        rng=np.random.default_rng(2)
        p=Pool('p',rng.normal(size=40),rng.uniform(100,500,size=40))
        top,ref,cost,q,_=prepare([p],40,2,13)
        np.testing.assert_allclose(evaluate(top,ref,cost,q)['spacing'],evaluate_reference(top,ref,cost,q))

    def test_better_incumbent_reduces_envelopes(self):
        top=np.broadcast_to(np.linspace(0,-4,33),(1,1,40,33)).copy()
        ref=np.zeros((1,1,40))
        old=tail_numerators(top,ref,'fixed4')
        top[...,0]+=3
        new=tail_numerators(top,ref,'fixed4')
        for family in ['dmrl_envelope','moment_envelope']:
            self.assertTrue(np.all(new[family][0][...,6:]<old[family][0][...,6:]))
        self.assertTrue(np.all(new['spacing'][0][...,6:]>old['spacing'][0][...,6:]))

    def test_evaluation_quality_never_changes_stops(self):
        p=Pool('p',np.linspace(-1,1,40),np.ones(40)*300)
        top,ref,cost,q,_=prepare([p],40,1,11)
        a=evaluate(top,ref,cost,q);b=evaluate(top,ref,cost,1-q)
        for family in a:np.testing.assert_array_equal(a[family][...,1:],b[family][...,1:])

    def test_future_scores_do_not_change_prefix_estimates(self):
        top=np.broadcast_to(np.linspace(0,-4,33),(1,1,40,33)).copy()
        ref=np.zeros((1,1,40))
        old=tail_numerators(top,ref,'quarter32')
        top[...,20:,:]+=10
        new=tail_numerators(top,ref,'quarter32')
        for family in old:np.testing.assert_array_equal(old[family][0][...,:20],new[family][0][...,:20])


if __name__=='__main__':unittest.main()
