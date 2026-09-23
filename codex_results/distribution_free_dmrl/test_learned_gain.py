import unittest
from unittest.mock import patch

import numpy as np

from evaluate import q99, sigmoid
from learned_gain_study import (MODEL_PARAMETERS, full_pool_gain, prefix_features,
                                fit_predict, replay)


class LearnedGainTests(unittest.TestCase):
    def test_features_are_prefix_only(self):
        r=np.arange(40,dtype=float);lengths=np.ones(40)*100
        a=prefix_features(r,lengths)
        r[20:]+=1000;lengths[20:]*=100
        b=prefix_features(r,lengths)
        np.testing.assert_array_equal(a[:20],b[:20])

    def test_gain_labels_match_direct_average(self):
        r=np.array([-3,-1,0,1,2,5.],float);maximum=np.array([-3,0,2,5.])
        u=sigmoid(r-q99(r));v=sigmoid(maximum-q99(r))
        expected=np.array([np.maximum(u-s,0).mean() for s in v])
        np.testing.assert_allclose(full_pool_gain(r,maximum),expected,atol=1e-15)

    def test_test_labels_never_affect_predictor(self):
        rng=np.random.default_rng(5)
        features=np.stack([prefix_features(rng.normal(size=40),np.ones(40)*100) for _ in range(4)])[:,None]
        gain=rng.uniform(0,.01,size=(4,1,40))
        with patch.dict(MODEL_PARAMETERS,dict(n_estimators=4,min_samples_leaf=2,max_leaf_nodes=8,n_jobs=1)):
            a=fit_predict(features,gain,np.array([0,1]),np.array([2,3]),'direct_gain')
            gain[2:]=10000
            b=fit_predict(features,gain,np.array([0,1]),np.array([2,3]),'direct_gain')
        np.testing.assert_array_equal(a,b)

    def test_evaluation_quality_does_not_change_stops(self):
        gain=np.ones((1,2,40))*.0001
        costs=np.broadcast_to(np.arange(1,41)*100.,gain.shape)
        a=replay(gain,costs,np.zeros_like(gain));b=replay(gain,costs,np.ones_like(gain))
        np.testing.assert_array_equal(a[...,1:],b[...,1:])
        self.assertTrue(np.all(a[...,2]>=4))


if __name__=='__main__':unittest.main()
