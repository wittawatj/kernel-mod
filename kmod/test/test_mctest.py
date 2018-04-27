"""
Module for testing general usages.
"""

__author__ = 'wittawat'

import autograd
import autograd.numpy as np
import numpy.testing as testing
import matplotlib.pyplot as plt

# Import all the submodules for testing purpose
import kmod
import kmod.config
import kmod.mctest as mct
from kmod import data, density, util, kernel
import scipy.stats as stats

import unittest


class TestDC_FSSD(unittest.TestCase):
    def setUp(self):
        pass

    def test_basic(self):
        """
        Nothing special. Just test basic things.
        """
        seed = 13
        # sample
        n = 103
        alpha = 0.01
        for d in [1, 4]:
            mean = np.zeros(d)
            variance = 1
            p = density.IsotropicNormal(mean, variance)
            q = density.IsotropicNormal(mean, variance+3)

            # only one dimension of the mean is shifted
            #draw_mean = mean + np.hstack((1, np.zeros(d-1)))
            draw_mean = mean +0
            draw_variance = variance + 1
            X = util.randn(n, d, seed=seed)*np.sqrt(draw_variance) + draw_mean
            dat = data.Data(X)

            # Test
            for J in [1, 3]:
                sig2 = util.meddistance(X, subsample=1000)**2
                k = kernel.KGauss(sig2)

                # random test locations
                V = util.fit_gaussian_draw(X, J, seed=seed+1)
                W = util.fit_gaussian_draw(X, J, seed=seed+8)

                mcfssd = mct.DC_FSSD(p, q, k, k, V, W, alpha=0.01)
                s = mcfssd.compute_stat(dat)
                s2, var = mcfssd.get_H1_mean_variance(dat)

                tresult = mcfssd.perform_test(dat)

                # assertions
                self.assertGreaterEqual(tresult['pvalue'], 0)
                self.assertLessEqual(tresult['pvalue'], 1)
                testing.assert_approx_equal(s, (n**0.5)*s2)

    def tearDown(self):
        pass


class TestSC_UME(unittest.TestCase):
    def test_basic(self):
        """
        Test basic things. Make sure SC_UME runs under normal usage.
        """
        mp, varp = 4, 1
        # q cannot be the true model. 
        # That violates our assumption and the asymptotic null distribution
        # does not hold.
        mq, varq = 0.5, 1

        # draw some data
        n = 2999 # sample size
        seed = 89
        with util.NumpySeedContext(seed=seed):
            X = np.random.randn(n, 1)*varp**0.5 + mp
            Y = np.random.randn(n, 1)*varq**0.5 + mq
            Z = np.random.randn(n, 1)
            
            datap = data.Data(X)
            dataq = data.Data(Y)
            datar = data.Data(Z)

        # hyperparameters of the test
        medxz = util.meddistance(np.vstack((X, Z)), subsample=1000)
        medyz = util.meddistance(np.vstack((Y, Z)), subsample=1000)
        k = kernel.KGauss(sigma2=medxz**2)
        l = kernel.KGauss(sigma2=medyz**2)

        # 2 sets of test locations
        J = 3
        Jp = J
        Jq = J
        V = util.fit_gaussian_draw(X, Jp, seed=seed+2)
        W = util.fit_gaussian_draw(Y, Jq, seed=seed+3)

        # construct a UME test
        alpha = 0.01 # significance level 
        scume = mct.SC_UME(datap, dataq, k, l, V, W, alpha=alpha)
        test_result = scume.perform_test(datar)

        # make sure it rejects
        #print(test_result)
        assert test_result['h0_rejected']

if __name__ == '__main__':
   unittest.main()

