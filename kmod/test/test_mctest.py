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

class TestSC_GaussUME(unittest.TestCase):
    def test_optimize_2sets_locs_width(self):
        mp, varp = 2, 1
        # q cannot be the true model. 
        # That violates our assumption and the asymptotic null distribution
        # does not hold.
        mq, varq = 1, 1

        # draw some data
        n = 800 # sample size
        seed = 6
        with util.NumpySeedContext(seed=seed):
            X = np.random.randn(n, 1)*varp**0.5 + mp
            Y = np.random.randn(n, 1)*varq**0.5 + mq
            Z = np.random.randn(n, 1)
            
            datap = data.Data(X)
            dataq = data.Data(Y)
            datar = data.Data(Z)

        # split the data into training/test sets
        [(datptr, datpte), (datqtr, datqte), (datrtr, datrte)] = \
            [D.split_tr_te(tr_proportion=0.3, seed=85) for D in [datap, dataq, datar]]
        Xtr, Ytr, Ztr = [D.data() for D in [datptr, datqtr, datrtr]]

        # initialize optimization parameters.
        # Initialize the Gaussian widths with the median heuristic
        medxz = util.meddistance(np.vstack((Xtr, Ztr)), subsample=1000)
        medyz = util.meddistance(np.vstack((Ytr, Ztr)), subsample=1000)
        gwidth0p = medxz**2
        gwidth0q = medyz**2

        # numbers of test locations in V, W
        J = 2
        Jp = J
        Jq = J

        # pick a subset of points in the training set for V, W
        Xyztr = np.vstack((Xtr, Ytr, Ztr))
        VW = util.subsample_rows(Xyztr, Jp+Jq, seed=73)
        V0 = VW[:Jp, :]
        W0 = VW[Jp:, :]

        # optimization options
        opt_options = {
            'max_iter': 100,
            'reg': 1e-4,
            'tol_fun': 1e-6,
            'locs_bounds_frac': 100,
            'gwidth_lb': None,
            'gwidth_ub': None,
        }

        umep_params, umeq_params = mct.SC_GaussUME.optimize_2sets_locs_width(
            datptr, datqtr, datrtr, V0, W0, gwidth0p, gwidth0q, 
            **opt_options)
        (V_opt, gw2p_opt, opt_infop) = umep_params
        (W_opt, gw2q_opt, opt_infoq) = umeq_params
        k_opt = kernel.KGauss(gw2p_opt)
        l_opt = kernel.KGauss(gw2q_opt)
        # construct a UME test
        alpha = 0.01 # significance level 
        scume_opt2 = mct.SC_UME(datpte, datqte, k_opt, l_opt, V_opt, W_opt, alpha=alpha)
        scume_opt2.perform_test(datrte)

if __name__ == '__main__':
   unittest.main()

