"""
Module containing implementations of various tests for model comparison.
"""
__author__ = 'wittawat'

from builtins import object
from future.utils import with_metaclass

from abc import ABCMeta, abstractmethod

import autograd
import autograd.numpy as np
# kgof can be obtained from https://github.com/wittawatj/kernel-gof
import kgof.goftest as gof
# freqopttest can be obtained from https://github.com/wittawatj/interpretable-test
import freqopttest.tst as tst
import freqopttest.data as tstdata
from kmod import data, density, kernel, util, log
#import matplotlib.pyplot as plt

import scipy
import scipy.stats as stats

class SCTest(with_metaclass(ABCMeta, object)):
    """
    An abstract class for a sample comparison (SC) test.
    This is a kind of a model comparison test where the two models P, Q are
    represented by two samples. Given an input (dat) (in perform_test()), the
    goal is to compare it to the (datap) and (dataq), and determine which of
    the two is closer to (dat).
    """
    def __init__(self, datap, dataq, alpha):
        """
        :param datap: a kmod.data.Data object representing an i.i.d. sample X
            (from model 1)
        :param dataq: a kmod.data.Data object representing an i.i.d. sample Y
            (from model 2)
        :param alpha: significance level of the test
        """
        self.datap = datap
        self.dataq = dataq
        self.alpha = alpha

    @abstractmethod
    def perform_test(self, dat):
        """perform the model comparison test and return values computed in a dictionary:
        {
            alpha: 0.01, 
            pvalue: 0.0002, 
            test_stat: 2.3, 
            h0_rejected: True, 
            time_secs: ...
        }

        :param dat: an instance of kmod.data.Data
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stat(self, dat):
        """
        Compute the test statistic
        :returns: the test statistic (a floating-point number)
        """
        raise NotImplementedError()

# end of SCTest

class DCTest(with_metaclass(ABCMeta, object)):
    """
    Abstract class for a density comparison (DC) test.
    """

    def __init__(self, p, q, alpha):
        """
        :param p: a kmod.density.UnnormalizedDensity (model 1)
        :param q: a kmod.density.UnnormalizedDensity (model 2)
        :param alpha: significance level of the test
        """
        assert(isinstance(p, density.UnnormalizedDensity))
        self.p = p
        self.q = q
        self.alpha = alpha

    @abstractmethod
    def perform_test(self, dat):
        """perform the model comparison test and return values computed in a dictionary:
        {
            alpha: 0.01, 
            pvalue: 0.0002, 
            test_stat: 2.3, 
            h0_rejected: True, 
            time_secs: ...
        }

        :param dat: an instance of kmod.data.Data
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stat(self, dat):
        """
        Compute the test statistic
        :returns: the test statistic (a floating-point number)
        """
        raise NotImplementedError()

# end of DCTest


class DC_FSSD(DCTest):
    """
    A test of for model comparison using the Finite-Set Stein Discrepancy
    (FSSD) as the base discrepancy measure. The FSSD was proposed in 

    | Jitkrittum, W., Xu, W., Szabo, Z., Fukumizu, K., and Gretton, A. 
    | A Linear-Time Kernel Goodness-of-Fit Test. 
    | NIPS 2017

    The statistic is the  \sqrt{n}*(FSSD^2(p, k, V) - FSSD^2(q, l, W)). 
    See the constructor for the meaning of each parameter.
    """
    def __init__(self, p, q, k, l, V, W, alpha=0.01):
        """
        :param p: a kmod.density.UnnormalizedDensity (model 1)
        :param q: a kmod.density.UnnormalizedDensity (model 2)
        :param k: a DifferentiableKernel for defining the Stein function class of p
        :param l: a DifferentiableKernel for defining the Stein function class of q
        :param V: Jp x d numpy array of Jp test locations used in FSSD(p, k, V)
        :param W: Jq x d numpy array of Jq test locations used in FSSD(q, l, W)
        :param alpha: significance level of the test
        """
        super(DC_FSSD, self).__init__(p, q, alpha)
        self.k = k
        self.l = l
        self.V = V
        self.W = W
        # Construct two FSSD objects
        self.fssdp = gof.FSSD(p=p, k=k, V=V, null_sim=None, alpha=alpha)
        self.fssdq = gof.FSSD(p=q, k=l, V=W, null_sim=None, alpha=alpha)
    
    def perform_test(self, dat):
        """
        :param dat: an instance of kmod.data.Data
        """
        with util.ContextTimer() as t:
            alpha = self.alpha
            X = dat.data()
            n = X.shape[0]
            #mean and variance are not yet scaled by \sqrt{n}
            mean, var = self.get_H1_mean_variance(dat)
            stat = (n**0.5)*mean
            # Assume the mean of the null distribution is 0
            pval = stats.norm.sf(stat, loc=0, scale=var**0.5)

        results = {'alpha': self.alpha, 'pvalue': pval, 'test_stat': stat,
                'h0_rejected': pval < alpha, 'time_secs': t.secs, }
        return results

    def compute_stat(self, dat):
        """Compute the test statistic"""
        X = dat.data()
        n = X.shape[0] # n = sample size
        # This returns n*FSSD^2(p, k, V)
        nfssdp2 = self.fssdp.compute_stat(dat)
        nfssdq2 = self.fssdq.compute_stat(dat)
        # want \sqrt{n}*(FSSD^2(p, k, V) - FSSD^2(q, l, W))
        s = (nfssdp2-nfssdq2)/(n**0.5)
        return s

    def get_H1_mean_variance(self, dat):
        """
        Return the mean and variance under H1 of the 
        test statistic = \sqrt{n}(FSSD(p)^2 - FSSD(q)^2).
        The estimator of the mean is unbiased (can be negative). The estimator
        of the variance is biased. The variance is also valid under H0.

        :returns: (mean, variance)
        """
        fssdp = self.fssdp
        fssdq = self.fssdq
        X = dat.data()

        # Feature tensor: n x d x Jp where n = sample size.
        Xip = fssdp.feature_tensor(X)
        n, d, Jp = Xip.shape
        # Feature tensor: n x d x Jq where n = sample size.
        Xiq = fssdq.feature_tensor(X)
        Jq = Xiq.shape[2]
        assert Xiq.shape[0] == n
        assert Xiq.shape[1] == d

        statp, varp = gof.FSSD.ustat_h1_mean_variance(Xip, return_variance=True, use_unbiased=True)
        assert varp > 0
        statq, varq = gof.FSSD.ustat_h1_mean_variance(Xiq, return_variance=True, use_unbiased=True)
        assert varq > 0
        mean_h1 = statp - statq

        # compute the cross covariance (i.e., diagonal entries of the
        # covariance of the asymptotic joint normal).
        # mu: d*J vector
        Taup = np.reshape(Xip, [n, d*Jp])
        Tauq = np.reshape(Xiq, [n, d*Jq])
        # length-d*Jp vector
        mup = np.mean(Taup, 0)
        muq = np.mean(Tauq, 0)
        varpq = 4.0*np.mean(np.dot(Taup, mup)*np.dot(Tauq, muq) ) - 4.0*np.sum(mup**2)*np.sum(muq**2)
        variance = varp - 2.0*varpq + varq
        if variance <= 0:
            log.l().warning('variance of the stat is not positive. Was {}'.format(variance))
        return mean_h1, variance

# end of MCFSSD


class SC_UME(SCTest):
    """
    A test of for model comparison using the unnormalized ME (UME) statistic
    as the base discrepancy measure. The UME statistic was mentioned (but not
    studied) in Chwialkovski et al., 2015 (NIPS), Jitkrittum et al., 2016 (NIPS).

    Terminology:
    * P = model 1
    * Q = model 2
    * R = data generating distribution (unknown)

    In constrast to DC_FSSD, the MCUME test is a three-sample test, meaning that 
    the two models P, Q are represented by two samples.
    """
    def __init__(self, datap, dataq, k, l, V, W, alpha=0.01):
        """
        :param datap: a kmod.data.Data object representing an i.i.d. sample X
            (from model 1)
        :param dataq: a kmod.data.Data object representing an i.i.d. sample Y
            (from model 2)
        :param k: a kmod.Kernel 
        :param l: a kmod.Kernel
        :param V: Jp x d numpy array of Jp test locations used in UME(q, r)
        :param W: Jq x d numpy array of Jq test locations used in UME(q, r)
        :param alpha: significance level of the test
        """
        super(SC_UME, self).__init__(datap, dataq, alpha)
        self.k = k
        self.l = l
        self.V = V
        self.W = W
        # Constrct two UMETest objects
        self.umep = tst.UMETest(V, k)
        self.umeq = tst.UMETest(W, l)

    def compute_stat(self, dat):
        """
        Compute the test statistic:
            test statistic = \sqrt{n}(UME(P, R)^2 - UME(Q, R))^2.
            
        dat: an instance of kmod.data.Data or kgof.data.Data
        """
        mean_h1 = self.get_H1_mean_variance(dat, return_variance=False)
        n = dat.sample_size()
        return (n**0.5)*mean_h1

    def perform_test(self, dat):
        """
        :param dat: an instance of kmod.data.Data
        """
        with util.ContextTimer() as t:
            alpha = self.alpha
            X = dat.data()
            n = X.shape[0]
            #mean and variance are not yet scaled by \sqrt{n}
            # The variance is the same for both H0 and H1.
            mean_h1, var = self.get_H1_mean_variance(dat)
            stat = (n**0.5)*mean_h1
            null_std = var**0.5
            if null_std <= 1e-5:
                log.l().warning('SD of the null distribution is too small. Was {}. Will not reject H0.'.format(null_std))
                pval = np.inf
            else:
                # Assume the mean of the null distribution is 0
                pval = stats.norm.sf(stat, loc=0, scale=null_std)

        results = {'alpha': self.alpha, 'pvalue': pval, 'test_stat': stat,
                'h0_rejected': pval < alpha, 'time_secs': t.secs, }
        return results

    def get_H1_mean_variance(self, dat, return_variance=True):
        """
        Return the mean and variance under H1 of the 
        test statistic = \sqrt{n}(UME(P, R)^2 - UME(Q, R))^2.
        The estimator of the mean is unbiased (can be negative). The variance
        is also valid under H0.

        :returns: (mean, variance)

        If return_variance is False, 
        :returns: mean
        """
        umep = self.umep
        umeq = self.umeq
        # form a two-sample test dataset between datap and dat (data from R)
        Z = dat.data()
        datapr = tstdata.TSTData(self.datap.data(), Z)
        dataqr = tstdata.TSTData(self.dataq.data(), Z)

        # get the feature matrices (correlated)
        fea_pr = umep.feature_matrix(datapr) # n x Jp
        fea_qr = umeq.feature_matrix(dataqr) # n x Jq
        assert fea_pr.shape[1] == self.V.shape[0]
        assert fea_qr.shape[1] == self.W.shape[0]

        # umehp = ume_hat(p, r)
        umehp, var_pr = tst.UMETest.ustat_h1_mean_variance(fea_pr,
                return_variance=True, use_unbiased=True)
        umehq, var_qr = tst.UMETest.ustat_h1_mean_variance(fea_qr,
                return_variance=True, use_unbiased=True)

        if var_pr <= 0:
            log.l().warning('Non-positive var_pr detected. Was {}'.format(var_pr))
        if var_qr <= 0:
            log.l().warning('Non-positive var_qr detected. War {}'.format(var_qr))
        #assert var_pr > 0, 'var_pr was {}'.format(var_pr)
        #assert var_qr > 0, 'var_qr was {}'.format(var_qr)
        mean_h1 = umehp - umehq

        if not return_variance:
            return mean_h1

        # mean features
        mean_pr = np.mean(fea_pr, axis=0)
        mean_qr = np.mean(fea_qr, axis=0)
        t1 = 4.0*np.mean(np.dot(fea_pr, mean_pr)*np.dot(fea_qr, mean_qr))
        t2 = 4.0*np.sum(mean_pr**2)*np.sum(mean_qr**2)

        # compute the cross-covariance
        var_pqr = t1-t2
        var_h1 = var_pr -2.0*var_pqr + var_qr
        return mean_h1, var_h1

    @staticmethod
    def power_criterion(datap, dataq, datar, k, l, V, W, reg=1e-3): 
        """
        Compute the mean and standard deviation of the three-sample test
        statistic under H1. 

        :param datap: kgof.data.Data. data from P (model 1)
        :param dataq: kgof.data.Data. data from Q (model 2)
        :param datar: kgof.data.Data. data from R (data generating distribution)
        :param k: kmod.kernel.Kernel for UME(P, R)
        :param l: kmod.kernel.Kernel for UME(Q, R)
        :param V: Jp x d numpy array of Jp test locations for UME(P, R)
        :param W: Jq x d numpy array of Jq test locations for UME(Q, R)
        :param reg: regularization parameter
        
        Return power criterion = mean_under_H1/sqrt(var_under_H1 + reg) .
        """
        scume = SC_UME(datap, dataq, k, l, V, W)
        mean_h1, var_h1 = scume.get_H1_mean_variance(datar, return_variance=True)
        ratio = mean_h1/np.sqrt(var_h1 + reg)
        return ratio

# end of class SC_UME

class SC_GaussUME(SC_UME):
    """
    A SC_UME using two Gaussian kernels.
    """
    def __init__(self, datap, dataq, gwidth2p, gwidth2q, V, W, alpha=0.01):
        """
        :param datap: a kmod.data.Data object representing an i.i.d. sample X
            (from model 1)
        :param dataq: a kmod.data.Data object representing an i.i.d. sample Y
            (from model 2)
        :param gwidth2p: squared Gaussian width for UME(P, R)
        :param gwidth2q: squared Gaussian width for UME(Q, R)
        :param V: Jp x d numpy array of Jp test locations used in UME(p, r)
        :param W: Jq x d numpy array of Jq test locations used in UME(q, r)
        :param alpha: significance level of the test
        """
        if not util.is_real_num(gwidth2p) or gwidth2p <= 0:
            raise ValueError('gwidth2p must be positive real. Was {}'.format(gwidth2p))
        if not util.is_real_num(gwidth2q) or gwidth2q <= 0:
            raise ValueError('gwidth2q must be positive real. Was {}'.format(gwidth2q))

        k = kernel.KGauss(gwidth2p)
        l = kernel.KGauss(gwidth2q)
        super(SC_UME, self).__init__(datap, dataq, k, l, V, W, alpha)

    @staticmethod
    def optimize_3sample_criterion(datap, dataq, datar, V0, gwidth0, reg=1e-3,
            max_iter=100, tol_fun=1e-6, disp=False, locs_bounds_frac=100,
            gwidth_lb=None, gwidth_ub=None):
        """
        Similar to optimize_2sets_locs_widths() but constrain V=W, and
        constrain the two Gaussian widths to be the same.
        Optimize one set of test locations and one Gaussian kernel width by
        maximizing the test power criterion of the UME *three*-sample test             

        This optimization function is deterministic.

        - datap: a kgof.data.Data from P (model 1)
       - dataq: a kgof.data.Data from Q (model 2)
        - datar: a kgof.data.Data from R (data generating distribution)
        - V0: Jxd numpy array. Initial V containing J locations. For both
              UME(P, R) and UME(Q, R)
        - gwidth0: initial value of the Gaussian width^2 for both UME(P, R),
              and UME(Q, R)
        - reg: reg to add to the mean/sqrt(variance) criterion to become
            mean/sqrt(variance + reg)
        - max_iter: #gradient descent iterations
        - tol_fun: termination tolerance of the objective value
        - disp: True to print convergence messages
        - locs_bounds_frac: When making box bounds for the test_locs, extend
              the box defined by coordinate-wise min-max by std of each
              coordinate (of the aggregated data) multiplied by this number.
        - gwidth_lb: absolute lower bound on both the Gaussian width^2
        - gwidth_ub: absolute upper bound on both the Gaussian width^2

        If the lb, ub bounds are None, use fraction of the median heuristics 
            to automatically set the bounds.
        
        Return (optimized V, optimized Gaussian width^2, info from the optimization)
        """
        J = V0.shape[0]
        X, Y, Z = datap.data(), dataq.data(), datar.data()
        n, d = X.shape

        # Parameterize the Gaussian width with its square root (then square later)
        # to automatically enforce the positivity.
        def obj(sqrt_gwidth, V):
            k = kernel.KGauss(sqrt_gwidth**2)
            return -SC_UME.power_criterion(datap, dataq, datar, k, k, V, V,
                    reg=reg)

        flatten = lambda gwidth, V: np.hstack((gwidth, V.reshape(-1)))
        def unflatten(x):
            sqrt_gwidth = x[0]
            V = np.reshape(x[1:], (J, d))
            return sqrt_gwidth, V

        def flat_obj(x):
            sqrt_gwidth, V = unflatten(x)
            return obj(sqrt_gwidth, V)

        # Initial point
        x0 = flatten(np.sqrt(gwidth0), V0)
        
        #make sure that the optimized gwidth is not too small or too large.
        XYZ = np.vstack((X, Y, Z))
        med2 = util.meddistance(XYZ, subsample=1000)**2
        fac_min = 1e-2 
        fac_max = 1e2
        if gwidth_lb is None:
            gwidth_lb = max(fac_min*med2, 1e-3)
        if gwidth_ub is None:
            gwidth_ub = min(fac_max*med2, 1e5)

        # Make a box to bound test locations
        XYZ_std = np.std(XYZ, axis=0)
        # XYZ_min: length-d array
        XYZ_min = np.min(XYZ, axis=0)
        XYZ_max = np.max(XYZ, axis=0)
        # V_lb: J x d
        V_lb = np.tile(XYZ_min - locs_bounds_frac*XYZ_std, (J, 1))
        V_ub = np.tile(XYZ_max + locs_bounds_frac*XYZ_std, (J, 1))
        # (J*d+1) x 2. Take square root because we parameterize with the square
        # root
        x0_lb = np.hstack((np.sqrt(gwidth_lb), np.reshape(V_lb, -1)))
        x0_ub = np.hstack((np.sqrt(gwidth_ub), np.reshape(V_ub, -1)))
        x0_bounds = list(zip(x0_lb, x0_ub))

        # optimize. Time the optimization as well.
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
        grad_obj = autograd.elementwise_grad(flat_obj)
        with util.ContextTimer() as timer:
            opt_result = scipy.optimize.minimize(
              flat_obj, x0, method='L-BFGS-B', 
              bounds=x0_bounds,
              tol=tol_fun, 
              options={
                  'maxiter': max_iter, 'ftol': tol_fun, 'disp': disp,
                  'gtol': 1.0e-08,
                  },
              jac=grad_obj,
            )

        opt_result = dict(opt_result)
        opt_result['time_secs'] = timer.secs
        x_opt = opt_result['x']
        sq_gw_opt, V_opt = unflatten(x_opt)
        gw_opt = sq_gw_opt**2

        assert util.is_real_num(gw_opt), 'gw_opt is not real. Was %s' % str(gw_opt)
        return V_opt, gw_opt, opt_result


    @staticmethod
    def optimize_2sets_locs_widths(datap, dataq, datar, V0, W0, gwidth0p,
            gwidth0q, reg=1e-3, max_iter=100,  tol_fun=1e-6, disp=False,
            locs_bounds_frac=100, gwidth_lb=None, gwidth_ub=None):
        """
        Optimize two sets of test locations and the Gaussian kernel widths by
        maximizing the test power criterion of the UME two-sample test (not
        three-sample test). Briefly,
            1. Optimize the set V of test locations for UME(P, R) by maximizing
            its two-sample test power criterion.
            2. Optimize the set W for UME(Q, R) in the same way.

        The two optimization problems are independent. The only dependency is
        the data from R. This optimization function is deterministic.

        - datap: a kgof.data.Data from P (model 1)
        - dataq: a kgof.data.Data from Q (model 2)
        - datar: a kgof.data.Data from R (data generating distribution)
        - V0: Jpxd numpy array. Initial V.
        - W0: Jqxd numpy array. Initial W.
        - gwidth0p: initial value of the Gaussian width^2 for UME(P, R)
        - gwidth0q: initial value of the Gaussian width^2 for UME(Q, R)
        - reg: reg to add to the mean/sqrt(variance) criterion to become
            mean/sqrt(variance + reg)
        - max_iter: #gradient descent iterations
        - tol_fun: termination tolerance of the objective value
        - disp: True to print convergence messages
        - locs_bounds_frac: When making box bounds for the test_locs, extend
              the box defined by coordinate-wise min-max by std of each coordinate
              (of the aggregated data) multiplied by this number.
        - gwidth_lb: absolute lower bound on both the Gaussian width^2
        - gwidth_ub: absolute upper bound on both the Gaussian width^2

        If the lb, ub bounds are None, use fraction of the median heuristics 
            to automatically set the bounds.
        
        Return (  
            (V test_locs, gaussian width^2 for UME(P, R), optimization info log),
            (W test_locs, gaussian width^2 for UME(Q, R), optimization info log),
                )
        """

        Z = datar.data()
        datapr = tstdata.TSTData(datap.data(), Z)
        dataqr = tstdata.TSTData(dataq.data(), Z)

        # optimization for UME(P,R)
        V_opt, gw2p_opt, opt_infop = \
        tst.GaussUMETest.optimize_locs_width(datapr, V0, gwidth0p, reg=reg,
                max_iter=max_iter, tol_fun=tol_fun, disp=disp,
                locs_bounds_frac=locs_bounds_frac, gwidth_lb=gwidth_lb,
                gwidth_ub=gwidth_ub)

        # optimization for UME(Q,R)
        W_opt, gw2q_opt, opt_infoq = \
        tst.GaussUMETest.optimize_locs_width(dataqr, W0, gwidth0q, reg=reg,
                max_iter=max_iter, tol_fun=tol_fun, disp=disp,
                locs_bounds_frac=locs_bounds_frac, gwidth_lb=gwidth_lb,
                gwidth_ub=gwidth_ub)

        return ( (V_opt, gw2p_opt, opt_infop), (W_opt, gw2q_opt, opt_infoq) )

# end class SC_GaussUME

class SC_MMD(SCTest):
    """
    A test for model comparison using the Maximum Mean Discrepancy (MMD) proposed 
    by Bounliphone, et al 2016 (ICLR)
    """

    def __init__(self, datap, dataq, k, alpha):
        """
        :param datap: a kmod.data.Data object representing an i.i.d. sample X
            (from model 1)
        :param dataq: a kmod.data.Data object representing an i.i.d. sample Y
            (from model 2)
        :param k: a kmod.Kernel
        :param alpha: significance level of the test
        """
        super(SC_MMD, self).__init__(datap, dataq, alpha)
        self.k = k

    def perform_test(self, dat):
        """perform the model comparison test and return values computed in a dictionary:
        {
            alpha: 0.01,
            pvalue: 0.0002,
            test_stat: 2.3,
            h0_rejected: True,
            time_secs: ...
        }

        :param dat: an instance of kmod.data.Data
        """
        with util.ContextTimer() as t:
            alpha = self.alpha
            X = dat.data()
            n = X.shape[0]
            # mean and variance are not yet scaled by \sqrt{n}
            # The variance is the same for both H0 and H1.
            mean_h1, var = self.get_H1_mean_variance(dat)
            stat = (n**0.5) * mean_h1
            # Assume the mean of the null distribution is 0
            pval = stats.norm.sf(stat, loc=0, scale=var**0.5)

        results = {
            'alpha': self.alpha, 'pvalue': pval, 'test_stat': stat,
            'h0_rejected': pval < alpha, 'time_secs': t.secs,
        }
        return results

    def compute_stat(self, dat):
        """
        Compute the test statistic
        :returns: the test statistic (a floating-point number)
        """
        mean_h1 = self.get_H1_mean_variance(dat, return_variance=False)
        n = dat.sample_size()
        return (n**0.5) * mean_h1

    def get_H1_mean_variance(self, dat, return_variance=True):
        """
        Return the mean and variance under H1 of the 
        test statistic = (MMD_u(Z_{n_z}, X_{n_x})^2 - MMD_u(Z_{n_z}, Y_{n_y})^2)^2.
        The estimator of the mean is unbiased (can be negative). The estimator
        of the variance is also unbiased. The variance is also valid under H0.

        :returns: (mean, variance)
        """
        # form a two-sample test dataset between datap and dat (data from R)
        Z = dat.data()
        X = self.datap.data()
        Y = self.dataq.data()
        # This always return a variance. But will be None if is_var_computed=False
        mmd_mean_pr, var_pr = tst.QuadMMDTest.h1_mean_var(X, Z, self.k,
                is_var_computed=return_variance)
        mmd_mean_qr, var_qr = tst.QuadMMDTest.h1_mean_var(Y, Z, self.k,
                is_var_computed=return_variance)
        mean_h1 = mmd_mean_pr - mmd_mean_qr
        if not return_variance:
            return mean_h1

        var_pqr = self.get_cross_covariance(X, Y, Z, self.k)
        var_h1 = var_pr - 2.0*var_pqr + var_qr
        return mean_h1, var_h1

    @staticmethod
    def get_cross_covariance(X, Y, Z, k):
        """
        Compute the covariance of the U-statistics for two MMDs
        (Bounliphone, et al. 2016, ICLR) 

        Args:
            X: numpy array of shape (nx, d), sample from the model 1 
            Y: numpy array of shape (ny, d), sample from the model 2
            Z: numpy array of shape (nz, d), sample from the reference
            k: a kernel object 

        Returns:
            cov: covariance of two U stats 
        """
        Kzz = k.eval(Z, Z)
        # Kxx
        Kzx = k.eval(Z, X)
        # Kxy
        Kzy = k.eval(Z, Y)
        # Kxz
        Kzznd = Kzz - np.diag(np.diag(Kzz))
        # Kxxnd = Kxx-diag(diag(Kxx));

        nz = Kzz.shape[0]
        nx = Kzx.shape[1]
        ny = Kzy.shape[1]
        # m = size(Kxx,1);
        # n = size(Kxy,2);
        # r = size(Kxz,2);

        u_zz = (1./(nz*(nz-1))) * np.sum(Kzznd)
        u_zx = np.sum(Kzx) / (nz*nx)
        u_zy = np.sum(Kzy) / (nz*ny)
        # u_xx=sum(sum(Kxxnd))*( 1/(m*(m-1)) );
        # u_xy=sum(sum(Kxy))/(m*n);
        # u_xz=sum(sum(Kxz))/(m*r);

        ct1 = 1./(nz*(nz-1)**2) * np.sum(np.dot(Kzznd,Kzznd))
        # ct1 = (1/(m*(m-1)*(m-1)))   * sum(sum(Kzznd*Kzznd));
        ct2 = u_zz**2
        # ct2 =  u_xx^2;
        ct3 = 1./(nz*(nz-1)*ny) * np.sum(np.dot(Kzznd,Kzy))
        # ct3 = (1/(m*(m-1)*r))       * sum(sum(Kzznd*Kxz));
        ct4 = u_zz * u_zy
        # ct4 =  u_xx*u_xz;
        ct5 = (1./(nz*(nz-1)*nx)) * np.sum(np.dot(Kzznd, Kzx))
        # ct5 = (1/(m*(m-1)*n))       * sum(sum(Kzznd*Kxy));
        ct6 = u_zz * u_zx
        # ct6 = u_xx*u_xy;
        ct7 = (1./(nx*nz*ny)) * np.sum(np.dot(Kzx.T, Kzy))
        # ct7 = (1/(n*m*r))           * sum(sum(Kzx'*Kxz));
        ct8 = u_zx * u_zy
        # ct8 = u_xy*u_xz;

        zeta_1 = (ct1-ct2)-(ct3-ct4)-(ct5-ct6)+(ct7-ct8)
        # zeta_1 = (ct1-ct2)-(ct3-ct4)-(ct5-ct6)+(ct7-ct8);
        cov = (4.0*(nz-2))/(nz*(nz-1)) * zeta_1
        # theCov = (4*(m-2))/(m*(m-1)) * zeta_1;
        return cov    

# end of class SC_UME
