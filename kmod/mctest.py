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
from kmod import data, density, kernel, util
import matplotlib.pyplot as plt

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
        self.datap = dataq
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
        assert variance > 0, 'variance of the stat is negative. Was {}'.format(variance)
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
        :param W: Jq x d numpy array of Jq test locations used in FSSD(q, l, W)
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
            # Assume the mean of the null distribution is 0
            pval = stats.norm.sf(stat, loc=0, scale=var**0.5)

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
        assert var_pr > 0
        assert var_qr > 0
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

# end of class SC_UME
