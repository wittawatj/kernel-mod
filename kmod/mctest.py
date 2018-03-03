"""
Module containing implementations of various tests for model comparison.
"""
__author__ = 'wittawat'

from builtins import object
from future.utils import with_metaclass

from abc import ABCMeta, abstractmethod

import autograd
import autograd.numpy as np
import kgof.goftest as gof
from kmod import data, density, kernel, util
import matplotlib.pyplot as plt

import scipy
import scipy.stats as stats

class MCTest(with_metaclass(ABCMeta, object)):
    """
    Abstract class for a model comparison (MC) test.
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

# end of MCTest


class MCFSSD(MCTest):
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
        super(MCFSSD, self).__init__(p, q, alpha)
        self.k = k
        self.l = l
        self.V = V
        self.W = W
        # Constrct two FSSD objects
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
        is the variance is biased. The variance is also valid under H0.

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

