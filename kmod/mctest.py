"""
Module containing implementations of various tests for model comparison.
"""

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import autograd
import autograd.numpy as np
# Require the kgof package
# https://github.com/wittawatj/kgof
import kgof.data as data
import kgof.util as util
import kgof.kernel as kernel
import matplotlib.pyplot as plt

import scipy
import scipy.stats as stats

class MCTest(object):
    """
    Abstract class for a model comparison test.
    """
    __metaclass__ = ABCMeta

    def __init__(self, p, q, alpha):
        """
        p: a kgof.density.UnnormalizedDensity (model 1)
        q: a kgof.density.UnnormalizedDensity (model 2)
        alpha: significance level of the test
        """
        self.p = p
        self.q = q
        self.alpha = alpha

    @abstractmethod
    def perform_test(self, dat):
        """perform the goodness-of-fit test and return values computed in a dictionary:
        {
            alpha: 0.01, 
            pvalue: 0.0002, 
            test_stat: 2.3, 
            h0_rejected: True, 
            time_secs: ...
        }

        dat: an instance of kgof.data.Data
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stat(self, dat):
        """Compute the test statistic"""
        raise NotImplementedError()

# end of MCTest


