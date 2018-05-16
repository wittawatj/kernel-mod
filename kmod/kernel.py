"""
A module containing kernel functions.
"""

from kgof.kernel import *
import autograd.numpy as np


class KHoPoly(Kernel):
    """Homogeneous polynomial kernel of the form
    (x.dot(y))**d
    """
    def __init__(self, degree):
        assert degree > 0
        self.degree = degree

    def eval(self, X1, X2):
        return np.dot(X1, X2.T)**self.degree

    def pair_eval(self, X, Y):
        return np.sum(X*Y, 1)**self.degree

    def __str__(self):
        return 'KHoPoly(d=%d)'%self.degree

# end of KHoPoly


class KKID(Kernel):
    """KID Kernel"""

    def eval(self, X1, X2):
        d = X1.shape[1]
        assert d == X2.shape[1]
        return (np.dot(X1, X2.T)/d + 1.)**3

    def pair_eval(self, X, Y):
        d = X1.shape[1]
        assert d == Y.shape[1]
        return (np.sum(X1, X2, axis=1)/d + 1.)**3

    def __str__(self):
        return 'KKID'
