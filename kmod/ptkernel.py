"""
A module containing pytorch kernel functions.
"""

from __future__ import division

from builtins import str
from past.utils import old_div

from kgof.kernel import *
import torch


class PTKGauss(Kernel):
    """
    Pytorch implementation of the isotropic Gaussian kernel.
    Parameterization is the same as in the density of the standard normal
    distribution. sigma2 is analogous to the variance.
    """

    def __init__(self, sigma2):
        """
        sigma2: a number representing squared width
        """
        assert (sigma2 > 0).any(), 'sigma2 must be > 0. Was %s'%str(sigma2)
        self.sigma2 = sigma2

    def eval(self, X, Y):
        """
        Evaluate the Gaussian kernel on the two 2d Torch Tensors

        Parameters
        ----------
        X : n1 x d Torch Tensor
        Y : n2 x d Torch Tensor

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        sigma2 = torch.sqrt(self.sigma2**2)
        sumx2 = torch.sum(X**2, dim=1).view(-1, 1)
        sumy2 = torch.sum(Y**2, dim=1).view(1, -1)
        D2 = sumx2 - 2*torch.matmul(X, Y.transpose(1, 0)) + sumy2
        K = torch.exp(-D2.div(2.0*sigma2))
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d Pytorch tensors

        Return
        -------
        a Torch tensor with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1==n2, 'Two inputs must have the same number of instances'
        assert d1==d2, 'Two inputs must have the same dimension'
        D2 = torch.sum( (X-Y)**2, 1)
        sigma2 = torch.sqrt(self.sigma2**2)
        Kvec = torch.exp(old_div(-D2, (2.0*sigma2)))
        return Kvec

    def __str__(self):
        return "PTKGauss(%.3f)" % self.sigma2

#
