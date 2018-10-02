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
        return 'KHoPoly(d=%d)' % self.degree

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
        return (np.sum(X1*X2, axis=1)/d + 1.)**3

    def __str__(self):
        return 'KKID'

# end KSTKernel


class KGaussPytorch(Kernel):

    def __init__(self, sigma2):
        """
        sigma2: torch.autograd.Variable
        """
        assert sigma2 > 0, 'sigma2 must be > 0. Was %s'%str(sigma2)
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
        sumx2 = torch.sum(X**2, dim=1).view(-1, 1)
        sumy2 = torch.sum(Y**2, dim=1).view(1, -1)
        D2 = sumx2 - 2*torch.matmul(X, Y.transpose(1, 0)) + sumy2
        K = torch.exp(-D2.div(2.0*self.sigma2))
        return K

    def __str__(self):
        return "KGaussPytorch(%.3f)" % self.sigma2


# end KGaussPytorch
