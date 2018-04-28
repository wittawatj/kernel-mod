"""
Module containing implementations of models. 
"""

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import autograd.numpy as np
from future.utils import with_metaclass


class Model(with_metaclass(ABCMeta, object)):
    """
    An abstract class of models. A model can be an implicit model (can only
    sample), or a probabilistic model whose unnormalized density is available.
    intended to be used to represent a model of the data for goodness-of-fit
    testing.
    """
    @abstractmethod
    def get_unnormalized_density(self):
        """
        Return an object of type `kgof.density.UnnormalizedDensity` representing 
        the underlying unnormalized density of this model. 
        Return None if it is not available.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_datasource(self):
        """
        Return a `kgof.data.DataSource` if the model has it.
        A DataSource allows one to sample from the model.
        Return None if not available.
        """
        raise NotImplementedError()

    @abstractmethod
    def dim(self):
        """
        Return the dimension of the input.
        """
        raise NotImplementedError()

    def has_unnormalized_density(self):
        """
        Return true if this model can return an unnormalized density as an
        objective of class `kgof.density.UnnormalizedDensity`.
        To get the density, use `get_unnormalized_density()`.
        """
        return self.get_unnormalized_density() is not None

    def has_datasource(self):
        """
        Return true if this model can provide a `kgof.data.DataSource` which
        allows sample generation.
        """
        return self.get_datasource() is not None

# end of Model

class ComposedModel(with_metaclass(ABCMeta, object)):
    """
    A simple model constructed directly from the specified UnnormalizedDensity
    and/or DataSource.

    * If p is provided and it has a DataSource, and ds is not provided, then
    the DataSource in p will be used (if p has it).
    * If p which has a DataSource is provided, and another ds is provided, then
    the provided DataSource will be used instead of the one in p.
    """

    def __init__(self, p=None, ds=None):
        """
        p: a kgof.densin.UnnormalizedDensity
        ds: a kgof.data.DataSource
        """
        if p is None and ds is None:
            raise ValueError('At least of of the arguments {p, ds} must be specified.')

        if p is not None and ds is None:
            # UnnormalizedDensity given but not DataSource
            # check if a DataSource is available. If so, use it.
            ds = p.get_datasource() # could still be None

        self.p = p
        self.ds = ds

    def get_unnormalized_density(self):
        """
        Return an object of type `kgof.density.UnnormalizedDensity` representing 
        the underlying unnormalized density of this model. 
        Return None if it is not available.
        """
        return self.p

    def get_datasource(self):
        """
        Return a `kgof.data.DataSource` if the model has it.
        A DataSource allows one to sample from the model.
        Return None if not available.
        """
        return self.ds

# end class ComposedModel


