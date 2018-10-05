"""
Module containing code to handle generative models.
"""

from __future__ import division

from builtins import object
from future.utils import with_metaclass

from abc import ABCMeta, abstractmethod
import kmod.net as net
# import numpy as np
import torch
import torch.autograd

class NoiseTransformer(with_metaclass(ABCMeta, object)):
    """
    Class representing functions which transform random noise vectors into
    objects of interests. A Generative Adversarial Network (GAN) model is one
    such an example.
    """

    @abstractmethod
    def forward(self, *input):
        """
        Compute the output of this function given the input.
        """
        pass

    @abstractmethod
    def sample_noise(self, n):
        """
        Sample n noise vectors from the distribution suitable for this function
        to transform. 
        
        Return the n noise vectors.
        """
        pass

    @abstractmethod
    def in_out_shapes(self):
        """
        Return a tuple (I, O), where 
        * I is a tuple describing the input shape of one noise vector
        * O is a tuple describing the output shape of one transformed object.
        """
        pass

    @abstractmethod
    def sample(self, n):
        """
        Sample n objects from this transformer. This involves sampling n noise
        vector from sample_noise(), then transforming them to genereate n
        objects. 

        Return the n objects.
        """
        pass

# end class NoiseTransformer

class PTNoiseTransformer(NoiseTransformer, net.SerializableModule):
    """
    A Pytorch implementation of NoiseTransformer, meaning that all input and
    output are given by Pytorch tensors.
    """
    pass

# end class PTNoiseTransformer


class PTNoiseTransformerAdapter(PTNoiseTransformer):
    """
    A PTNoiseTransformer whose components are specified manually as input.
    Adapter pattern.
    """
    def __init__(self, module, f_sample_noise, in_out_shapes, 
            tensor_type=torch.cuda.FloatTensor
            ):
        """
        ptmodule: a instance of torch.nn.Module represent a function to transform
            noise vectors.
        f_sample_noise: a function or a callable object n |-> (n x
            in_out_shapes[0] ) to sample noise vectors.
        """
        self.module = module
        self.f_sample_noise = f_sample_noise
        self.in_out_shapes = in_out_shapes
        self.tensor_type = tensor_type

        # minimal compatibility check
        try:
            self.sample(1)
        except:
            raise ValueError("Noise sampled from f_sample_noise may be incompatible with the specified transformer module")

    def forward(self, *input):
        return self.module.forward(*input)

    def sample_noise(self, n):
        f = self.f_sample_noise
        return f(n)

    def in_out_shapes(self):
        """
        Return a tuple (I, O), where 
        * I is a tuple describing the input shape of one noise vector
        * O is a tuple describing the output shape of one transformed object.
        """
        return self.in_out_shapes

    def sample(self, n):
        """
        Sample n objects from this transformer. This involves sampling n noise
        vector from sample_noise(), then transforming them to genereate n
        objects. 

        Return the n objects.
        """
        Z = self.sample_noise(n).type(self.tensor_type)
        Zvar = torch.autograd.Variable(Z)
        X = self.module(Zvar)
        return X

# PTNoiseTransformerAdapter



