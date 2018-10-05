
"""
Module containing general Pytorch code for neural networks.
"""


from builtins import object
from future.utils import with_metaclass

from abc import ABCMeta, abstractmethod
import torch

class SerializableModule(object):
    """
    An interface to mark that a Pytorch module can be serialized to disk.
    """

    @abstractmethod
    def save(self, f):
        """
        Save the state of this model to a file.
        f can be a file handle or a string representing the file path.
        Subclasses should override this method if needed.
        """
        torch.save(self, f)

    @staticmethod
    def load(f, **opt):
        """
        Load the module as saved by the self.save(f) method.
        Subclasses should override this static method.
        """
        return torch.load(f, **opt)

# end class SerializableModule



