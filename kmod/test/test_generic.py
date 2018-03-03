"""
Module for testing general usages.
"""

__author__ = 'wittawat'

import autograd
import autograd.numpy as np
import numpy.testing as testing
import matplotlib.pyplot as plt

# Import all the submodules for testing purpose
import kmod
import kmod.config
import kmod.mctest
from kmod import data, density, util, kernel
import scipy.stats as stats

import unittest


class Test(unittest.TestCase):
    def setUp(self):
        pass


    def test_dummy(self):
        pass
        #testing.assert_almost_equal(grad_log, my_grad_log)

    def tearDown(self):
        pass


if __name__ == '__main__':
   unittest.main()

