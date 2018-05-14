
"""
Module for testing kmod.util .
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


    def test_multi_way_split(self):
        test_arr = np.random.randn(20, 3)
        restack = np.vstack(util.multi_way_split(test_arr, [12, 2, 6]))
        testing.assert_almost_equal(test_arr, restack)

        restack = np.vstack(util.multi_way_split(test_arr, [0, 11,1, 2, 0,6]))
        testing.assert_almost_equal(test_arr, restack)

    def test_top_lowzerohigh(self):
        arr = np.array([0, 1, 2, 3, 4])
        A = np.array([0, 1, 2])
        B = A
        C = np.array([4, 3, 2])
        xa, xb, xc = util.top_lowzerohigh(arr, k=3)

        testing.assert_almost_equal(xa, A)
        testing.assert_almost_equal(xb, B)
        testing.assert_almost_equal(xc, C)

    def tearDown(self):
        pass


if __name__ == '__main__':
   unittest.main()

