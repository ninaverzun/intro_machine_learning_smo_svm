"""

Course:     20942 - Intro to computational learning
Institute:  Open University of Israel
Assignment: Maman 13
Name:       Nina Verzun
ID:         304680473
Date:       Jan 8 2021

"""
import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler


class SMOsvm:
    """ Implement the Sequential Minimal Optimization (SMO) algorithm designed by J. C. Platt and described in:
     "Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines", 1998.
     """

    def __init__(self, x_in, y_in, c_in, kernel, alphas, b, errors):
        """ Initializing container used for "Sequential Minimal Optimization" (SMO) SVM
        :param x_in: training data vector
        :param y_in: class label vector
        :param c_in: regularization parameter
        :param kernel:  kernel function
        :param alphas: lagrange multiplier vector
        :param b: scalar bias term
        :param errors: error cache
        """
        self.X = x_in
        self.y = y_in
        self.C = c_in
        self.kernel = kernel
        self.lagrange_alphas = alphas
        self.bias = b
        self.errors_cache = errors
        self._obj = []  # record of objective function value
        self.m = len(self.X)  # store size of training set
