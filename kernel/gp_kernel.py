

from sklearn.gaussian_process.kernels import Kernel as ParentKernel, StationaryKernelMixin, NormalizedKernelMixin
import numpy as np
from sklearn import preprocessing


# Jaccard index or IoU (Intersection over Union)
class Jaccard(StationaryKernelMixin, NormalizedKernelMixin, ParentKernel):


    def __init__(self):
        return


    def __call__(self, X, Y=None):
        X = np.atleast_2d(X)
        scale_factor = np.sum(X[0])*1.0

        if Y is None:
            # print(scale_factor, X.shape)
            X = np.matrix(X)
            prod = X*X.T/scale_factor
            # print("Only Y ",prod, X, Y)
            return prod
        else:
            # print(scale_factor, X.shape, Y.shape)
            X = np.matrix(X)
            Y = np.matrix(Y)
            prod = np.asarray(X*Y.T)/scale_factor
            # print("Both X, Y ", prod)
            return prod


class DotProduct(StationaryKernelMixin, NormalizedKernelMixin, ParentKernel):


    def __init__(self):
        return


    def __call__(self, X, Y=None):
        X = np.atleast_2d(X)

        if Y is None:
            X = np.matrix(preprocessing.normalize(X, norm='l2'))
            prod = np.asarray(X*X.T)
            # if(X.shape[0] < 100):
            #     print(prod)
            # print("Only Y ",prod, X)
            return prod
        else:
            X = np.matrix(preprocessing.normalize(X, norm='l2'))
            Y = np.matrix(preprocessing.normalize(Y, norm='l2'))
            prod = np.asarray(X*Y.T)
            # print(X.shape, Y.shape)
            # if(Y.shape[0] < 100):
            #     print(prod)
            return prod
  