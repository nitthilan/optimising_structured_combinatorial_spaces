from __future__ import print_function
from __future__ import division
import numpy as np
from datetime import datetime
from scipy.stats import norm
from scipy.optimize import minimize
#from scipydirect import minimize
# import scipydirect as sd
# import utility_functions as U

import sys

# Not used currently. Can be commented out when required
# def acq_max_random(ac, reg, y_max):
#     """
#     A function to find the maximum of the acquisition function
#     It just uses random sampling to determine the position
#     Parameters
#     ----------
#     :param ac:
#         The acquisition function object that return its point-wise value.

#     :param gp:
#         A gaussian process fitted to the relevant data.

#     :param y_max:
#         The current maximum known value of the target function.

#     """
#     x_tries = ld.generate_sw_feature(1000)
#     ys = ac(x_tries, reg, y_max)
#     x_max = x_tries[ys.argmax()]
#     max_acq = ys.max()

#     return x_max

# Direct Algo has to be fixed properly. So currently commnting it out
# Ideally to be moved to a different file    
# global_counter = 0
# def acq_max_DIRECT(ac, reg, y_max):
#     N = U.GET_NUM_CONNECTION()
#     bound = np.zeros((N,2))
#     for i in range(N):
#         bound[i,0] = 0.0
#         bound[i,1] = 1.0

#     global global_counter 
#     global_counter = 0
#     OptimizeResult = sd.minimize(ac, bound, args=(reg, y_max))
#     print(global_counter, OptimizeResult.fun)

#     #x, fmin, ierror = solve(ac, l, u, user_data=(gp, y_max))
#     #print (x, fmin, ierror)
#     # print (OptimizeResult.fun)
#     #print (OptimizeResult.x, OptimizeResult.fun)
#     (connection_list_list, connection_idx_list_list) = \
#         ld.LinkDistribution().generate_valid_graph(x_feature = OptimizeResult.x)
#     #print(connection_list_list)
#     # Expects a array. Try modifying it to work with numpy arrays
#     feature_vector = \
#         U.generate_feature_list([connection_idx_list_list])

#     # feature_vector = OptimizeResult.x
#     # feature_vector[feature_vector >= 0.5] = 1
#     # feature_vector[feature_vector <  0.5] = 0
#     # print("Feature vector ", feature_vector)

#     # opt_mean, opt_std = gp.predict(OptimizeResult.x, return_std=True)
#     # approx_mean, approx_std = gp.predict(feature_numpy_array, return_std=True)

#     # print(OptimizeResult.fun, OptimizeResult.message, OptimizeResult.success)
#     # print(opt_mean, opt_std, approx_mean, approx_std)


#     # print(feature_numpy_array)
#     # ld.approximate_sw_feature(x)
#     return feature_vector #acq_max_random(ac, gp, y_max)

# def acq_max(ac, gp, y_max, 
#     # bounds
#     ):
"""
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling 1e5 points at random, and then
    running L-BFGS-B from 250 random starting points.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.


    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """

    # Warm up with random points
    # x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],
    #                              size=(100000, bounds.shape[0]))
    

    # # Explore the parameter space more throughly
    # x_seeds = np.random.uniform(bounds[:, 0], bounds[:, 1],
    #                             size=(250, bounds.shape[0]))
    # for x_try in x_seeds:
    #     # Find the minimum of minus the acquisition function
    #     res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
    #                    x_try.reshape(1, -1),
    #                    bounds=bounds,
    #                    method="L-BFGS-B")

    #     # Store it if better than previous minimum(maximum).
    #     if max_acq is None or -res.fun[0] >= max_acq:
    #         x_max = res.x
    #         max_acq = -res.fun[0]

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    # return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    # return x_max


# https://stackoverflow.com/questions/20615750/how-do-i-output-the-regression-prediction-from-each-tree-in-a-random-forest-in-p
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.predict
def get_random_forest(reg_model, feature_vector):
    pred_mean_dir = reg_model.predict(feature_vector)
    pred_val = np.asarray([tree.predict(feature_vector) for tree in reg_model.estimators_])
    # pred_mean = np.mean(pred_val, axis=0)
    # print("RF values ", pred_mean_dir, pred_mean, np.std(pred_val, axis=0))
    # print("Pred Val ", pred_val, np.mean(pred_val, axis=0), np.std(pred_val, axis=0))
    return(pred_mean_dir, np.std(pred_val, axis=0))

class ModelRegressor(object):

    def __init__(self, kind, reg):
        if kind not in ['gp', 'rf']:
            err = "The ModelRegressor " \
                  "{} has not been implemented, " \
                  "please choose one of gp, rf.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

        self.reg = reg
        # self.fg = fg

    def predict(self, x):
        # print("Pred Dimension ", x.shape, x)
        # x = x.dot(2**np.arange(x.shape[1]-1,-1,-1))
        # x = np.expand_dims(x, axis=1)
        # print("Pred Dimension ", x.shape, x)

        # print(x.shape)
        if self.kind == 'gp':
            return self.reg.predict(x, return_std=True)
        elif self.kind == 'rf':
            return get_random_forest(self.reg, x)

    def fit(self, X, Y):
        # X = X.dot(2**np.arange(X.shape[1]-1,-1,-1))
        # X = np.expand_dims(X, axis=1)
        
        # print("Fit Dimension ", X.shape, X)
        # print("Shape  ", X.shape, Y)
        return self.reg.fit(X, Y)

    def set_params(self, **params):
        return self.reg.set_params(**params)

class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.kappa = kappa

        self.xi = xi

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    # Direct Algo has to be fixed properly. So currently commnting it out
    # Ideally to be moved to a different file    
    # def utility_DIRECT(self, x, reg, y_max):
    #     global global_counter
    #     global_counter = global_counter + 1
    #     # print (x.shape)
    #     x[x >= 0.5] = 1
    #     x[x <  0.5] = 0
    #     # print ("X Vector", x)
    #     if(self.ld.check_sw_connectivity(x)):
    #         x = x.reshape(1, -1)
    #         print("Found a valid graph :) ")
    #         return self.utility(x, reg, y_max)
    #         # return -1*self.utility(x, gp, y_max)
    #     else:
    #         # print("Returning a large negative value")
    #         return -999999
        
    
    def utility(self, x, reg, y_max):
        

        if self.kind == 'ucb':
            return self._ucb(x, reg, y_max, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, reg, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, reg, y_max, self.xi)

    @staticmethod
    def _ucb(x, reg, y_max, kappa):
        mean, std = reg.predict(x)
        # print(mean)
        # print(std)
        kappa = 1.0
        ucb = mean + kappa * std

        # ucb[ucb<=y_max] = 0.0
        return ucb

    @staticmethod
    def _ei(x, reg, y_max, xi):
        y_max += 0.001
        mean, std = reg.predict(x)

        # Below is the test for checking why all the mean are same
        # One probable reason is because we are searching for near by points 
        # to the base design and so the probability of all the designs to be same is 
        # high
        # print(x.shape)
        # for i in range(x.shape[0]):
        #     mean_x, std_x = reg.predict(x[i].reshape(1, -1))
        #     # print(mean_x, std_x)
        #     if(mean[i] - mean_x > 0.0000001):
        #         print(i, mean[i], mean_x)

        # if(np.all(mean - mean[0] < 0.0000001)):
        #     print("all means same" , mean[0])

        # mean = np.squeeze(np.asarray(mean1))
        # std = std.squeeze()
        z = (mean - y_max - xi)/std
    
        # print(mean, std, z, xi, y_max, norm.cdf(z), norm.pdf(z))        
        # print(mean.shape, std.shape, type(mean))
        output = (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

        # CODE FROM SMAC implementation
        # z = (self.eta - m - self.par) / s
        # f = (self.eta - m - self.par) * norm.cdf(z) + s * norm.pdf(z)
 
        # Check for all values of std lesser than zero and return EI as zero
        value_near_zero = 0.0000000001
        if(np.any(std < value_near_zero)):
            print("Std less than zero", mean, std)
            output[std<value_near_zero] = 0

        return output

    @staticmethod
    def _poi(x, reg, y_max, xi):
        mean, std = reg.predict(x)
        z = (mean - y_max - xi)/std
        return norm.cdf(z)


def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]


class BColours(object):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


class PrintLog(object):

    def __init__(self, params={}):

        self.ymax = None
        self.xmax = None
        self.params = params
        self.ite = 1

        #self.start_time = datetime.now()
        self.last_round = datetime.now()
        self.module_timer = datetime.now()
        self.module_timer_log = {}

        # sizes of parameters name and all
        self.sizes = [max(len(ps), 7) for ps in params]

        # Sorted indexes to access parameters
        self.sorti = sorted(range(len(self.params)),
                            key=self.params.__getitem__)

    def reset_timer(self):
        #self.start_time = datetime.now()
        self.last_round = datetime.now()
        self.module_timer = datetime.now()

    def log_timer(self, tag):
        self.module_timer_log[tag] = \
            (datetime.now() - self.module_timer)
        self.module_timer = datetime.now()

    def print_header(self, initialization=True):

        if initialization:
            print("{}Initialization{}".format(BColours.RED,
                                              BColours.ENDC))
        else:
            print("{}Bayesian Optimization{}".format(BColours.RED,
                                                     BColours.ENDC))

        print(BColours.BLUE + "-" * (29 + sum([s + 5 for s in self.sizes])) +
            BColours.ENDC)

        print("{0:>{1}}".format("Step", 5), end=" | ")
        print("{0:>{1}}".format("Time", 6), end=" | ")
        print("{0:>{1}}".format("Value", 10), end=" | ")

        for index in self.sorti:
            print("{0:>{1}}".format(self.params[index],
                                    self.sizes[index] + 2),
                  end=" | ")
        print('')

    def print_step(self, x, y, warning=False):

        print("{:>5d}".format(self.ite), end=" | ")

        m, s = divmod((datetime.now() - self.last_round).total_seconds(), 60)
        print("{:>02d}m{:>02d}s".format(int(m), int(s)), end=" | ")

        
        if self.ymax is None or self.ymax < y:
            self.ymax = y
            self.xmax = x
            print("{0}{2: >10.5e}{1}".format(BColours.MAGENTA,
                                             BColours.ENDC,
                                             y),
                  end=" | ")

            for index in self.sorti:
                print("{0}{2: >{3}.{4}f}{1}".format(
                            BColours.GREEN, BColours.ENDC,
                            x[index],
                            self.sizes[index] + 2,
                            min(self.sizes[index] - 3, 6 - 2)
                        ),
                      end=" | ")
        else:
            print("{: >10.5e}".format(y), end=" | ")
            for index in self.sorti:
                print("{0: >{1}.{2}f}".format(x[index],
                                              self.sizes[index] + 2,
                                              min(self.sizes[index] - 3, 6 - 2)),
                      end=" | ")

        if warning:
            print("{}Warning: Test point chose at "
                  "random due to repeated sample.{}".format(BColours.RED,
                                                            BColours.ENDC))

        print()
        # Print module timing info
        for t in self.module_timer_log:
            print(t, self.module_timer_log[t])
        print()

        self.last_round = datetime.now()
        self.ite += 1
        sys.stdout.flush()

    def print_summary(self):
        pass
