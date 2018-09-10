from __future__ import print_function
from __future__ import division

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from bayesian_helpers import UtilityFunction, unique_rows, PrintLog

# import app.noc.link_distribution_16 as ld16
import bayesian_helpers as bh
import afo.stage_rls as afosr
import afo.mcts as afom
import afo.bamlogo as afobl
import afo.struct_pred as afosp

import os
import sys
import pandas as pd
# import utility_functions as U
# import simulator_interface as si
from sklearn.gaussian_process.kernels import RBF
# import link_distribution_16 as ld16
import app.syn_func.feature_generator as sffg
import app.noc.link_distribution_16 as ld16
import app.tsv.feature_generator as tsvfg
import app.bocs.feature_generator as bocsfg


import matplotlib.pyplot as plt
import math


class BayesianOptimization(object):

    def __init__(self, 
        sim, 
        # pbounds, 
        verbose=1):
        # Some function to be optimized
        self.f = sim.run
        self.sim = sim
        # Utility Function placeholder
        self.util = None

        # Verbose
        self.verbose = verbose

    def tsv_init(self, max_num_spare_tsv, 
        max_num_iterations, max_num_levels):
        self.max_num_spare_tsv = max_num_spare_tsv
        self.max_num_iterations = max_num_iterations
        self.max_num_levels = max_num_levels
        return

    def init(self, init_points, output_file_name, sim_type, opt_algo):
        """
        Initialization method to kick start the optimization process. It is a
        combination of points passed by the user, and randomly sampled ones.

        :param init_points:
            Number of random points to probe.
        """
        # Initialization flag
        self.initialized = False

        # Initialization lists --- stores starting points before process begins
        self.init_points = []
        self.x_init = []
        self.y_init = []

        # Numpy array place holders
        self.X = None
        self.Y = None

        # Counter of iterations
        self.i = 0

        # PrintLog object
        # self.plog = PrintLog(self.keys)
        self.plog = PrintLog()
        # Reset timer
        self.plog.reset_timer()
        # Output dictionary
        self.res = {}
        # Output dictionary
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values': [], 'params': []}
        self.res['max_every_25'] = {'values': [], 'params': [], 'iteration':[]}
        if self.verbose:
            self.plog.print_header()

        if(sim_type == "dummy" or sim_type == "actual"):
            # Generate max value for model
            feature_list = pd.read_csv("./app/noc/init_points.txt", delim_whitespace=True, header=None, comment='#').as_matrix()
            feature_list = feature_list[:5,1:]
            # Since we want to store the worst values. Find the min and store it
            # Initialise the first max to be the worst value possible :)    
            # This is done so that the graph starts from a very high value
            y_min = 999999999
            x_min = []
            x_init = []
            y_init = []
            for idx in range(5):
                x = feature_list[idx]
                y_out = self.f(x)
                x_init.append(x)
                y_init.append(y_out)
                if(y_min > y_out):
                    y_min = y_out
                    x_min = x
            self.res['max_every_25']['values'].append(y_min)
            self.res['max_every_25']['params'].append(x_min.astype(int))
            self.res['max_every_25']['iteration'].append(-1)
            if self.verbose:
                self.plog.print_step(x_min, y_min)
            # Turn it into np array and store.
            # Dummy operation for graph plotting
            self.X = np.asarray(x_init)
            self.Y = np.asarray(y_init)
            if output_file_name is not None:
                self.points_to_csv(output_file_name)

        print("Sim type ", sim_type)
        if(sim_type == "actual" or sim_type == "dummy"):
            is_small_world = (opt_algo == "RLS_ORDERED"
                or opt_algo == "RLS_UNORDERED"
                or opt_algo == "STAGE_ORDERED"
                or opt_algo == "STAGE_UNORDERED"
                or opt_algo == "SMAC_ORDERED"
                or opt_algo == "SMAC_UNORDERED")
            is_ordered = (opt_algo == "RLS_ORDERED"
                or opt_algo == "STAGE_ORDERED"
                or opt_algo == "SMAC_ORDERED"
                or opt_algo == "RLS_ORDERED_NSW"
                or opt_algo == "STAGE_ORDERED_NSW"
                or opt_algo == "SMAC_ORDERED_NSW")
            self.fg = ld16.LinkDistribution(
                is_small_world=is_small_world, 
                is_ordered=is_ordered)
        elif(sim_type == "actual_tsv" or sim_type == "dummy_tsv"):
            self.fg = tsvfg.FeatureGenerator(self.max_num_spare_tsv,
                self.max_num_levels)
        elif(sim_type == "bocs" or sim_type == "ising" or sim_type == "contamination"):
            self.fg = bocsfg.FeatureGenerator(self.sim)
        # elif(sim_type != "actual_tsv" and sim_type != "dummy_tsv"):
        else:
            self.fg = sffg.FeatureGenerator(self.sim)

        self.sp = afosp.StructPred(feature_generator=self.fg)
        self.sa = afosr.SearchAlgorithms(feature_generator=self.fg)

        # if(sim_type == "dummy_tsv" or sim_type == "actual_tsv"):
        #     # KJN Init
        #     feature_list = afom.generate_random_design(init_points, self.max_num_spare_tsv)
        # else:

        # Initialise the model with random
        feature_list = self.fg.generate_n_random_feature(init_points)
        # x_max = self.fg.generate_n_random_feature(1)[0]
            
        # Create empty list to store the new values of the function
        y_init = []
        x_init = []
        y_max_list = []
        y_max = -1*999999.0

        for idx in range(init_points):
            x = feature_list[idx]
            y_out = self.f(x)
            # y_out = feature_list_1[idx,0]
            
            # print(y_out)
            y_init.append(y_out)
            x_init.append(x)
            if(y_max < y_out):
                y_max = y_out

            y_max_list.append(y_max)
            if self.verbose:
                self.plog.print_step(x, y_init[-1])
        
        # Turn it into np array and store.
        self.X = np.asarray(x_init)
        self.Y = np.asarray(y_init)
        self.Y_max = np.asarray(y_max_list)

        # # Hack to debug MCTS
        # feature_list = pd.read_csv("init_points_tsv.txt", delim_whitespace=True, header=None, comment='#').as_matrix()
        # self.X = feature_list[:,1:]
        # self.Y = feature_list[:,0]

        # Updates the flag
        self.initialized = True


    def init_gp_params(self, 
        kernel=Matern(nu=2.5),
        n_restarts_optimizer=25,
        acq='ucb',
        kappa=2.576,
        xi=0.0,
        **params):
        
        """
        Parameters
        ----------
        :param acq:
            Acquisition function to be used, defaults to Upper Confidence Bound.

        :param gp_params:
            Parameters to be passed to the Scikit-learn Gaussian Process object

        """
        # Internal GP regressor
        self.reg = bh.ModelRegressor("gp",
            GaussianProcessRegressor(kernel=kernel,
                    n_restarts_optimizer=n_restarts_optimizer)
        )
        # Set acquisition function
        self.util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)
        # Set parameters if any was passed
        self.reg.set_params(**params)

        self.kernel = kernel

    def init_random_forest(self,
        n_estimators=10,
        min_samples_split=2,
        acq='ucb',
        kappa=2.576,
        xi=0.0,
        **params):
    
        # n_estimators = 15
        self.reg = bh.ModelRegressor("rf",
            RandomForestRegressor(n_estimators=n_estimators)
                #min_samples_split=min_samples_split)
        )

        # Set acquisition function
        self.util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)
        # Set parameters if any was passed
        self.reg.set_params(**params)

    def dummy_utility(self, x, reg, y_max):
        num_val = len(x)
        # print("dummy utility len ", num_val)
        val_list = np.zeros(num_val)
        for i in range(num_val):
            val_list[i] = self.f(x[i])

        # print("dummy utility done")
        return val_list

    def _get_next_x_max(self, opt_algo, y_max):
        # print(opt_algo, y_max)
        # if(opt_algo == "DIRECT"):
        #     x_max = bh.acq_max_DIRECT(
        #         ac=self.util.utility_DIRECT,
        #         reg=self.reg,
        #         y_max=y_max)
        # elif(opt_algo == "RANDOM"):
        #     x_max = bh.acq_max_random(
        #         ac=self.util.utility,
        #         reg=self.reg,
        #         y_max=y_max)
        if(opt_algo == "MCTS"):
            x_max, _ = afom.monte_carlo_tree_search(
                ac=self.util.utility,
                reg=self.reg,
                y_max=y_max,
                max_num_iterations = self.max_num_iterations,
                max_num_spare_tsv = self.max_num_spare_tsv)
        if(opt_algo == "RLS_ORDERED" 
        or opt_algo == "RLS_UNORDERED"
        or opt_algo == "RLS_ORDERED_NSW" 
        or opt_algo == "RLS_UNORDERED_NSW"):
            x_max, _ = self.sa.random_local_search(
            ac=self.util.utility,
            reg=self.reg,
            y_max=y_max)
        if(opt_algo == "STAGE_ORDERED" 
        or opt_algo == "STAGE_UNORDERED"
        or opt_algo == "STAGE_ORDERED_NSW" 
        or opt_algo == "STAGE_UNORDERED_NSW"):
            x_max, _ = self.sa.stage_algorithm(
                ac=self.util.utility,
                reg=self.reg,
                y_max=y_max,
                X=self.X)
        if(opt_algo == "SMAC_UNORDERED_NSW" 
        or opt_algo == "SMAC_UNORDERED"):
            x_max, _ = self.sa.smac_srch(
                ac=self.util.utility,
                reg=self.reg,
                y_max=y_max,
                X=self.X)
        if(opt_algo == "BAMLOGO"):
            bl = afobl.BAMLOGO(
                num_features = self.X.shape[1],
                ac=self.util.utility,
                reg=self.reg,
                y_max=y_max)
            costs, bestValues, queryPoints = bl.maximize(budget=10.0)
            x_max = np.array(queryPoints[-1])
            _ = bestValues[-1]
            print("Costs ", costs[-1], bestValues[-1], queryPoints[-1])
        if(opt_algo == "STRUCTPRED"):
            # x_max, _ = sp.maximize_beam(ac=self.util.utility,
            #     reg=self.reg,
            #     y_max=y_max)
            x_max, _ = self.sp.maximize_mcts(ac=self.util.utility,
                reg=self.reg,
                y_max=y_max)
            

        # print(x_max)
        return x_max, _

    def maximize(self, init_points, n_iter, 
        opt_algo, dump_max_every_n,
        output_file_name, sim_type):

        # Initialize x, y and find current y_max
        if not self.initialized:
            self.init(init_points, 
                output_file_name,
                sim_type, opt_algo)

        y_max = self.Y.max()

        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        # print(self.X.shape)        
        self.reg.fit(self.X[ur], self.Y[ur])

        # Finding argmax of the acquisition function.
        x_max, _ = self._get_next_x_max(opt_algo, y_max)
        
        # Print new header
        if self.verbose:
            self.plog.print_header(initialization=False)
        # Iterative process of searching for the maximum. At each round the
        # most recent x and y values probed are added to the X and Y arrays
        # used to train the Gaussian Process. Next the maximum known value
        # of the target function is found and passed to the acq_max function.
        # The arg_max of the acquisition function is found and this will be
        # the next probed value of the target function in the next round.
        for i in range(n_iter):
            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            pwarning = False
            # print(x_max.shape, self.X.shape)
            if np.any((np.absolute(self.X - x_max)).sum(axis=1) == 0):
                print("x_max is already used. generating random x_max")
                print((np.absolute(self.X - x_max)).sum(axis=1))
                # print((np.absolute(self.X[0]-x_max)).sum())
                # if(sim_type == "dummy_tsv" or sim_type == "actual_tsv"):
                #     # KJN
                #     x_max = afom.generate_random_design(1, self.max_num_spare_tsv)[0]
                # else:
                x_max = self.fg.generate_n_random_feature(1)[0]

                pwarning = True

            self.plog.log_timer("Misc")
            # Append most recently generated values to X and Y arrays
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))
            #self.Y = np.append(self.Y, self.f(**dict(zip(self.keys, x_max))))
            self.Y = np.append(self.Y, self.f(x_max))

            self.plog.log_timer("Fn Opt")

            # Updating the GP.
            ur = unique_rows(self.X)
            self.reg.fit(self.X[ur], self.Y[ur])

            # Update maximum value to search for next probe point.
            if self.Y[-1] > y_max:
                y_max = self.Y[-1]
            self.Y_max = np.append(self.Y_max, y_max)

            # Print stuff
            if self.verbose:
                self.plog.print_step(self.X[-1], self.Y[-1], warning=pwarning)

            # Keep track of total number of iterations
            self.i += 1

            self.res['max'] = {'max_val': self.Y.max(),
                               'max_params': self.X[self.Y.argmax()].astype(int)}
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(self.X[-1])
            if(i%dump_max_every_n == 0):
                self.res['max_every_25']['values'].append(self.Y.max())
                self.res['max_every_25']['params'].append(self.X[self.Y.argmax()].astype(int))
                self.res['max_every_25']['iteration'].append(i)
                if output_file_name is not None:
                    self.points_to_csv(output_file_name)


            self.plog.log_timer("GP Fit")
            self.plot_graph(y_max)

            # Maximize acquisition function to find next probing point
            x_max, _ = self._get_next_x_max(opt_algo, y_max)

            self.plog.log_timer("Acq Max")

        # Print a final report if verbose active.
        if self.verbose:
            self.plog.print_summary()

    def find_average(self, init_points, n_iter, 
        opt_algo, dump_max_every_n,
        output_file_name, sim_type):
        Y_list = []
        for i in range(5):
            self.initialized = False
            self.maximize(init_points, n_iter, 
                opt_algo, dump_max_every_n,
                output_file_name, sim_type)
            Y_list.append(np.array(self.Y_max))
        Y_list = np.array(Y_list)
        Y_avg = np.average(Y_list, axis=0)
        print(Y_avg)
        return

    def points_to_csv(self, file_name):
        """
        After training all points for which we know target variable
        (both from initialization and optimization) are saved

        :param file_name: name of the file where points will be saved in the csv
            format

        :return: None
        """
        # Generate format string for dumping output
        format_string = ""
        format_string += "%3.4e "
        for i in range(self.X.shape[1]):
            format_string += "%i "

        # Create two files one for max and another for each iteration
        base, extension = os.path.splitext(file_name)
        file_name_max = base+"_max"+extension
        file_name_all = base+"_all"+extension

        # Save all the points
        points = np.hstack((np.expand_dims(self.Y, axis=1), self.X.astype(int)))
        # print(points)
        # header = ', '.join(self.keys + ['target']) # header=header,
        np.savetxt(file_name_all, points,  fmt=format_string,delimiter=',')
        print("Completed all value dump")

        # Save the max value calculated
        # max_points = np.hstack((self.res['max']['max_val'], self.res['max']['max_params']))
        # # print(max_points, max_points.shape, self.X.shape[1])
        # np.savetxt(file_name_max, np.atleast_2d(max_points),  fmt=format_string, delimiter=',')
        # print("Completed max value dump")

        # Save all the points
        max_points = np.hstack((np.expand_dims(self.res['max_every_25']['iteration'], axis=1), \
            np.expand_dims(self.res['max_every_25']['values'], axis=1), \
            self.res['max_every_25']['params']))
        # print(max_points, self.res['max_every_25']['values'])
        np.savetxt(file_name_max, np.atleast_2d(max_points),  fmt="%i "+format_string,delimiter=',')
        print("Completed max value dump")

    def plot_graph(self, y_max):

        ucb = UtilityFunction(kind='ucb', kappa=3, xi=0.0)
        ei = UtilityFunction(kind='ei', kappa=3, xi=0.0)
        poi = UtilityFunction(kind='poi', kappa=3, xi=0.0)
        if(True): #num_features == 1):
            # N = 10000
            x_vec, x_range = self.fg.get_all_combinations(1)
            # x_range = range(x_vec.shape[0])
            X = self.X

            X_Val = X.dot(2**np.arange(X.shape[1]-1,-1,-1))
            X_Val = np.expand_dims(X_Val, axis=1)
            y_value = self.sim.run_list(x_vec)

            # for i in range(16):
            #     x_pred = x_vec[::2**i]
            #     y_pred = y_value[::2**i]
            #     print("Downsampled ", x_pred.shape, y_pred.shape)
            #     self.reg.fit(x_pred, y_pred)

            # x_pred = x_vec[::]
            # y_pred = y_value[::]
            # print("Downsampled ", x_pred.shape, y_pred.shape)
            self.reg.fit(x_pred, y_pred)

            mean, std = self.reg.predict(x_vec)
            ucb_val = ucb.utility(x_vec, self.reg, y_max)
            ei_val = ei.utility(x_vec, self.reg, y_max)
            poi_val = poi.utility(x_vec, self.reg, y_max)

            plt.figure(1)
            plt.subplot(211)
            plt.plot(x_range, y_value, 'r--', 
                     x_range, mean, 'b--',
                     x_range, mean+std, 'g--',
                     x_range, mean-std, 'g--',
                     X_Val, self.Y, 'bo')#, t, t**3, 'g^')
            plt.subplot(212)
            plt.plot(x_range, ucb_val, 'r--', 
                     x_range, ei_val, 'b--',
                     x_range, poi_val, 'g--')
            plt.show()
        elif(num_features == 2):
            N = 100
            x_range = np.arange(N)*1.0/N
            x_vec = np.array(np.meshgrid(x_range, x_range)).T.reshape(-1,2)
            X = self.X
            if(is_discrete):
                x_vec = x_vec*num_parts
                X = X*1.0/num_parts
            y_value = self.sim.run_list(x_vec)
            y_value[y_value<-5] = -5
            # print(x_vec.shape)
            mean, std = self.reg.predict(x_vec)
            ucb_val = ucb.utility(x_vec, self.reg, y_max)
            ei_val = ei.utility(x_vec, self.reg, y_max)
            poi_val = poi.utility(x_vec, self.reg, y_max)

            plt.subplot(221)
            plt.imshow(y_value.reshape(100,100), 
                origin='lower',extent=[0, 100, 0, 100])
            plt.colorbar()
            plt.scatter(X[:,0], X[:,1])
            plt.subplot(222)
            plt.imshow(mean.reshape(100,100), 
                origin='lower',extent=[0, 100, 0, 100])
            plt.colorbar()
            plt.subplot(223)
            plt.imshow(std.reshape(100,100), 
                origin='lower',extent=[0, 100, 0, 100])
            plt.colorbar()
            plt.subplot(224)
            plt.imshow((mean+std).reshape(100,100), 
                origin='lower',extent=[0, 100, 0, 100])
            plt.colorbar()

            # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
            # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
            # plt.colorbar(cax=cax)
            plt.show()
            plt.subplot(221)
            plt.imshow(y_value.reshape(100,100), 
                origin='lower',extent=[0, 100, 0, 100])
            plt.colorbar()
            plt.scatter(self.X[:,0], self.X[:,1])
            plt.subplot(222)
            plt.imshow(ucb_val.reshape(100,100), 
                origin='lower',extent=[0, 100, 0, 100])
            plt.colorbar()
            plt.subplot(223)
            plt.imshow(ei_val.reshape(100,100), 
                origin='lower',extent=[0, 100, 0, 100])
            plt.colorbar()
            plt.subplot(224)
            plt.imshow(poi_val.reshape(100,100), 
                origin='lower',extent=[0, 100, 0, 100])
            plt.colorbar()

            # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
            # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
            # plt.colorbar(cax=cax)
            plt.show()
            
        else:
            print("Number of features more than 2 ", num_features)

# Debug:
        # if(self.kernel):
        #     self.kernel.fit(self.X[ur], self.Y[ur])
        # print(self.reg)

        # print(self.X[ur].shape)

        # To be commented out later. 
        # Debugging the model
        # for i in range(self.X[ur].shape[0]):
        #     # print(self.X[i])
        #     mean_x, std_x = self.reg.predict(\
        #         ld16.LinkDistribution().generate_n_random_feature(1)[0].reshape(1, -1))
        #     print("First Pred validation", i, mean_x, std_x, self.Y[i])

        # for i in range(10):
        #     X = self.X[:10+2*i]
        #     Y = 100+self.Y[:10+2*i]
        #     y_max = Y.max()
        #     # Find unique rows of X to avoid GP from breaking
        #     ur = unique_rows(X)
        #     # print(self.X.shape)
        #     self.reg.fit(X, Y)
        #     print("Num unique rows ",len(ur), Y, y_max)

        #     _, x_max_stg = self._get_next_x_max("STAGE_ORDERED", y_max)
        #     _, x_max_rls = self._get_next_x_max("RLS_ORDERED", y_max)

        #     print(i, x_max_rls, x_max_stg)

        # return
