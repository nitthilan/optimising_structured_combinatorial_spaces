from __future__ import division
import numpy as np
import sys
from sklearn.gaussian_process.kernels import Kernel as ParentKernel, StationaryKernelMixin, NormalizedKernelMixin
from scipy.sparse import lil_matrix
import copy
import constants as C
import utility_functions as U
import link_distribution_16 as ld16
from datetime import datetime


def adjList(design, listOfCorePairs):
    label = []
    ad_list = []
    for i in range(C.NUM_CORES):
        temp_list = []
        ad_list.append(temp_list)
        label.append(i)
    
    for i in range(len(design)):
        if design[i] == 1:
            x = int(listOfCorePairs[i][0])
            y = int(listOfCorePairs[i][1])
            ad_list[x].append(y)
            ad_list[y].append(x)

    vertical_link_list = U.generate_vertical_links_list()
    for i in range(len(vertical_link_list)):
        x = vertical_link_list[i][0]
        y = vertical_link_list[i][1]
        ad_list[x].append(y)
        ad_list[y].append(x)

    for i in range(C.NUM_CORES):
        ad_list[i].sort()
        #print(i, ad_list[i])
    return label, ad_list

def WL_compute(ad_list, node_label, h, num_learn):
    # Total number of graphs in the dataset
    n = len(ad_list)

    # Total number of nodes in dataset: initialized as zero
    tot_nodes = 0

    # list of kernel matrices
    K = [0] * (h + 1)
    # list of feature mtrices
    phi_list = [0] * (h + 1)

    # total number of nodes in the dataset
    for i in range(n):
        tot_nodes = tot_nodes + int(len(ad_list[i]))

    # print("Actual Num nodes ", tot_nodes)

    # tot_nodes = 64*(h+1)
    # print("Thresholding it to num iterations ", tot_nodes)

    # each column of phi will be the explicit feature representation for the graph j
    phi = lil_matrix((tot_nodes, n), dtype=np.uint32)

    # labels will be used to store the new labels
    labels = [0] * n

    # label lookup is a dictionary which will contain the mapping
    # from multiset labels (strings) to short labels (integers)
    label_lookup = {}

    # counter to create possibly new labels in the update step
    label_counter = 0

    # Note: here we are just renaming the node labels from 0,..,num_labels
    # for each graph
    for i in range(n):

        # copy the original labels
        l_aux = np.copy(node_label[i])

        # will be used to store the new labels
        labels[i] = np.zeros(len(l_aux), dtype=np.int32)

        # for each label in graph
        for j in range(len(l_aux)):
            l_aux_str = str(l_aux[j])

            # If the string do not already exist
            # then create a new short label
            if not label_lookup.has_key(l_aux_str):
                label_lookup[l_aux_str] = label_counter
                labels[i][j] = label_counter
                label_counter += 1
            else:
                labels[i][j] = label_lookup[l_aux_str]

            # node histograph of the new labels
            phi[labels[i][j], i] += 1

    L = label_counter


    #####################
    # --- Main code --- #
    #####################

    # Now we are starting with the first iteration of WL

    # features obtained from the original node (renamed) labels
    phi_list[0] = phi

    # print("Phi ", phi.shape, phi.toarray())

    if(num_learn):
        phi_sparse_a = phi[:,:num_learn]
        phi_sparse_b = phi[:,num_learn:]
        K[0] = phi_sparse_a.transpose().dot(phi_sparse_b).astype(np.float32)
        # print("Understanding dimension ", n, phi_sparse_a.shape, phi_sparse_b.shape, K[0].shape)
        # print("K[0]", K[0])

        # phi_sparse_a_temp = phi_sparse_a.toarray().T
        # phi_sparse_b_temp = phi_sparse_b.toarray().T
        # print("phi matrices start", phi_sparse_a_temp.shape, phi_sparse_b_temp.shape, 
        #     (np.absolute(phi_sparse_a_temp - phi_sparse_b_temp)).sum(axis=1))

    else:
        # Kernel matrix based on original features
        K[0] = phi.transpose().dot(phi).astype(np.float32)



    # Initialize iterations to 0
    it = 0

    # copy of the original labels: will stored the new labels
    new_labels = np.copy(labels)

    # until the number of iterations is less than h
    while it < h:

        # Initialize dictionary and counter
        # (same meaning as before)
        label_lookup = {}
        label_counter = 0

        # Initialize phi as a sparse matrix
        phi = lil_matrix((tot_nodes, n), dtype=np.int32)
        # convert it to array
        phi = phi.toarray()

        start_timer = datetime.now()


        # for each graph in the dataset
        for i in range(n):

            # will store the multilabel string
            l_aux_long = np.copy(labels[i])

            # for each node in graph
            for v in range(len(ad_list[i])):

                # the new labels convert to tuple
                new_node_label = tuple([l_aux_long[v]])

                # form a multiset label of the node neighbors
                new_ad = np.zeros(len(ad_list[i][v]), dtype=int)
                for j in range(len(ad_list[i][v])):
                    new_ad[j] = ad_list[i][v][j]

                ad_aux = tuple([l_aux_long[j] for j in new_ad])

                # long labels: original node plus sorted neughbors
                long_label = tuple(tuple(new_node_label) + tuple(sorted(ad_aux)))

                # if the multiset label has not yet occurred , add
                # it to the lookup table and assign a number to it
                if not label_lookup.has_key(long_label):
                    label_lookup[long_label] = str(label_counter)
                    new_labels[i][v] = str(label_counter)
                    label_counter += 1

                # else assign it the already existing number
                else:
                    new_labels[i][v] = label_lookup[long_label]

            # count the node label frequencies
            aux = np.bincount(new_labels[i])
            # print("Aux ")
            phi[new_labels[i], i] += aux[new_labels[i]]

        L = label_counter

        # create phi for iteration it+1
        phi_sparse = lil_matrix(phi)
        phi_list[it + 1] = phi_sparse
 
        #print("feature cal ", datetime.now() - start_timer)
        start_timer = datetime.now()

        if(num_learn):
            phi_sparse_a = phi_sparse[:,:num_learn]
            phi_sparse_b = phi_sparse[:,num_learn:]
            # print("Understanding dimension ", n, phi_sparse_a.shape, 
            #     phi_sparse_b.shape, K[it].shape)
            # phi_sparse_a = phi_sparse_a ** 2
            # norm_phi_sparse_a = np.sum(phi_sparse_a.T**2,axis=-1)**(1./2)
            # norm_phi_sparse_b = np.sum(phi_sparse_b.T**2,axis=-1)**(1./2)
            # print(norm_phi_sparse_a[:, np.newaxis], norm_phi_sparse_b[:, np.newaxis])
            # norm_ab = np.dot(norm_phi_sparse_a[:, np.newaxis], norm_phi_sparse_b[:, np.newaxis].T)
            
            # norm_phi_sparse_a = np.linalg.norm(phi_sparse_a, axis=1, keepdims=False)
            # norm_phi_sparse_b = np.linalg.norm(phi_sparse_b, axis=1, keepdims=False)
            # norm_phi_sparse_a = np.sqrt(np.sum(np.multiply(phi_sparse_a, phi_sparse_a), axis=0))
            # norm_phi_sparse_b = np.sqrt(np.sum(np.multiply(phi_sparse_b, phi_sparse_b), axis=0))

            # create K at iteration it+1
            cross_correlation = phi_sparse_a.transpose().dot(phi_sparse_b).astype(np.float32)
            K[it + 1] = K[it] + cross_correlation
            # print("Corss correlation ", cross_correlation.toarray())
            # norm_ab = norm_phi_sparse_a*norm_phi_sparse_b
            # print(norm_ab.shape)

            # phi_sparse_a_temp = phi_sparse_a.toarray().T
            # phi_sparse_b_temp = phi_sparse_b.toarray().T
            # print("phi matrices ", it, phi_sparse_a_temp.shape, phi_sparse_b_temp.shape, 
            #     (np.absolute(phi_sparse_a_temp - phi_sparse_b_temp)).sum(axis=1))
            # print(phi_sparse_a_temp[:,64:128], phi_sparse_b_temp[:,64:128])

        else:
            # create K at iteration it+1
            K[it + 1] = K[it] + phi_sparse.transpose().dot(phi_sparse).astype(np.float32)

        #print("Mat mul ", datetime.now() - start_timer)
        

        # Initialize labels for the next iteration as the new just computed
        labels = copy.deepcopy(new_labels)

        # increment the iteration
        it = it + 1

    normalised_K = K[it].toarray()/(h*64)
    # print(" total num nodes ", label_counter, normalised_K)
    # return the "h iteration" of Kernel Matrix
    # if(np.any(normalised_K>=1) or np.any(normalised_K<=0.0001)):
    #     print("K value greater ", K[it])
    return normalised_K.T, phi_list


def WL_compute_new(ad_list, node_label, h, num_learn):
    # Total number of graphs in the dataset
    n = len(ad_list)

    # Total number of nodes in dataset: initialized as zero
    tot_nodes = 0

    # list of kernel matrices
    K = [0] * (h + 1)
    # list of feature mtrices
    phi_list = [0] * (h + 1)

    # total number of nodes in the dataset
    for i in range(n):
        tot_nodes = tot_nodes + int(len(ad_list[i]))

    # print("Actual Num nodes ", tot_nodes)

    # tot_nodes = 64*(h+1)
    # print("Thresholding it to num iterations ", tot_nodes)

    # each column of phi will be the explicit feature representation for the graph j
    phi = lil_matrix((tot_nodes, n), dtype=np.uint32)

    # labels will be used to store the new labels
    labels = [0] * n

    # label lookup is a dictionary which will contain the mapping
    # from multiset labels (strings) to short labels (integers)
    label_lookup = {}

    # counter to create possibly new labels in the update step
    label_counter = 0

    # Note: here we are just renaming the node labels from 0,..,num_labels
    # for each graph
    for i in range(n):

        # copy the original labels
        l_aux = np.copy(node_label[i])

        # will be used to store the new labels
        labels[i] = np.zeros(len(l_aux), dtype=np.int32)

        # for each label in graph
        for j in range(len(l_aux)):
            l_aux_str = str(l_aux[j])

            # If the string do not already exist
            # then create a new short label
            if not label_lookup.has_key(l_aux_str):
                label_lookup[l_aux_str] = label_counter
                labels[i][j] = label_counter
                label_counter += 1
            else:
                labels[i][j] = label_lookup[l_aux_str]

            # node histograph of the new labels
            phi[labels[i][j], i] += 1

    L = label_counter


    #####################
    # --- Main code --- #
    #####################

    # Now we are starting with the first iteration of WL

    # features obtained from the original node (renamed) labels
    phi_list[0] = phi

    # print("Phi ", phi.shape, phi.toarray())

    if(num_learn):
        phi_sparse_a = phi[:,:num_learn]
        phi_sparse_b = phi[:,num_learn:]
        K[0] = phi_sparse_a.transpose().dot(phi_sparse_b).astype(np.float32)
        # print("Understanding dimension ", n, phi_sparse_a.shape, phi_sparse_b.shape, K[0].shape)
        # print("K[0]", K[0])

        # phi_sparse_a_temp = phi_sparse_a.toarray().T
        # phi_sparse_b_temp = phi_sparse_b.toarray().T
        # print("phi matrices start", phi_sparse_a_temp.shape, phi_sparse_b_temp.shape, 
        #     (np.absolute(phi_sparse_a_temp - phi_sparse_b_temp)).sum(axis=1))

    else:
        # Kernel matrix based on original features
        K[0] = phi.transpose().dot(phi).astype(np.float32)



    # Initialize iterations to 0
    it = 0

    # copy of the original labels: will stored the new labels
    new_labels = np.copy(labels)

    # until the number of iterations is less than h
    while it < h:

        # Initialize dictionary and counter
        # (same meaning as before)
        label_lookup = {}
        label_counter = 0

        # Initialize phi as a sparse matrix
        phi = lil_matrix((tot_nodes, n), dtype=np.int32)
        # convert it to array
        phi = phi.toarray()

        start_timer = datetime.now()


        # for each graph in the dataset
        for i in range(n):

            # will store the multilabel string
            l_aux_long = np.copy(labels[i])

            # for each node in graph
            for v in range(len(ad_list[i])):

                # the new labels convert to tuple
                new_node_label = tuple([l_aux_long[v]])

                # form a multiset label of the node neighbors
                # new_ad = np.zeros(len(ad_list[i][v]), dtype=int)
                # for j in range(len(ad_list[i][v])):
                #     new_ad[j] = ad_list[i][v][j]

                ad_aux = tuple([l_aux_long[j] for j in ad_list[i][v]])

                # long labels: original node plus sorted neughbors
                long_label = tuple(tuple(new_node_label) + tuple(sorted(ad_aux)))

                # if the multiset label has not yet occurred , add
                # it to the lookup table and assign a number to it
                if not label_lookup.has_key(long_label):
                    label_lookup[long_label] = str(label_counter)
                    new_labels[i][v] = str(label_counter)
                    label_counter += 1

                # else assign it the already existing number
                else:
                    new_labels[i][v] = label_lookup[long_label]

            # count the node label frequencies
            aux = np.bincount(new_labels[i])
            # print("Aux ")
            phi[new_labels[i], i] += aux[new_labels[i]]

        L = label_counter

        # create phi for iteration it+1
        phi_sparse = lil_matrix(phi)
        phi_list[it + 1] = phi_sparse
 
        print("feature cal new ", datetime.now() - start_timer)
        start_timer = datetime.now()

        if(num_learn):
            phi_sparse_a = phi_sparse[:,:num_learn]
            phi_sparse_b = phi_sparse[:,num_learn:]
            # print("Understanding dimension ", n, phi_sparse_a.shape, 
            #     phi_sparse_b.shape, K[it].shape)
            # phi_sparse_a = phi_sparse_a ** 2
            # norm_phi_sparse_a = np.sum(phi_sparse_a.T**2,axis=-1)**(1./2)
            # norm_phi_sparse_b = np.sum(phi_sparse_b.T**2,axis=-1)**(1./2)
            # print(norm_phi_sparse_a[:, np.newaxis], norm_phi_sparse_b[:, np.newaxis])
            # norm_ab = np.dot(norm_phi_sparse_a[:, np.newaxis], norm_phi_sparse_b[:, np.newaxis].T)
            
            # norm_phi_sparse_a = np.linalg.norm(phi_sparse_a, axis=1, keepdims=False)
            # norm_phi_sparse_b = np.linalg.norm(phi_sparse_b, axis=1, keepdims=False)
            # norm_phi_sparse_a = np.sqrt(np.sum(np.multiply(phi_sparse_a, phi_sparse_a), axis=0))
            # norm_phi_sparse_b = np.sqrt(np.sum(np.multiply(phi_sparse_b, phi_sparse_b), axis=0))

            # create K at iteration it+1
            cross_correlation = phi_sparse_a.transpose().dot(phi_sparse_b).astype(np.float32)
            K[it + 1] = K[it] + cross_correlation
            # print("Corss correlation ", cross_correlation.toarray())
            # norm_ab = norm_phi_sparse_a*norm_phi_sparse_b
            # print(norm_ab.shape)

            # phi_sparse_a_temp = phi_sparse_a.toarray().T
            # phi_sparse_b_temp = phi_sparse_b.toarray().T
            # print("phi matrices ", it, phi_sparse_a_temp.shape, phi_sparse_b_temp.shape, 
            #     (np.absolute(phi_sparse_a_temp - phi_sparse_b_temp)).sum(axis=1))
            # print(phi_sparse_a_temp[:,64:128], phi_sparse_b_temp[:,64:128])

        else:
            # create K at iteration it+1
            K[it + 1] = K[it] + phi_sparse.transpose().dot(phi_sparse).astype(np.float32)

        print("Mat mul new", datetime.now() - start_timer)
        

        # Initialize labels for the next iteration as the new just computed
        labels = copy.deepcopy(new_labels)

        # increment the iteration
        it = it + 1

    normalised_K = K[it].toarray()/(h*64)
    # print(" total num nodes ", label_counter, normalised_K)
    # return the "h iteration" of Kernel Matrix
    # if(np.any(normalised_K>=1) or np.any(normalised_K<=0.0001)):
    #     print("K value greater ", K[it])
    return normalised_K.T, phi_list

class GraphKernel(StationaryKernelMixin, NormalizedKernelMixin, ParentKernel):


    def __init__(self, h):
        self.listOfCorePairs = ld16.generate_core_connection_options();
        self.h = h
        # self.h = 5 # Hack to debug


    def __call__(self, X, Y=None, eval_gradient=False):

        ad_list = []
        node_label = []
        X = np.atleast_2d(X)
        #hyperparameter = 3

        if Y is None:
            # print("Dimension ", len(X), X.shape)
            start_timer = datetime.now()
            
            for i in range(len(X)):
                x, y = adjList(X[i], self.listOfCorePairs)
                node_label.append(x)
                ad_list.append(y)

            K, phi = WL_compute(ad_list, node_label, 5, 0)
            K_gradient = np.empty((X.shape[0], X.shape[0], 0))

        else:

            start_timer = datetime.now()
            # print("Dimension ", len(X), X.shape, len(Y), Y.shape)
            for i in range(len(Y)):
                label, a_list = adjList(Y[i], self.listOfCorePairs)
                ad_list.append(a_list)
                node_label.append(label)
            
            # print("Y Y Adj Calc ", len(Y), datetime.now() - start_timer)
            start_timer = datetime.now()

            # 5041/4000 - 145
            # 4863/1000 - 71
            # 5480/500 - 73
            # 5224/100 - 67
            # 5062/200 - 60
            # The optimum value seems to be 200 when len(y) is 5 (init points)
            # the ratio with respect to Y len is approx 40. 
            # Probably reducing it to 35 to not drastically scale with len Y value
            # So num_inner = 200 - 40*5
            # Reason is as len(Y) increases the effective matrix size increases so the 
            # X

            # Looks like the iteration inside the WL kernal takes a lot of time
            # Find a better way to calculate the feature vector
            # Probably a hash map using the vector

            num_x = len(X)
            scale_wrt_leny = 35
            num_inner = num_x # 200 # scale_wrt_leny*len(Y) # This value would be 35*5 roughly 175
            num_outer = int((num_x+num_inner-1)/num_inner)

            K = np.zeros((num_x, len(Y)))
            for outer in range(num_outer):
                ad_list_cpy = list(ad_list) # Making another copy
                node_label_cpy = list(node_label)
                start_inner = outer*num_inner
                end_inner = (outer+1)*num_inner
                if(end_inner > num_x):
                    end_inner = num_x
                
                for inner in range(start_inner, end_inner):
                    label, a_list = adjList(X[inner], self.listOfCorePairs)
                    ad_list_cpy.append(a_list)
                    node_label_cpy.append(label)

                K_inner, phi = WL_compute(ad_list_cpy, 
                    node_label_cpy, self.h, len(ad_list))
                # WL_compute_new(ad_list_cpy, 
                #     node_label_cpy, self.h, len(ad_list))
                K[start_inner:end_inner, :] = K_inner
                # print("K Shape", K_inner, K.shape, K_inner.shape,  len(ad_list), len(ad_list_cpy))
                # print(Y, X)
                # print("K ", (np.absolute(Y - X)).sum(axis=1))

            print("Y WL Compute ", K.shape, K_inner.shape, len(X), datetime.now() - start_timer)

        if eval_gradient:
            return K, K_gradient
        else:
            return K
