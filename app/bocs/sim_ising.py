'''
Python implementation of Ising Model domain
from Bayesisan Optimization of Combinatorial structures paper
Authors: Ricardo Baptista & Matthias Poloczek
Original Matlab code at https://github.com/baptistar/BOCS/tree/master/test_problems/IsingModel
'''
import numpy as np
#import matplotlib.pyplot as plt
from . import feature_generator as fg

class Ising_Model(object):
	def __init__(self, num_features, n_vars, lam):
		# input parameters
		self.num_features = num_features # no. of edges
		self.n_vars = n_vars # no. of graph nodes 
		self.Q = self.rand_ising(n_vars) # grid graph representation of symmetric interaction matrix
		self.ising_moments = self.ising_mom() # moments required in KL divergence computation
		self.lam = lam # regularization constant
	def get_config(self):
		return (self.num_features, self.n_vars, self.lam)

	def rand_ising(self,nodes):
		# Total number of edges = 2*n*(n-1)
	    Q = np.zeros((nodes, nodes))
	    n_side = np.sqrt(nodes).astype(np.int32)
	    # all the right edges
	    for i in range(n_side):
	    	for j in range(n_side-1):
	            node = i*n_side + j
			    # assign edges weights randomly as in the paper
	            par = 4.95*np.random.rand() + 0.05
	            if (np.random.rand() > 0.5):
	                par = par*-1
	            Q[node,node+1] = par
	            Q[node+1,node] = Q[node,node+1]
	    # all the down edges
	    for i in range(n_side-1):
	    	for j in range(n_side):
	            node = i*n_side + j
	            par = 4.95*np.random.rand() + 0.05
	            if (np.random.rand() > 0.5):
	                par = par*-1         
	            Q[node,node+n_side] = par 
	            Q[node+n_side,node] = Q[node,node+n_side]
	    return Q

	def ising_mom(self):
		nodes = self.Q.shape[0]
		bin_vals = np.zeros((2**nodes, nodes))
		for i in range(2**nodes):
		        bin_vals[i] = np.array(list(np.binary_repr(i).zfill(nodes))).astype(np.int8)
		bin_vals[bin_vals == 0] = -1
		n_vectors = bin_vals.shape[0]

		pdf_vals = np.zeros(n_vectors)
		for i in range(n_vectors):
		        pdf_vals[i] = np.exp(np.dot(bin_vals[i, :], self.Q).dot(bin_vals[i, :].T))
		#pdf_vals = np.exp(np.dot(bin_vals, Q).dot(bin_vals.T))
		norm_const = np.sum(pdf_vals)
		ising_moments = np.zeros((nodes, nodes))
		# Second moment for each pair of values
		for i in range(nodes):
		       for j in range(nodes):
		            bin_pair = bin_vals[:, i]*bin_vals[:, j]
		            ising_moments[i][j] = np.sum(bin_pair*pdf_vals)/norm_const
		return ising_moments

	def run(self, x):
	    Theta_P = self.Q
	    if (x.ndim == 2):
        	x = x.reshape(x.shape[1])
	    nodes = self.Q.shape[0]
	    bin_vals = np.zeros((2**self.n_vars, self.n_vars))
	    for i in range(2**self.n_vars):
	            bin_vals[i] = np.array(list(np.binary_repr(i).zfill(self.n_vars))).astype(np.int8)
	    bin_vals[bin_vals == 0] = -1
	    n_vectors = bin_vals.shape[0]
	    #P_vals = np.exp(np.dot(bin_vals, Theta_P).dot(bin_vals.T))
	    P_vals = np.zeros(n_vectors)
	    for i in range(n_vectors):
	            P_vals[i] = np.exp(np.dot(bin_vals[i, :], Theta_P).dot(bin_vals[i, :].T))
	    Zp = np.sum(P_vals)  # partition function of random variable 
	    Theta_Q = np.tril(Theta_P, -1)
	    nnz_Q = Theta_Q!=0
	    Theta_Q[nnz_Q] = Theta_Q[nnz_Q]*x.T
	    Theta_Q = Theta_Q + Theta_Q.T
	    #Q_vals = np.exp(np.dot(bin_vals, Theta_Q).dot(bin_vals.T))
	    Q_vals = np.zeros(n_vectors)
	    for i in range(n_vectors):
	        Q_vals[i] = np.exp(np.dot(bin_vals[i, :], Theta_Q).dot(bin_vals[i, :].T))
	    Zq = np.sum(Q_vals) # partition function of random variable Q
	    KL = np.sum(np.sum((Theta_P - Theta_Q)*self.ising_moments)) + np.log(Zq) - np.log(Zp)
	    if (KL < 0):
	        print("KL less than zero!")
	        print(x)
	        print(np.sum(Q<0)/2)
	        print(np.sum(np.sum((Theta_P - Theta_Q))))
	        print(np.log(Zq))
	        print(np.log(Zp))
	    return -1*(KL+self.lam*np.sum(x))


	def run_list(self, X):
		n = X.shape
		out = np.zeros(n)
		for i in range(X.shape[0]):
			out[i] = self.run(X[i])
		return out
	def compute_max(self):
		# brute force function to check the working of code
		fgobj = fg.FeatureGenerator(self.num_features)
		max_value = -1 * np.inf
		max_num_vec = []
		cost = np.zeros(2**self.num_features)
		for num in range(2**self.num_features):
			num_vec = np.array([fgobj.get_binary(num, self.num_features)])
			value = self.run(num_vec)
			cost[num] = value
			if(value > max_value):
				max_value = value
				max_num_vec = num_vec
		return max_value, max_num_vec, cost			

if __name__=="__main__":
	sim = Ising_Model(12, 9, 1e-4)
	x = fg.FeatureGenerator(sim).generate_n_random_feature(1)
	print(sim.run(x))
