'''
Python implementation of Ising Model domain
from Bayesisan Optimization of Combinatorial structures paper
Authors: Ricardo Baptista & Matthias Poloczek
Original Matlab code at https://github.com/baptistar/BOCS/tree/master/test_problems/IsingModel
'''
import numpy as np
import matplotlib.pyplot as plt
import feature_generator as fg

class Ising_Model(object):
	def __init__(self):
		# input parameters
		self.n_vars = 9 # no. of graph nodes 
		self.num_features = 12 # no. of edges
		self.Q = self.rand_ising_grid() # grid graph representation of symmetric interaction matrix
		self.ising_moments = self.ising_mom() # moments required in KL divergence computation
		self.lam = 1e-4 # regularization constant
	def get_config(self):
		return (self.num_features, self.n_vars, self.lam)

	def rand_ising_grid(self):
	    Q = np.zeros((self.n_vars, self.n_vars))
	    n_side = np.sqrt(self.n_vars).astype(np.int32)
	    np.random.seed(42)
	    for i in range(n_side):
	        for j in range(n_side-1):
                    node = i*n_side + j
		    # assign edges weights randomly as in the paper
                    Q[node,node+1] = 4.95*np.random.rand() + 0.05;#0.95*rand() + 0.05
                    Q[node+1,node] = Q[node,node+1]
	    for i in range(n_side-1):
                for j in range(n_side):
                    node = i*n_side + j
                    Q[node,node+n_side] = 4.95*np.random.rand() + 0.05;#0.95*rand() + 0.05
                    Q[node+n_side,node] = Q[node,node+n_side]
	    # assign sign of each edge parameter positive or negative with probability half
	    rand_sign = np.tril((np.random.rand(self.n_vars, self.n_vars) > 0.5)*2-1, -1)
	    rand_sign = rand_sign + rand_sign.T
	    Q = rand_sign * Q
	    return Q

	def ising_mom(self):
	    ising_moments = np.zeros((self.n_vars, self.n_vars))
	    bin_vals = np.zeros((2**self.n_vars, self.n_vars))
	    for i in range(2**self.n_vars):
                bin_vals[i] = np.array(list(np.binary_repr(i).zfill(self.n_vars))).astype(np.int8)
	    bin_vals[bin_vals == 0] = -1
	    n_vectors = bin_vals.shape[0]
	    
	    pdf_vals = np.zeros(n_vectors)
	    for i in range(n_vectors):
                pdf_vals[i] = np.exp(np.dot(bin_vals[i, :], self.Q).dot(bin_vals[i, :].T))
		
	    norm_const = np.sum(pdf_vals)
	    
	    for i in range(self.n_vars):
               for j in range(self.n_vars):
                    bin_pair = bin_vals[:, i]*bin_vals[:, j]
                    ising_moments[i][j] = np.sum(bin_pair*pdf_vals)/norm_const
	    return ising_moments

	def run(self, x):
	    Theta_P = self.Q
	    bin_vals = np.zeros((2**self.n_vars, self.n_vars))
	    for i in range(2**self.n_vars):
                bin_vals[i] = np.array(list(np.binary_repr(i).zfill(self.n_vars))).astype(np.int8)
	    bin_vals[bin_vals == 0] = -1
	    n_vectors = bin_vals.shape[0]
	    P_vals = np.zeros(n_vectors)
	    for i in range(n_vectors):
                P_vals[i] = np.exp(np.dot(bin_vals[i, :], Theta_P).dot(bin_vals[i, :].T))
		
	    Zp = np.sum(P_vals)  # partition function of random variable P
	       
            Theta_Q = np.tril(Theta_P, -1)
            nnz_Q = Theta_Q!=0
            Theta_Q[nnz_Q] = Theta_Q[nnz_Q]*x[:].T
            Theta_Q = Theta_Q + Theta_Q.T
	    Q_vals = np.zeros(n_vectors)
            for i in range(n_vectors):
        	Q_vals[i] = np.exp(np.dot(bin_vals[i, :], Theta_Q).dot(bin_vals[i, :].T))
	    Zq = np.sum(Q_vals)	# partition function of random variable Q
            KL = np.sum(np.sum((Theta_P - Theta_Q)*self.ising_moments)) + np.log(Zq) - np.log(Zp)
            return -1 * (KL + self.lam*np.sum(x))
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
	sim = Ising_Model()
	(mv, mvec, kl) = sim.compute_max()
	print(mv)
	print(mvec)
	plt.plot(kl)
