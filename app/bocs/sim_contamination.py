'''
Python implementation of Contamination Model
from Bayesisan Optimization of Combinatorial structures paper
Authors: Ricardo Baptista & Matthias Poloczek
Original Matlab code at https://github.com/baptistar/BOCS/tree/master/test_problems/ContStudy
'''
import numpy as np
import matplotlib.pyplot as plt
import feature_generator as fg

class Contamination_Model(object):
	def __init__(self):
		# input parameters
		self.n_samples = 20 # no of Monte Carlo samples
		self.num_features = 20 # length of food chain
		self.lam = 1e-4 # regularization constant
	def get_config(self):
		return (self.num_features, self.n_samples, self.lam)
	
	def contamination(self, x):
	    n = x.shape[0]
	    nGen = self.n_samples   # no of samples to generate
	    # contamination variable
	    Z = np.zeros((n, nGen))
	    epsilon = 0.05 * np.ones(n)
	    u = 0.1*np.ones(n)
	    cost = np.ones(n)
	    # Beta parameters
	    initialAlpha=1
	    initialBeta=30
	    contamAlpha=1
	    contamBeta=17/3
	    restoreAlpha=1
	    restoreBeta=3/7
	    
	    # generate initial contamination fraction for each sample
	    initialZ = np.random.beta(initialAlpha, initialBeta, nGen)
	    # generate rates of contamination for each stage and sample
	    lambdad = np.random.beta(contamAlpha, contamBeta, (n ,nGen))
	    # generate rates of contamination for each stage and sample
	    gamma = np.random.beta(restoreAlpha, restoreBeta, (n, nGen))
	    
	    
	    # calculate rates of contamination 
	    Z[1, :] = lambdad[1, :]*(1-x[1])*(1-initialZ) + (1-gamma[1, :])*(1-x[1])*initialZ
	    for i in range(2, n):
                Z[i, :] = lambdad[i, :]*(1-x[i-1])*(1-Z[i-1, :]) + (1-gamma[i, :])*(1-x[i-1])*Z[i-1, :]
	    print(Z)

	    limit = 1-epsilon
	    # cost of contamination
	    fn = np.sum(np.dot(cost, x))
	    # checking probability that Xi is <=pi is less than 1-eps
	    con = np.zeros((n, nGen))
	    for j in range(nGen):
                con[:, j] = Z[:, j] <= u
	    print(con)
	    
	    con = con.T
	    constraint = np.zeros(n)
	    for i in range(n):
                constraint[i] = (np.sum(con[:, i])/nGen)-limit[i]     
	    return (fn, constraint)

	def run(self, x):
	    gamma = 1
	    (cost, constraint) = self.contamination(x)
	    return -1 * (cost - np.sum(gamma*constraint) + self.lam*np.sum(x))

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
	sim = Contamination_Model()
	x = np.random.randint(0, 1, 20)
	print(sim.run(x))
