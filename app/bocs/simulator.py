import numpy as np
from . import feature_generator as fg
np.random.seed(42)
class BPQ_Sim(object):
	def __init__(self, num_features, lam, alpha):
		self.num_features = num_features
		self.alpha = alpha
		self.lam = lam
		self.Q = self.quad_mat(self.num_features, self.alpha)
		return
	def get_config(self):
		return (self.num_features, self.lam, self.alpha)
	def quad_mat(self, n_vars, alpha):
 		# n_vars which is same as n_features which is same as no of variables in a BQP
		# evaluate decay function
		i = np.linspace(1,n_vars,n_vars)
		j = np.linspace(1,n_vars,n_vars)
		
		K = lambda s,t: np.exp(-1*(s-t)**2/alpha)
		decay = K(i[:,None], j[None,:])

		# Generate random quadratic model
		# and apply exponential decay to Q
		Q  = np.random.randn(n_vars, n_vars)
		Qa = Q*decay
		
		return Qa
	def compute_min(self):
		import feature_generator as fg
		fg = fg.FeatureGenerator(self.num_features)
		max_value = np.inf
		max_num_vec = []
		for num in range(2**self.num_features):
			num_vec = np.array([fg.get_binary(num, self.num_features)], dtype=np.int32)
			print(num_vec) 
			value = self.run(num_vec)
			if(value < max_value):
				max_value = value
				max_num_vec = num_vec
		return max_value, max_num_vec

	def run(self, feature_vector):
		x = feature_vector
		'''
		if(x.ndim != 1):
			print("ERROR in dimension ", x.ndim)
			return -1
		#model_val = (x.dot(self.Q)*x).sum() # compute x^TQx row-wise
		'''
		model_val = (x.dot(self.Q)*x).sum() # compute x^TQx row-wise
		penalty  = self.lam*np.sum(x)
		return (model_val - penalty)

	def run_list(self, feature_vector):
		x = feature_vector
#		model_val = (x.dot(self.Q)*x).sum(axis=1) # compute x^TQx row-wise
		model_val = (x.dot(self.Q)*x).sum(axis=1) # compute x^TQx row-wise
		penalty  = self.lam*np.sum(x,axis=1)
		return (model_val - penalty)

if __name__=="__main__":
	num_features = 10
	x1 = fg.FeatureGenerator(num_features).generate_n_random_feature(1)
	x = fg.FeatureGenerator(num_features).generate_n_random_feature(3)
	suc_vec = fg.FeatureGenerator(num_features).generate_successor_graph(x[0])
	for i in [100]:
		for j in [1e-4]:
			sim = BPQ_Sim(num_features, i, j)
			print(sim.compute_min())
			print(sim.run(np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])[np.newaxis, :]))
