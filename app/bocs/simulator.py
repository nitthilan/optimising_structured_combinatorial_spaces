import numpy as np
class BPQ_Sim(object):
	def __init__(self, num_features, alpha, lam):
		self.num_features = num_features
		self.alpha = alpha
		self.lam = lam
		self.Q = self.quad_mat(self.num_features, self.alpha)
		return
	def get_config(self):
		return (self.num_features, self.alpha, self.lam)
	def quad_mat(self, n_vars, alpha):
 
		# evaluate decay function
		i = np.linspace(1,n_vars,n_vars)
		j = np.linspace(1,n_vars,n_vars)
		
		K = lambda s,t: np.exp(-1*(s-t)**2/alpha)
		decay = K(i[:,None], j[None,:])

		# Generate random quadratic model
		# and apply exponential decay to Q
		np.random.seed(42)
		Q  = np.random.randn(n_vars, n_vars)
		Qa = Q*decay
		
		return Qa

	def compute_min(self):
		import feature_generator as fg
		fg = fg.FeatureGenerator(self.num_features)
		max_value = -1*np.inf
		max_num_vec = []
		for num in range(2**self.num_features):
			num_vec = np.array([fg.get_binary(num, self.num_features)])
			value = self.run(num_vec)
			if(value > max_value):
				max_value = value
				max_num_vec = num_vec
		return max_value, max_num_vec

	def run(self, feature_vector):
		x = feature_vector
		if(x.ndim != 1):
			print("ERROR in dimension ", x.ndim)
			return -1
		model_val = (x.dot(self.Q)*x).sum() # compute x^TQx row-wise
		penalty  = self.lam*np.sum(x)
		return -1*(model_val + penalty)

	def run_list(self, feature_vector):
		x = feature_vector
		model_val = (x.dot(self.Q)*x).sum(axis=1) # compute x^TQx row-wise
		penalty  = self.lam*np.sum(x,axis=1)
		return -1*(model_val + penalty)

if __name__=="__main__":
	import feature_generator as fg
	num_features = 10
	x1 = fg.FeatureGenerator(num_features).generate_n_random_feature(1)

	x = fg.FeatureGenerator(num_features).generate_n_random_feature(3)
	suc_vec = fg.FeatureGenerator(num_features).generate_successor_graph(x[0])

	sim = BPQ_Sim(num_features, 10, 1e-4)

	print(sim.run(x1))
	print(sim.run(x))
	print(sim.run(suc_vec))
	print(sim.compute_min())
