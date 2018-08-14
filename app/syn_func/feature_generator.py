
import random
import numpy as np

class FeatureGenerator(object):
	# FEature restrictions:
	# min, max, discrete/continous, num_parts
	def __init__(self, sim):
		self.sim = sim
		num_features, is_discrete, num_levels, num_bits_per_dim = sim.get_config()
		# self.num_features = num_features
		self.is_discrete = is_discrete
		self.num_levels = num_levels
		# self.num_bits_per_dim = num_bits_per_dim
		if(self.is_discrete):
			self.feature_length = num_features*num_bits_per_dim
		else:
			self.feature_length = num_features
		return

	def get_num_iterations(self):
		return 1
	def generate_random_feature(self):
		x = np.random.rand(self.feature_length)
		if(self.is_discrete):
			x = x*self.num_levels#+0.5
			x = x.astype(int)
		# print(x)
		return x

	def generate_n_random_feature(self, num_rnds):
		feature_vector_list = []
		for i in range(num_rnds):
			feature_vector_list.append(self.generate_random_feature())
		return np.array(feature_vector_list)

	def generate_successor_graph(self, feature_vector):
		fv_vec_list = []
		if(self.is_discrete):
			for i in range(self.feature_length):
				fv_cpy = feature_vector.copy()
				# print("Index values ", i, feature_vector)

				fv_cpy[i] += 1
				if(fv_cpy[i] < self.num_levels):
					fv_vec_list.append(fv_cpy)
				fv_cpy = feature_vector.copy()
				fv_cpy[i] -= 1
				if(fv_cpy[i] >= 0):
					fv_vec_list.append(fv_cpy)
		else:
			for i in range(self.feature_length):
				for n in range(4):
					fv_cpy = feature_vector.copy()
					sigma = 0.1
					delta = random.gauss(0, sigma)
					fv_cpy[i] += delta
					# print(delta, sigma, fv_cpy[i], feature_vector[i])
					if(fv_cpy[i] <= 1 and fv_cpy[i] >= 0):
						fv_vec_list.append(fv_cpy)

			# print(i, fr)

		return np.array(fv_vec_list)

	def get_root_node(self):
		return (-1, np.zeros(self.feature_length))
	def expand_node(self, node_info):
		fea_idx, feature_vector = node_info
		if(fea_idx == self.feature_length-1):
			return []
		else:
			fea_idx += 1

		node_info_list = []
		for i in range(self.num_levels):
			fv_cpy = feature_vector.copy()
			fv_cpy[fea_idx] = i
			node_info = (fea_idx, fv_cpy)
			node_info_list.append(node_info)
		return node_info_list
	def expand_node_randomly(self, node_info, num_rnds):
		fea_idx, feature_vector = node_info
		if(fea_idx == self.feature_length-1):
			return np.array([feature_vector])
		else:
			fea_idx += 1
		rand_fea_vec_list = self.generate_n_random_feature(num_rnds)

		rand_fea_vec_list[:,:fea_idx] = feature_vector[:fea_idx]

		# feature_vector_list = []
		# for i in range(num_rnds):
		# 	fv_cpy = feature_vector.copy()
		# 	fea_rnd_idx = random.randint(fea_idx, self.feature_length-1)
		# 	prt_rnt_idx = random.randint(0, self.num_levels-1)
		# 	fv_cpy[fea_rnd_idx] = prt_rnt_idx
		# 	feature_vector_list.append(fv_cpy)
		return rand_fea_vec_list
	def get_node(self, node_info):
		_, node = node_info
		return node
	def get_correlation(self, node_info, feature_vector_b):
		fea_idx_a, feature_vector_a = node_info
		fv_a = feature_vector_a[:fea_idx_a]
		fv_b = feature_vector_b[:fea_idx_a]
		if np.all(fv_b == fv_a):
			correlation = fea_idx_a*1.0/self.feature_length
		else:
			correlation = 0.0
		return correlation

	def get_discrete_vec(self, value, feature_length):
		vector = np.zeros(feature_length)
		idx = feature_length-1
		while(value):
			level = value%self.num_levels
			value = int(value/self.num_levels)
			vector[idx] = level
			idx -= 1
		return vector

	def get_all_combinations(self, step_size):
		num_features, is_discrete, num_levels, num_bits_per_dim \
			= self.sim.get_config()

		feature_length = num_bits_per_dim*num_features
		value_list = range(0, \
			np.power(self.num_levels, feature_length), step_size)
		# print(value_list)

		feature_vector_list = []
		for value in value_list:
			feature_vector_list.append(self.get_discrete_vec(value, feature_length))
		# print(value_list)

		return np.array(feature_vector_list), value_list

if __name__=="__main__":
	import simulator as sim

	sim1 = sim.Simulator("borehole", True, 100, 1)
	x = FeatureGenerator(sim1).generate_n_random_feature(3)
	suc_vec = FeatureGenerator(sim1).generate_successor_graph(x[0])
	print(suc_vec)
	print(x)
	x = FeatureGenerator(sim1).generate_n_random_feature(3)
	suc_vec = FeatureGenerator(sim1).generate_successor_graph(x[0])
	print(suc_vec)
	print(x)
	sim2 = sim.Simulator("borehole", True, 10, 1)
	fg = FeatureGenerator(sim2)
	root_node = fg.get_root_node()
	node_info_list = fg.expand_node(root_node)
	rnd_node_list = fg.expand_node_randomly(root_node, 10)
	rnd_fst_node_list = fg.expand_node_randomly(node_info_list[2], 10)
	print(root_node)
	print(node_info_list)
	print(rnd_node_list)
	print(rnd_fst_node_list)

	sim2 = sim.Simulator("univariate", True, 2, 10)
	fg = FeatureGenerator(sim2)
	# x = fg.get_all_combinations(1)
	# print(x.shape)
	# print(x[:4])
	x = fg.get_all_combinations(10)
	print(x.shape)
	print(x[:4])
	for i in range(10):
		print(sim2.get_continous_value(x[i],0)*1024)

