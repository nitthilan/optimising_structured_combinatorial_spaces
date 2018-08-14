
import random
import numpy as np

class FeatureGenerator(object):
	# FEature restrictions:
	# min, max, discrete/continous, num_parts
	def __init__(self, sim):
		self.sim = sim
		num_features, _, _ = self.sim.get_config()
		self.feature_length = num_features
		self.num_features = num_features
		self.num_levels = 2
		return

	def get_num_iterations(self):
		return 1

	def get_binary(self, num, num_features):
		return np.array(list(np.binary_repr(num).zfill(self.num_features))).astype(np.float)

	def generate_random_feature(self):
		num = np.random.randint(2**self.num_features)
		# print(num)

		return self.get_binary(num, self.num_features)

	def generate_n_random_feature(self, num_rnds):
		feature_vector_list = []
		for i in range(num_rnds):
			feature_vector_list.append(self.generate_random_feature())
		return np.array(feature_vector_list)

	def generate_successor_graph(self, feature_vector):
		fv_vec_list = []
		for i in range(self.num_features):
			if(feature_vector[i]):
				for j in range(self.num_features):
					if(i != j and feature_vector[j] == 0):
						fv_cpy = feature_vector.copy()
						fv_cpy[i] -= 1
						fv_cpy[j] += 1
						fv_vec_list.append(fv_cpy)
		# print(len(fv_vec_list))
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
		num_features, alpha, lam \
			= self.sim.get_config()

		value_list = range(0, np.power(2, num_features), step_size)
		# print(value_list)

		feature_vector_list = []
		for value in value_list:
			feature_vector_list.append(self.get_discrete_vec(value, num_features))
		# print(value_list)

		return np.array(feature_vector_list), value_list


if __name__=="__main__":
	x = FeatureGenerator(10).generate_n_random_feature(3)
	suc_vec = FeatureGenerator(10).generate_successor_graph(x[0])
	print(suc_vec)
	print(x)
	x = FeatureGenerator(5).generate_n_random_feature(3)
	suc_vec = FeatureGenerator(5).generate_successor_graph(x[0])
	print(suc_vec)
	print(x)
