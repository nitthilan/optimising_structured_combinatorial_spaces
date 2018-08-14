
import random
import numpy as np
import constants_tsv as CT


class random_tsv:
	def __init__(self):
		vlIndexList = []
		for i in range(48):
			if i in range(16,32):
				vlIndexList.append(i)
				vlIndexList.append(i)
			else:
				vlIndexList.append(i)
		self.vlIndexList = vlIndexList
		# print(self.vlIndexList)

		tsvIndexList = []
		for i in range(8):
			if i in [0,2,6]:
				tsvIndexList.append(i)
			elif i in [1,3,5,7]:
				tsvIndexList.append(i)
				tsvIndexList.append(i)
			else:
				for j in range(12):
					tsvIndexList.append(i)
		self.tsvIndexList = tsvIndexList
		# print(self.tsvIndexList)
		self.generate_tsv_map()

		self.generate_priority_map()
		return

	def generate_priority_map(self):
		self.priority_map = []
		self.inv_priority_map = np.zeros(384)
		for i in range(384):
			self.inv_priority_map[i] = len(self.priority_map)

			if(i<16):
				for j in range(12*2):
					self.priority_map.append(i)
			elif(i<48):
				for j in range(12):
					self.priority_map.append(i)
			elif(i<48+4*16):
				for j in range(2*2):
					self.priority_map.append(i)
			elif(i<48+4*16+4*32):
				for j in range(2):
					self.priority_map.append(i)
			elif(i<48+4*16+4*32+3*16):
				for j in range(2*1):
					self.priority_map.append(i)
			elif(i<48+4*16+4*32+3*16+3*32):
				for j in range(1):
					self.priority_map.append(i)
		# print(self.priority_map)
		# print(self.inv_priority_map)
		return

	def rand_idx_based_on_priority_1(self, start):
		start_idx = int(self.inv_priority_map[start])

		return int(np.random.choice(self.priority_map[start_idx:]))

	def rand_idx_based_on_priority(self):
		i = np.random.choice(self.vlIndexList)
		j = np.random.choice(self.tsvIndexList)	
		return int(self.tsv_map[i, j]), i, j

	# Create a tsv mapping matrix to map from 2D to 1D feature vector
	def generate_tsv_map(self):
		
		tsv_matrix = np.zeros((CT.NUM_VERTICAL_LINKS, CT.NUM_TSV_PER_LINK))
		
		# layer offsets:
		# Top-middle-bottom
		top_off = 0
		mid_off = 16
		bot_off = 32

		# TSV position offsets
		# center [4], edge[1,3,5,7], corners [0,2,6]
		cen_idx = 4
		edg_idx = [1,3,5,7]
		cor_idx = [0,2,6]

		def fill_edge(tsv_matrix, tsv_off, fea_off):
			tsv_matrix[tsv_off+i, edg_idx[0]] =\
				fea_off+0+4*i
			tsv_matrix[tsv_off+i, edg_idx[1]] =\
				fea_off+1+4*i
			tsv_matrix[tsv_off+i, edg_idx[2]] =\
				fea_off+2+4*i
			tsv_matrix[tsv_off+i, edg_idx[3]] =\
				fea_off+3+4*i
			return

		def fill_corner(tsv_matrix, tsv_off, fea_off):
			tsv_matrix[tsv_off+i, cor_idx[0]] =\
				fea_off+0+3*i
			tsv_matrix[tsv_off+i, cor_idx[1]] =\
				fea_off+1+3*i
			tsv_matrix[tsv_off+i, cor_idx[2]] =\
				fea_off+2+3*i
			return
		
		# Feature vector ordering
		# [center-16]
		for i in range(16):
			
			# center
			tsv_matrix[mid_off+i, cen_idx] = i # Middle-center
			tsv_matrix[top_off+i, cen_idx] = 16+i # Top-center
			tsv_matrix[bot_off+i, cen_idx] = 32+i # Bottom-center
			# edge
			fill_edge(tsv_matrix, mid_off, 48) # Middle-edge
			fill_edge(tsv_matrix, top_off, 48+16*4) # Top-edge
			fill_edge(tsv_matrix, bot_off, 48+2*16*4) # Bottom-edge
			# corner
			fill_corner(tsv_matrix, mid_off, 48+3*16*4) # Middle-corner
			fill_corner(tsv_matrix, top_off, 48+3*16*4+16*3) # Top-corner
			fill_corner(tsv_matrix, bot_off, 48+3*16*4+2*16*3) # Bottom-corner

		self.tsv_map = tsv_matrix

		return

class FeatureGenerator(object):
	# FEature restrictions:
	# min, max, discrete/continous, num_parts
	def __init__(self, max_num_spare_tsv, max_num_levels):
		self.rand_tsv_idx_gen = random_tsv()
		self.max_num_spare_tsv = max_num_spare_tsv
		self.max_num_levels = max_num_levels

		self.feature_length = CT.NUM_TSVS
		return

	def get_num_iterations(self):
		return 1

	def generate_random_feature(self, start_idx, num_spare_tsv):
		random_design = np.zeros(CT.NUM_TSVS, dtype=np.int)
		# print("Start ")
		# Currently making sure the TSVs are allocated within the max
		# Center TSV range to get good init points for modeling
		# 48 is the hardcoded value for the center TSV
		# Since init points are returing zero the model seems to return 
		# zero for AFO
		# restrict_to_center = RESTRICT_SEARCH
		# fill_random_design(random_design, 0, restrict_to_center, max_num_spare_tsv)

		# fill_based_on_priority(random_design, max_num_spare_tsv)
		total_alloc = num_spare_tsv
		while(total_alloc>0):
			# dsg_idx, _, _ = \
			# 	self.rand_tsv_idx_gen.rand_idx_based_on_priority(start_idx)
			dsg_idx = \
				self.rand_tsv_idx_gen.rand_idx_based_on_priority_1(start_idx)
			random_design[dsg_idx] += 1
			total_alloc -= 1
		# print(random_design)
		if(np.sum(random_design) != num_spare_tsv):
			print("Error in random design generation ", random_design, num_spare_tsv)

		return random_design

	def generate_n_random_feature(self, num_rnds):
		feature_vector_list = []
		for i in range(num_rnds):
			feature_vector_list.append(\
				self.generate_random_feature(0, self.max_num_spare_tsv))
		return np.array(feature_vector_list)
	def generate_n_random_with_offset(self, num_rnds, start_idx, num_spare_tsv):
		feature_vector_list = []
		for i in range(num_rnds):
			feature_vector_list.append(\
				self.generate_random_feature(start_idx, num_spare_tsv))
		return np.array(feature_vector_list)

	def gen_succ_region(self, feature_vector, range_list):
		fv_vec_list = []
		for i in range_list:
			if(feature_vector[i]):
				for j in range_list:
					if(i != j):
						fv_cpy = feature_vector.copy()
						fv_cpy[i] -= 1
						fv_cpy[j] += 1
						fv_vec_list.append(fv_cpy)
		# print(len(fv_vec_list))
		return fv_vec_list


	def generate_successor_graph(self, feature_vector):
		fv_vec_list = []
		fv_vec_list.extend(self.gen_succ_region(feature_vector, range(48)))
		fv_vec_list.extend(self.gen_succ_region(feature_vector, range(48,5*48)))
		fv_vec_list.extend(self.gen_succ_region(feature_vector, range(5*48,8*48)))
		# print(i, fr)

		return np.array(fv_vec_list)


	def get_root_node(self):
		return (-1, np.zeros(self.feature_length))
	def expand_node(self, node_info):
		fea_idx, feature_vector = node_info
		num_tsv_allocated = np.sum(feature_vector[:fea_idx+1])
		num_avail_tsv = int(self.max_num_spare_tsv - num_tsv_allocated)

		num_levels = self.max_num_levels
		if(num_avail_tsv < num_levels):
			num_levels = num_avail_tsv


		if(fea_idx == self.feature_length-1 or num_levels == 0):
			return []
		else:
			fea_idx += 1

		node_info_list = []
		for i in range(num_levels+1):
			fv_cpy = feature_vector.copy()
			fv_cpy[fea_idx] = i
			node_info = (fea_idx, fv_cpy)
			node_info_list.append(node_info)
		return node_info_list
	def expand_node_randomly(self, node_info, num_rnds):
		fea_idx, feature_vector = node_info
		num_spare_tsv = self.max_num_spare_tsv - np.sum(feature_vector[:fea_idx])
		if(num_spare_tsv < 0):
			print("ERROR in spare tsv ", num_spare_tsv, feature_vector, fea_idx)
	
		if(fea_idx == self.feature_length-1 or num_spare_tsv == 0):
			return np.array([feature_vector])
		else:
			fea_idx += 1

		rand_fea_vec_list = \
			self.generate_n_random_with_offset(num_rnds, fea_idx, num_spare_tsv)

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



if __name__=="__main__":
	x = FeatureGenerator(15,2).generate_n_random_feature(3)
	print(np.sum(x,axis=1))
	print(x[0])

	suc_vec = FeatureGenerator(5,2).generate_successor_graph(x[0])
	print(suc_vec.shape)
	x = FeatureGenerator(5,2).generate_n_random_feature(3)
	suc_vec = FeatureGenerator(5,2).generate_successor_graph(x[0])
	print(suc_vec.shape)
	# print(x)
	# hist_value = np.zeros(384)
	# rand_tsv_idx_gen = random_tsv()
	# for i in range(100000):
	# 	hist_value[rand_tsv_idx_gen.rand_idx_based_on_priority_1(8)] += 1
	# print(hist_value)

	fg = FeatureGenerator(5,2)
	root_node = fg.get_root_node()
	node_info_list = fg.expand_node(root_node)
	rnd_node_list = fg.expand_node_randomly(root_node, 10)
	rnd_fst_node_list = fg.expand_node_randomly(node_info_list[2], 10)
	# print(root_node)
	# print(node_info_list)
	print(rnd_node_list)
	print(rnd_fst_node_list)
	for fv in rnd_node_list:
		print(np.sum(fv))
	for fv in rnd_fst_node_list:
		print(np.sum(fv))

