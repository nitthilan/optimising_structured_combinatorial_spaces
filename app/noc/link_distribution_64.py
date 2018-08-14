
import numpy as np
import pandas as pd
import constants as C
import utility_functions as U
import random
from collections import defaultdict
import time



def generate_link_distribution(alpha, k_avg, num_cores, max_link_length, num_layers):
	total_links = k_avg*num_cores/2
	scale_factor = 0
	distribution = np.zeros(max_link_length)
	for i in range(max_link_length):
		scale_factor_val = (1/((1+i)**alpha))
		# print(scale_factor_val)
		scale_factor += scale_factor_val
	for i in range(max_link_length):
		scale_factor_val = (1/((1+i)**alpha))
		distribution[i] = (scale_factor_val/scale_factor)
	num_link_dist = np.rint(distribution*total_links)
	# print(num_link_dist)
	# Remove the number of vertical links of length 1
	num_link_dist[0] -= U.get_num_vertical_link(num_layers, num_cores)
	return(num_link_dist)

def get_distance_distribution(num_width, num_height, cur_x, cur_y):
	distance_distribution = np.zeros((num_width, num_height))
	for i in range(num_height):
		for j in range(num_width):
			distance_distribution[i,j] = U.get_distance(i,j,cur_x,cur_y)

	return distance_distribution

# Generate all the possible connection options that are possible
# The order in which they are generated is used for generating feature vectors
def generate_core_connection_options():
	core_pos = U.get_2d_core_pos()
	total_cores = len(core_pos)
	core_connection_options = []
	#
	for i in range(total_cores):
		for j in range(i+1,total_cores):
			distance = int(round(U.get_core_distance(core_pos[i], core_pos[j])))
			for k in range(C.Z_LYR):
				start_core = U.get_node_index((k, core_pos[i][1], core_pos[i][0]))
				end_core = U.get_node_index((k, core_pos[j][1], core_pos[j][0]))
				# Store the distance, start and end core idx
				core_connection_options.append((start_core, end_core, distance))
				# print(core_connection_options[-1][0], 
				# 	core_connection_options[-1][1], core_connection_options[-1][2])
	print("Length of connection option ", len(core_connection_options))
	return core_connection_options


# Currently not used. Dumps the distribution of the number of links for each distance value
def generate_core_distance_distribution():
	core_pos = U.get_2d_core_pos()
	# For each core based on the distance bucket the connection list
	total_cores = len(core_pos)
	max_distance = U.GET_MAX_DISTANCE()
	core_distance_distribution = []
	for i in range(max_distance):
		core_distance_distribution.append([])

	for i in range(total_cores):
		for j in range(i+1,total_cores):
			distance = int(round(U.get_core_distance(core_pos[i], core_pos[j])))
			# print(start_core, end_core, int(round(distance)))
			for k in range(C.Z_LYR):
				start_core = U.get_node_index((k, core_pos[i][1], core_pos[i][0]))
				end_core = U.get_node_index((k, core_pos[j][1], core_pos[j][0]))
				core_distance_distribution[distance-1].append((start_core, end_core))

	# For a 4x4 grid the distribution is
	# (0, 42) (1, 40) (2, 28) (3, 10)
	# (0, 168) (1, 160) (2, 112) (3, 40)
	for i in range(max_distance):
		print(i, len(core_distance_distribution[i]))

	return core_distance_distribution

def generate_similar_value_idx(sort_idx, x_feature):
	dict_idx_list = defaultdict(list)
	for idx in sort_idx:
		val = -1*int(round(x_feature[idx]*1000))
		dict_idx_list[val].append(idx)
	keys_idx = np.array(dict_idx_list.keys())
	# print(-1*np.array(keys_idx), np.sort((-1*keys_idx)))
	return (dict_idx_list, np.sort(keys_idx))


def generate_connection_list(feature_vector):
	core_connection_options = generate_core_connection_options()
	# Recreate the connection list from the feature vector
	connection_idx_list = []
	for idx in range(feature_vector.shape[0]):
		if feature_vector[idx]:
			# print(idx, feature_vector.shape)
			start_core =  core_connection_options[idx][0]
			end_core = core_connection_options[idx][1]
			connection_idx_list.append((end_core, start_core))
	# Add the vertical link list to the connection list
	vertical_link_list = U.generate_vertical_links_list()
	connection_idx_list.extend(vertical_link_list)
	return connection_idx_list




class LinkDistribution(object):
	def __init__(self):
		# Generate all the single link combination
		self.core_connection_options = generate_core_connection_options()
		self.total_connections = len(self.core_connection_options)

		self.reset()
		return

	def reset(self):
		# Since Max distance on a layer would be less than 4
		# max distance = sqrt(3^2+3^2) = 4.24
		# Basic link distribution = [ 63.   21.   8.    4.]
		self.link_distribution = [ 63.,  21.,   8.,   4.] # generate_link_distribution(C.ALPHA, C.K_AVG, C.NUM_CORES, U.GET_MAX_DISTANCE(), C.Z_LYR)
		# print(self.link_distribution)
		
		# Initialise all nodes as unconnected
		self.is_core_connected = np.zeros(C.NUM_CORES)
		self.is_connection_available = np.ones(self.total_connections)
		self.num_core_connected = C.K_MAX*np.ones(C.NUM_CORES)
		num_core_in_layer = C.X_WTH*C.Y_HGT
		# for i in range(C.X_WTH*C.Y_HGT):
		self.num_core_connected[0 : num_core_in_layer] -= 1
		self.num_core_connected[num_core_in_layer : 3*num_core_in_layer] -= 2
		self.num_core_connected[3*num_core_in_layer : 4*num_core_in_layer] -= 1

		return

	def check_connection_adds_new_node(self,conn_opt):
		# print(is_core_connected_list[conn_opt[0]], is_core_connected_list[conn_opt[1]])
		# return ((self.is_core_connected[conn_opt[0]] and not self.is_core_connected[conn_opt[1]])
		# 	 or(not self.is_core_connected[conn_opt[0]] and self.is_core_connected[conn_opt[1]]))
		# print(self.is_core_connected[conn_opt[0]] and self.is_core_connected[conn_opt[1]])
		return not (self.is_core_connected[conn_opt[0]] and self.is_core_connected[conn_opt[1]])

	def is_valid_connection(self, conn_opt, conn_idx):
		distance = conn_opt[2]
		start_idx = conn_opt[0]
		end_idx = conn_opt[1]
		return (self.link_distribution[distance-1]
			and self.is_connection_available[conn_idx]
			and self.num_core_connected[start_idx]
			and self.num_core_connected[end_idx])

	def check_graph_complete(self):
		# print(np.sum(self.is_core_connected), np.sum(self.link_distribution))
		return (not (np.sum(self.is_core_connected) == 64 
			and np.sum(self.link_distribution) == 0))

	# Iterate over all the connections and identify the conection which has one node in graph and another outside the graph
	# Also make sure the connection is valid
	# Collect all such nodes and randomly choose one connection
	# Repat this until SW conditions and all the nodes are connected
	def get_next_random_connection(self):
		conn_opts_idx = []
		# If all cores are not connected connect them first
		if(np.sum(self.is_core_connected) != 64):
			for i in range(self.total_connections):
				conn_opt = self.core_connection_options[i]
				# print(conn_opt, 
				# 	check_connection_adds_new_node(is_core_connected_list, conn_opt), 
				# 	link_distribution[conn_opt[2]-1])
				if(self.check_connection_adds_new_node(conn_opt) 
				and self.is_valid_connection(conn_opt, i)):
					conn_opts_idx.append(i)
		# Once all the cores are connected iterate till link distribution goes to zero
		elif(np.sum(self.link_distribution) != 0):
			for i in range(self.total_connections):
				conn_opt = self.core_connection_options[i]
				# print(conn_opt, 
				# 	check_connection_adds_new_node(is_core_connected_list, conn_opt), 
				# 	link_distribution[conn_opt[2]-1])
				if(self.is_valid_connection(conn_opt, i)):
					conn_opts_idx.append(i)
		
		# else: return a empty array

		# print(len(conn_opts_idx))
		if(len(conn_opts_idx)-1 and len(conn_opts_idx)):
			# print(conn_opts_idx)
			rand_idx = random.randint(0,len(conn_opts_idx)-1)
		else:
			rand_idx = 0
		# print (len(conn_opts_idx), rand_idx)
		conn_idx = conn_opts_idx[rand_idx]
		# print(core_connection_options[conn_idx])

		# return conn_opts_idx
		return conn_idx

	# Similar to the get_next_random_connection but instead of randomly choosing
	# choose a random connection whose value is close to the continous X feature
	# value returned (in this case by DIRECT algo)
	def get_next_connection_by_approx(self, dict_idx_list, sort_dict_idx):
		# If all cores are not connected connect them first
		if(np.sum(self.is_core_connected) != 64):
			for j in sort_dict_idx:
				idx_list = np.array(dict_idx_list[j])
				np.random.shuffle(idx_list)
				# print(j, dict_idx_list[j], np.array(dict_idx_list[j]),
				# 	np.random.shuffle())
				for i in idx_list:
					# print(i)
					# idx = sort_idx[i]
					conn_opt = self.core_connection_options[i]
					# print(conn_opt, 
					# 	check_connection_adds_new_node(is_core_connected_list, conn_opt), 
					# 	link_distribution[conn_opt[2]-1])
					if(self.check_connection_adds_new_node(conn_opt) 
					and self.is_valid_connection(conn_opt, i)):
						# print("cc", is_core_connected_list, i, conn_opt)
						return i
		# Once all the cores are connected iterate till link distribution goes to zero
		elif(np.sum(self.link_distribution) != 0):
			for j in sort_dict_idx:
				idx_list = np.array(dict_idx_list[j])
				np.random.shuffle(idx_list)
				for i in idx_list:
					conn_opt = self.core_connection_options[i]
					# print(conn_opt, 
					# 	check_connection_adds_new_node(is_core_connected_list, conn_opt), 
					# 	link_distribution[conn_opt[2]-1])
					if(self.is_valid_connection(conn_opt, i)):
						# print("ld", link_distribution, i, conn_opt)
						return i
		# return -1. IDeally should not happen
		return -1

	def _set_is_core_connected(self, is_core_connected, conn_idx, num_core_connected):
		# Set the corresponding
		(sz,sy,sx) = U.get_node_position(self.core_connection_options[conn_idx][0])
		(ez,ey,ex) = U.get_node_position(self.core_connection_options[conn_idx][1])
		num_core_connected[self.core_connection_options[conn_idx][0]] -= 1
		num_core_connected[self.core_connection_options[conn_idx][1]] -= 1
		for i in range(C.Z_LYR):
			is_core_connected[U.get_node_index((i,sy,sx))] = 1
			is_core_connected[U.get_node_index((i,ey,ex))] = 1
		return is_core_connected

	def _set_flags(self, conn_idx):
		self._set_is_core_connected(self.is_core_connected, conn_idx, self.num_core_connected)
		self.is_connection_available[conn_idx] = 0			
		# print(is_core_connected, conn_idx, start_idx, end_idx, int(core_connection_options[conn_idx][2]))
		self.link_distribution[int(self.core_connection_options[conn_idx][2]-1)] -= 1
		# print(link_distribution)
		return

	# Generate a random graph or a closest approximation to the given feature vector
	def generate_valid_graph(self, x_feature=[]):
		connection_list = []
		connection_idx_list = []
		self.reset()


		if(len(x_feature) == 0):
			# Bootstrap the first connection
			conn_idx = random.randint(0,self.total_connections-1)
			#print("Random")
		else:
			#print("Direct")
			# Sort the input feature vector and use it as a probability distribution to 
			# approximate the graph
			# Multiplied by -1 to reverse the sorting order
			sort_idx = np.argsort(-1*x_feature)
			dict_idx_list, sort_dict_idx = generate_similar_value_idx(sort_idx, x_feature)
			# Bootstrap the first connection
			idx_list = np.array(dict_idx_list[sort_dict_idx[0]])
			np.random.shuffle(idx_list)
			conn_idx = idx_list[0]
	
		while(True):
			conn_info = self.core_connection_options[conn_idx]
			# Add the connetion to the graph list
			connection_list.append((conn_info[1], conn_info[0]))
			connection_idx_list.append(conn_idx)

			self._set_flags(conn_idx)
			# # Set the corresponding
			# (sz,sy,sx) = U.get_node_position(self.core_connection_options[conn_idx][0])
			# (ez,ey,ex) = U.get_node_position(self.core_connection_options[conn_idx][1])
			# for i in range(C.Z_LYR):
			# 	self.is_core_connected[U.get_node_index((i,sy,sx))] = 1
			# 	self.is_core_connected[U.get_node_index((i,ey,ex))] = 1
			# self.is_connection_available[conn_idx] = 0
			# # print(is_core_connected, conn_idx, start_idx, end_idx, int(core_connection_options[conn_idx][2]))
			# self.link_distribution[int(self.core_connection_options[conn_idx][2]-1)] -= 1
			# # print(link_distribution)

			if(not self.check_graph_complete()):
				break

			if(len(x_feature) == 0):
				# Based on the connected cores and link distribution get the next set of connection options
				conn_idx = self.get_next_random_connection()
			else:
				# Based on the connected cores and link distribution get the next set of connection options
				conn_idx = self.get_next_connection_by_approx(dict_idx_list, sort_dict_idx)

			
		return (connection_list, connection_idx_list)




	def check_sw_connectivity(self, x_feature):
		self.reset()
		# print(np.sum(self.is_core_connected), np.sum(self.link_distribution), self.check_graph_complete())

		for i in range(U.GET_NUM_CONNECTION()):
			if(x_feature[i]):
				self._set_flags(i)
		print(np.sum(self.is_core_connected), np.sum(self.link_distribution), 
			self.check_graph_complete(), self.num_core_connected)
		return not self.check_graph_complete()


	def generate_successor_graph(self, feature_vector, length_list):
		# find the 
		len_conn_list = []
		len_unconn_list = []
		conn_list = []
		# Initialise a empty array
		for i in range(U.GET_MAX_DISTANCE()):
			len_conn_list.append([])
			len_unconn_list.append([])
		# based on feature vector bucket connection based on length
		for i in range(U.GET_NUM_CONNECTION()):
			core_connec_opt = self.core_connection_options[i]
			length = int(core_connec_opt[2]-1)
			if(feature_vector[i]):
				len_conn_list[length].append(i)
				conn_list.append(i)
			else:
				len_unconn_list[length].append(i)

		# for i in range(U.GET_MAX_DISTANCE()):
		# 	print(len(len_conn_list[i]), 
		# 		len(len_unconn_list[i]), len(conn_list),
		# 		np.sum(feature_vector))

		successor_graphs = []
		# generate for each length requested
		for length in length_list:
			for conn_idx in len_conn_list[length]:
				# Flag to check whether the graph is connected
				is_core_connected = np.zeros(C.NUM_CORES)
				num_core_connected = C.K_MAX*np.ones(C.NUM_CORES)
				for i in range(16):
					num_core_connected[i] -= 1
					num_core_connected[16+i] -= 2
					num_core_connected[32+i] -= 2
					num_core_connected[48+i] -= 1

				# Initialise the flag with all the connections except the one selected
				for conn_idx_1 in conn_list:
					if(conn_idx != conn_idx_1):
						self._set_is_core_connected(is_core_connected, 
							conn_idx_1, num_core_connected)

				# For each connection in the unconnected list check whether the graph is connected
				for unconn_idx in len_unconn_list[length]:

					is_core_connected_copy = np.copy(is_core_connected)
					num_core_connected_copy = np.copy(num_core_connected)
					self._set_is_core_connected(is_core_connected_copy, 
						unconn_idx, num_core_connected_copy)
					unconn_opt = self.core_connection_options[unconn_idx]
					# print(unconn_opt, num_core_connected_copy, is_core_connected_copy)
					if( np.sum(is_core_connected_copy) == 64 
					and num_core_connected_copy[unconn_opt[0]] >= 0
					and num_core_connected_copy[unconn_opt[1]] >= 0):
						feature_vector_copy = np.copy(feature_vector)
						feature_vector_copy[unconn_idx] = 1
						feature_vector_copy[conn_idx] = 0
						successor_graphs.append(feature_vector_copy)

						# sw_dist = [0,0,0,0]
						# for i in range(U.GET_NUM_CONNECTION()):
						# 	if(feature_vector_copy[i]):
						# 		len_val = int(self.core_connection_options[i][2]-1)
						# 		sw_dist[len_val] += 1
						# print("Distribution", sw_dist)

		# ('Num Successors all ', 10510)
		# ('Num Successors 0', 6615)
		# ('Num Successors 1', 2919)
		# ('Num Successors 2', 832)
		# ('Num Successors 3', 144)
		# 63*105 + 21*139 + 8*104 + 4*36 = 10510 - This is the ideal case
		# But the restriction of max links per router  C.K_MAX(7) reduces the number
		# based on the init graph choosen. Some sample numbers are:
		# ('Num Successors ', 0, 4936)
		# ('Num Successors ', 0, 5078)
		# ('Num Successors ', 0, 5928)
		# ('Num Successors ', 1, 2210)
		# ('Num Successors ', 1, 2207)
		# ('Num Successors ', 1, 2375)
		# ('Num Successors ', 2, 703)
		# ('Num Successors ', 2, 702)
		# ('Num Successors ', 3, 128)
		# ('Num Successors ', 2, 691)
		# ('Num Successors ', 3, 128)
		# ('Num Successors ', 3, 128)

		# ('Num Successors all', 8813)
		# ('Num Successors all', 8348)
		# ('Num Successors all', 8918)
		# ('Num Successors all', 8813)
		# ('Num Successors all', 8100)
		# ('Num Successors all', 9394)


		# print("Num Successors ", length, len(successor_graphs))
		return np.asarray(successor_graphs)

	def generate_successor_graph_1(self, feature_vector):
		unordered_feature_vector = []
		
		connection_idx_list = \
			U.generate_conn_idx_list_list(np.asarray([feature_vector]))[0]

		len_excess_conn_list = []
		# Initialise a empty array
		for i in range(U.GET_MAX_DISTANCE()):
			len_excess_conn_list.append([])
		idx = 0
		for conn_idx in connection_idx_list:
			core_connec_opt = self.core_connection_options[conn_idx]
			length = int(core_connec_opt[2]-1)
			len_excess_conn_list[length].append(idx)
			idx = idx + 1

		# Switch all some random number of switches
		for i in range(U.GET_MAX_DISTANCE()):
			random_idx = random.randint(0, len(len_excess_conn_list[i])-1)
			conn_idx_list_idx = len_excess_conn_list[i][random_idx]
			conn_idx = connection_idx_list[conn_idx_list_idx]
			length = int(self.core_connection_options[conn_idx][2]-1)
			
			self.reset()
			for conn_idx_1 in connection_idx_list:
				if(conn_idx_1 != conn_idx):
					self._set_flags(conn_idx_1)
			new_conn_idx = self.get_next_random_connection()
			# Replace the selected connection with new index
			connection_idx_list[conn_idx_list_idx] = new_conn_idx

			unordered_feature_vector.append(U.generate_feature_list([connection_idx_list])[0])
			# print("Conn idx ", conn_idx_list_idx, 
			# 	int(self.core_connection_options[conn_idx][2]-1),
			# 	int(self.core_connection_options[new_conn_idx][2]-1),
			# 	self.link_distribution)

			# sw_dist = [0,0,0,0]
			# for conn_idx_1 in connection_idx_list:
			# 	sw_dist[int(self.core_connection_options[conn_idx_1][2]-1)] += 1
			# print("Distribution", sw_dist)


		return np.asarray(unordered_feature_vector)


	def set_base_graph(self, feature_vector):
		self.connection_idx_list = []
		self.excess_conn_list = []
		self.reset()
		self.connection_idx_list = \
			U.generate_conn_idx_list_list(np.asarray([feature_vector]))[0]

		idx = 0
		for conn_idx in self.connection_idx_list:
			core_connec_opt = self.core_connection_options[conn_idx]
			length = int(core_connec_opt[2]-1)

			# Store the index in the connection idx list
			# if the connection is not adding a new node
			# i.e. the excess node list
			if((not self.check_connection_adds_new_node(core_connec_opt))):
				self.excess_conn_list.append(idx)
				
			self._set_flags(conn_idx)
			idx = idx + 1

		# print("The num excess connection length ", len(self.excess_conn_list))

		# self.feature_vector = feature_vector

		return


	def generate_nearest_random_graph(self, num_switches):

		conn_idx_list_to_reset = []
		connection_idx_list = []
		
		# Create a local copy so that the old state is maintained
		for i in range(len(self.connection_idx_list)):
			connection_idx_list.append(self.connection_idx_list[i])

		# Switch all some random number of switches
		for i in range(num_switches):		
			random_idx = random.randint(0, len(self.excess_conn_list)-1)
			# print("Random idx ", random_idx)
			conn_idx_list_idx = self.excess_conn_list[random_idx]

			conn_idx = self.connection_idx_list[conn_idx_list_idx]
			
			# reset the link distribution to choose the correct length values in selection
			self.link_distribution[int(self.core_connection_options[conn_idx][2]-1)] += 1

			# Since all the cores(self.is_core_connected) are already set need not reset them
			# We are choosing connection after 64 cores have been connected
			# self.is_connection_available is not reset so that we do not choose the already choosen links
			new_conn_idx = self.get_next_random_connection()

			new_conn_info = self.core_connection_options[new_conn_idx]
			# Replace the selected connection with new index
			connection_idx_list[conn_idx_list_idx] = new_conn_idx
			
			# Set the flags for the new chosen connection
			self._set_flags(new_conn_idx)
			conn_idx_list_to_reset.append(new_conn_idx)

			# print("Conn idx ", conn_idx_list_idx, 
			# 	int(self.core_connection_options[conn_idx][2]-1),
			# 	int(self.core_connection_options[new_conn_idx][2]-1),
			# 	self.link_distribution)
			# sw_dist = [0,0,0,0]
			# for conn_idx_1 in connection_idx_list:
			# 	sw_dist[int(self.core_connection_options[conn_idx_1][2]-1)] += 1
			# print("Distribution", sw_dist)


		# Reset self.is_connection_available for future use
		for conn_idx in conn_idx_list_to_reset:
			self.is_connection_available[conn_idx] = 1

		# Generate the feature array
		feature_vector = U.generate_feature_list([connection_idx_list])[0]

		return feature_vector

	def generate_nearest_n_random_graph(self, feature_vector, N, num_switches):
		self.set_base_graph(feature_vector)
		feature_vector_list = []
		for i in range(N):
			feature_vector_list.append(self.generate_nearest_random_graph(num_switches))
		return np.asarray(feature_vector_list)


	def generate_random_sw_connection_idx_list(self, num_random_points):

		# generate random graphs satisfying the sw world criteria
		connection_idx_list_list = []
		connection_list_list = []
		for i in range(num_random_points):
			(connection_list, connection_idx_list) = \
				self.generate_valid_graph()
			connection_idx_list_list.append(connection_idx_list)
			connection_list_list.append(connection_list)

		return (connection_list_list, connection_idx_list_list)

	def generate_sw_feature(self, num_random_points):
		start_time = time.clock()
		(connection_list_list, connection_idx_list_list) \
		    = self.generate_random_sw_connection_idx_list(num_random_points)
		print("Time 1 ", time.clock() - start_time)
		start_time = time.clock()
		feature_numpy_array = \
		    U.generate_feature_list(connection_idx_list_list)
		print("Time 2 ", time.clock() - start_time)
		# print(feature_numpy_array.shape)
		return feature_numpy_array


# Function not used presently. Used initialy for testing graphs
def generate_node_connection_idx_list(num_random_points):
	connection_list = []
	# generate the basic vertical links list
	vertical_link_list = U.generate_vertical_links_list()
	connection_list.extend(vertical_link_list)

	# Subtract the link distribution for length 1
	link_distribution[0] -= len(vertical_link_list)
	print(link_distribution)


	# # 
	# rand_connection = \
	# 	get_connection_idx_list_based_on_link_distribution(link_distribution)
	# connection_list.extend(rand_connection)
	

	# Basic num links for mesh connection for NxM = N*(M-1) + M*(N-1)
	# So for a single layer 4x4 = it should be 2*3*4 = 24
	for y in range(4):
		for x in range(4):
			if(y != 0 and x != 0):
				connection_list.append(
					(U.get_node_index((0,0,0)),U.get_node_index((0,y,x))))
			if(y != 0 and x != 1):
				connection_list.append(
					(U.get_node_index((1,0,1)),U.get_node_index((1,y,x))))
			if(y != 0 and x != 2):
				connection_list.append(
					(U.get_node_index((2,0,2)),U.get_node_index((2,y,x))))
			if(y != 0 and x != 3):
				connection_list.append(
					(U.get_node_index((3,0,3)),U.get_node_index((3,y,x))))

	return connection_list

# Check for SW values or constraint???
# The init values just using the SW constraint should itself give a good value???
# Issue with model??
# - Replace RF model with GP

def validate_sw_distribution(feature_vector):
	core_connection_opt = generate_core_connection_options()
	connection_list_list = U.generate_conn_idx_list_list(feature_vector)
	ref_dist = [63, 21, 8, 4]
	for connection_list in connection_list_list:
		dist_hist = []
		for idx in range(4):
			dist_hist.append(0)
		for conn in connection_list:
			x,y,dist = core_connection_opt[conn]
			dist_hist[dist-1] += 1
		for idx in range(4):
			if(dist_hist[idx] != ref_dist[idx]):
				print("Error ", dist_hist)
		# print(dist_hist)
	print("Done validation all checks")
	return

# TSV - Compare TSV implementation
if __name__=="__main__":

	generate_core_connection_options()

	start_time = time.clock()
	feature_vector = LinkDistribution().generate_sw_feature(100)
	print("time taken ", time.clock() - start_time)
	validate_sw_distribution(feature_vector)
	# print(np.sort(np.sum(feature_vector, axis=0)))
	connection_list = U.generate_conn_idx_list_list(feature_vector)
	print(connection_list)
	print(feature_vector)
	core_connection = generate_connection_list(feature_vector[0])
	print(core_connection)
	core_connection_opt = generate_core_connection_options()
	dist_hist = {}
	dist_hist[0] = 0;dist_hist[1] = 0;
	dist_hist[2] = 0;dist_hist[3] = 0;
	for conn in connection_list[0]:
		x,y,dist = core_connection_opt[conn]
		dist_hist[dist-1] += 1
	print(dist_hist)
	# feature_list = LinkDistribution().generate_nearest_n_graph(feature_vector, 5, 1)
	# print(feature_list)
	# xor_val = np.logical_xor(feature_list, feature_vector)
	# print(np.where(xor_val))

	# print generate_link_distribution(C.ALPHA, C.K_AVG, C.NUM_CORES, 8, C.Z_LYR)
	# print generate_link_distribution(C.ALPHA, C.K_AVG, C.NUM_CORES, 16, C.Z_LYR)
	# print generate_link_distribution(C.ALPHA, C.K_AVG, C.NUM_CORES, 4, C.Z_LYR)

	# print get_distance_distribution(4, 4, 0, 0)
	# print get_distance_distribution(4, 4, 1, 1)

	
	# vertical_link_list = generate_vertical_links_list()
	# print (len(vertical_link_list))

	# # print(generate_core_distance_distribution())
	# generate_core_distance_distribution()

	# # connection_idx_list = generate_node_connection_idx_list(10)
	# # connection_idx_list = get_connection_idx_list(connection_list)
	# # print(connection_list)
	# # print(connection_idx_list)

	# # generate_random_sw_connection_idx_list(100)

	# feature = approximate_sw_feature(0.5*np.ones(480))