import numpy as np
import bottleneck as bn

# List of thinkgs to do
# - Implement the feature generators
# - Function approximation for roll out values
#	- Read the reference shared by Jana
# - Modify the n max value

# Things to experiment:
#	- Change the beam size, num_roll_outs
#	- Try merging GP search with actual search like in bamlogo
#	- Modify mean to UCT
#	- Modify mean to mean + std deviation [x]
#	- run SMAC directly and not from the code. [x]
#		- Since now the application is not so complex
#	- USe the MF Bamlogo idea to reduce the search region
#	- tsv: make num spare tsv per input as 4
#	- Ordering of the feature expansion could be randomised 
#	
#	- Try changing the num_roll_outs and check performance
#	- Try thresholding the total evaluations
#	- Try changing the k parameter of std deviation in mean+std_dev
#	- Instead of mean+variance use Max
#	- How to vary K across the depth of the tree
#	- REduce the number of nodes open at any time to reduce the memory

# - ucb/ei not used
# - function approx does not make sense
# - tsv simulator runs using wine
# - noc is not a good candidate
# - ucb using current maximum to ignore nodes
# - why is ei not working??
# - 

# - Use every evaluation to decide a lower threshold and make that node to be not evaluated
# - Use the variance and tree depth to decide unexplored regions
# - Unexplored regions get high priority (since it is not below lower threshold)
# 	- Probably the Num explored /Num passed tthrough would be agood measure???
# - carry fwd the lower thresold parts of the tree
# Tree pruning algorithm: total budget

class StructPred(object):
	def __init__(self, feature_generator):
		self.fg = feature_generator
		self.num_iterations = self.fg.get_num_iterations()
		self.history = []
		self.beam_size = 5
		self.num_roll_outs =  100 #100, 50, 150, 1000, 500, 250
		self.K = 1 #1 2 0.5 0 1.5
		self.prev_tree = []
		return

	def roll_out_value(self, node_info, ac, reg, y_max):
		rand_feature_vector = self.fg.expand_node_randomly(node_info, self.num_roll_outs)
		# print(rand_feature_vector.shape)
		acq_rand_feature_vect = ac(rand_feature_vector, reg, y_max)
		mean_value = np.mean(acq_rand_feature_vect)
		std_dev = np.std(acq_rand_feature_vect)

		max_feature_vector = rand_feature_vector[acq_rand_feature_vect.argmax()]
		max_value = acq_rand_feature_vect[acq_rand_feature_vect.argmax()]
		# print(rand_feature_vector.shape, mean_value)

		return mean_value+self.K*std_dev, max_feature_vector, max_value

	def maximize_beam(self, ac, reg, y_max):
		# 
		root_node_info = self.fg.get_root_node()
		beam_node_list = [(-1*float("inf"), root_node_info)]
		end_state_reached = False
		while(not end_state_reached):
			# Iterate through all the beam nodes and pick the next largest
			for beam_node in beam_node_list:
				beam_value_list = []
				beam_node_info_list = []
				node_value, beam_node_info = beam_node
				# Get the next set of expanded nodes
				expanded_node_list = self.fg.expand_node(beam_node_info)

				if(len(expanded_node_list) == 0):
					end_state_reached = True
					break

				for node_info in expanded_node_list:
					# Calculate the value for each node using random rollouts
					node_value, _, _ = \
						self.roll_out_value(node_info, ac, reg, y_max)
					# print(node_value, node_info)
					beam_value_list.append(node_value)
					beam_node_info_list.append(node_info)

			if(end_state_reached):
				print(beam_node_list)
			else:
				# Find the values of the next set of list
				beam_idx_list = bn.argpartition(-1*np.array(beam_value_list), kth=self.beam_size)[:self.beam_size]
				# print(beam_value_list)
				# print(beam_idx_list)
				# beam_node_info_list = np.array(beam_node_info_list)
				beam_node_list = []
				for i in range(self.beam_size):
					beam_node_list.append(
						(beam_value_list[beam_idx_list[i]],
						beam_node_info_list[beam_idx_list[i]]))
				# print(beam_node_list)

		# Find the max value
		max_node_info = []
		max_value = -1*float("inf")
		for i in range(self.beam_size):
			node_value, node_info = beam_node_list[i]
			if(node_value>max_value):
				max_value = node_value
				max_node_info = node_info
		return self.fg.get_node(max_node_info), max_value


	def maximize_mcts(self, ac, reg, y_max):
		# 
		root_node_info = self.fg.get_root_node()
		node_info_list = [root_node_info]
		node_value_list = [-1*float("inf")]

		max_node_info = node_info_list[0]
		max_value = node_value_list[0]

		rnd_max_feature_vector = []
		rnd_max_value = -1*float("inf")

		end_state_reached = False
		num_nodes_expanded = 0
		while(not end_state_reached):
			expanded_node_list = self.fg.expand_node(max_node_info)
			if(len(expanded_node_list) == 0):
				end_state_reached = True
				break

			if(num_nodes_expanded == 1000):
				max_value = node_value_list[max_idx]
				max_feature_vector = rnd_max_feature_vector
				break

			for node_info in expanded_node_list:
				# Calculate the value for each node using random rollouts
				node_value, feature_vector, value = \
					self.roll_out_value(node_info, ac, reg, y_max)

				if(rnd_max_value < value):
					rnd_max_value = value
					rnd_max_feature_vector = feature_vector

				# print(node_value, node_info)
				node_value_list.append(node_value)
				node_info_list.append(node_info)

			max_idx = np.array(node_value_list).argmax()
			max_node_info = node_info_list[max_idx]
			max_value = node_value_list[max_idx]
			max_feature_vector = self.fg.get_node(max_node_info)

			node_value_list[max_idx] = -1*float("inf")
			num_nodes_expanded += 1

		self.validate_update(node_value_list, node_info_list,
			max_feature_vector, ac, reg, y_max)

		# self.prev_tree = (node_value_list, node_info_list, max_feature_vector)


		print("Total number of nodes expanded ", num_nodes_expanded)
		return max_feature_vector, max_value

	def validate_update(self, node_value_list, 
		node_info_list, max_feature_vector,
		ac, reg, y_max):
		if(len(self.prev_tree)):
			prev_nvl, prev_nil, prev_mfv = self.prev_tree
			prev_true_value = ac(prev_mfv, reg, y_max)
			i = 0
			for prev_nv, prev_ni  in zip(prev_nvl, prev_nil):
				if(prev_nv != -1*float("inf")):
					correl = self.fg.get_correlation(prev_ni, prev_mfv)
					new_value = \
						correl*prev_true_value + (1-correl)*prev_nv
					prev_nvl[i] = new_value
					new_nv_1, fv, value = self.roll_out_value(prev_ni, ac, reg, y_max)
					new_nv_2, fv, value = self.roll_out_value(prev_ni, ac, reg, y_max)

					print("Comparing ", new_nv_1, new_nv_2, prev_nvl[i],
						prev_true_value, prev_nv, correl)
				i += 1


			# print(node_info_list)
			# print(prev_nil)
			# print(node_value_list)
			# print(prev_nvl)


		return

