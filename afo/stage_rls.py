
from sklearn.ensemble import RandomForestRegressor
import utility_functions as U
import numpy as np
from bayesian_helpers import unique_rows
from sklearn.gaussian_process import GaussianProcessRegressor
from kernel import gp_kernel as gpk
from sklearn.gaussian_process.kernels import RBF


# Ordered should work better since the termination condition seems to be that for two successive iteration max value does not change
# Ordered should work better with Stage since stage requires more input points for modeling
# Use a different model instead of RF, using gaussian??
# 	- Gaussioan seems to model things better than RF (prediction values are closer to actual values)
#	- However STAGE does not use the previous iteration starting point as the actual value and almost always goes to random init
#	- The number of iterations for RLS is much higher for 
#   - Why is the RF model not modeling even after 100 iterations??


# List of parameters:
# Two greedy searches - num_switches, n_neighbors, 
# Two models - n_estimators, num_min_split
# Num iterations - STAGE algo, RLS
# 

class SearchAlgorithms(object):
	def __init__(self, feature_generator):
		self.fg = feature_generator
		self.num_iterations = self.fg.get_num_iterations()
		self.history = []
		return

	def greedy_search_aqu_fn(self, ac, reg, y_max, search_start_x):

		# print(" Start Vector ", x_base_rand)
		# Evaluate the first random feature vector
		acq_max = ac(search_start_x, reg, y_max)
		x_max = search_start_x[0]
		# print("Random Max ", acq_max)

		# Store the list of X features to model STAGE approx
		x_feature_list = [x_max]
		num_iter = 0
		ordered_loop_count = 0

		# Iterate until local maximum is reached
		while(True):

			prev_x_max = x_max
			prev_acq_max = acq_max

			for i in range(self.num_iterations):

				# print("x_max ", x_max.shape)
				x_next_n_neighbors = \
					self.fg.generate_successor_graph(x_max)
				if(len(x_next_n_neighbors) == 0):
					continue
				# Remove this later
				# ld16.validate_sw_distribution(x_next_n_neighbors)

				acq_next_n_neighbors = ac(x_next_n_neighbors, reg, y_max)
				max_id = np.argmax(acq_next_n_neighbors)

				# print("Max val ", i, acq_next_n_neighbors[max_id], acq_max)
				if(acq_next_n_neighbors[max_id] > acq_max):
					x_max = x_next_n_neighbors[max_id]
					acq_max = acq_next_n_neighbors[max_id]
					x_feature_list.append(np.copy(x_max))
				# Store them so that finally we can create a numpy array
				# print(x_next_n_neighbors.shape)
				# for i in range(x_next_n_neighbors.shape[0]):
				# 	x_feature_list.append(x_next_n_neighbors[i])

				# print("GS ", num_iter, ordered_loop_count, max_id, acq_max, 
				# 	prev_acq_max, np.sum(unique_rows(x_next_n_neighbors)))


			num_iter = num_iter + 1

			if(prev_acq_max < acq_max):
				ordered_loop_count = 0
			elif(self.num_iterations == U.GET_MAX_DISTANCE()):
				if(ordered_loop_count >= 1):
					break
				else:
					ordered_loop_count = ordered_loop_count + 1
			else:
				break;

		print("GS ", num_iter, len(x_feature_list), acq_max)
		return acq_max, x_max, x_feature_list

	def gready_search_stage_reg(self, x_feature_list, 
		y_feature_list, search_start_x):

		# Generate X and Y for fit regresson model
		x_feature_list = np.asarray(x_feature_list)
		y_feature_list = np.asarray(y_feature_list)*100

		# print(x_feature_list.shape, y_feature_list, search_start_x.shape)


		# Create the regressor
		# n_estimators = 15
		# min_samples_split = 2
		# stage_reg_learner = RandomForestRegressor(
		# 	n_estimators=n_estimators)
			# ,min_samples_split=min_samples_split

		# kernel = gk.GraphKernel(h=5)
		# kernel = gpk.Jaccard()
		kernel = RBF()
		n_restarts_optimizer = 25
		stage_reg_learner = GaussianProcessRegressor(kernel=kernel,
	                       n_restarts_optimizer=n_restarts_optimizer)
		
		# Find unique rows of X to avoid reg from breaking
		ur = unique_rows(x_feature_list)
		# print(len(ur), np.sum(ur), ur, x_feature_list.shape, y_feature_list, y_feature_list.shape)
		stage_reg_learner.fit(x_feature_list[ur], \
			y_feature_list[ur])

		# Validating predicted values
		# itr = 0
		# for feature_vector in x_feature_list:
		# 	print("Predict value ",stage_reg_learner.predict(feature_vector.reshape(1,-1)),
		# 		y_feature_list[itr])
		# 	itr+=1

		# Evaluate the first random feature vector
		acq_max = stage_reg_learner.predict(search_start_x.reshape(1,-1))
		x_max = search_start_x
		# print("acq max ", acq_max, np.sum(search_start_x))

		# Store the list of X features to model STAGE approx
		# x_feature_list = [x_max]
		num_iter = 0
		ordered_loop_count = 0


		# Iterate until local maximum is reached
		while(True):

			prev_x_max = x_max
			prev_acq_max = acq_max

			for i in range(self.num_iterations):

				if(self.num_iterations == U.GET_MAX_DISTANCE()):
					x_next_n_neighbors = \
						self.fg.generate_successor_graph(x_max)
				else:
					x_next_n_neighbors = \
						self.fg.generate_successor_graph(x_max)
				if(len(x_next_n_neighbors) == 0):
					continue
				# Remove this later
				# ld.validate_sw_distribution(x_next_n_neighbors)
				
				# print("Num design points ", i, x_next_n_neighbors.shape)
				acq_next_n_neighbors = stage_reg_learner.predict(x_next_n_neighbors)
				max_id = np.argmax(acq_next_n_neighbors)
				if(acq_next_n_neighbors[max_id] > acq_max):
					x_max = x_next_n_neighbors[max_id]
					acq_max = acq_next_n_neighbors[max_id]
				# Store them so that finally we can create a numpy array
				# print(x_next_n_neighbors.shape)
				# for i in range(x_next_n_neighbors.shape[0]):
				# 	x_feature_list.append(x_next_n_neighbors[i])

				# print("GS ", num_iter, ordered_loop_count, max_id, acq_max, 
				# 	prev_acq_max, np.sum(unique_rows(x_next_n_neighbors)))


			num_iter = num_iter + 1

			if(prev_acq_max < acq_max):
				ordered_loop_count = 0
			elif(self.num_iterations == U.GET_MAX_DISTANCE()):
				if(ordered_loop_count >= 1):
					break
				else:
					ordered_loop_count = ordered_loop_count + 1
			else:
				break;
		print("GS STG ", num_iter, acq_max)

		return x_max, acq_max

	def smac_srch(self, ac, reg, y_max, X):
		# Do local search using the already suggested past history
		local_aqu_max = -9999999
		local_x_max = []
		# Numer of local searches
		if(len(self.history) == 0):
			history_sort_max = self.fg.generate_n_random_feature(10)
		else:
			num_local_srch = 10
			if num_local_srch > len(self.history):
				num_local_srch = len(self.history)
			history_aqu_max = ac(self.history, reg, y_max)
			history_idx = np.argsort(history_aqu_max)[::-1][:num_local_srch]
			np_history = np.array(self.history)
			history_sort_max = np_history[history_idx]
			# print(history_aqu_max, history_idx)
			# history_sort_max = history_sort_max[:num_local_srch]
		
		for search_start_x in history_sort_max:
			# print("Search start ", search_start_x)
			aqu_max, x_max, _ = \
				self.greedy_search_aqu_fn(ac, reg, 
					y_max, [search_start_x])
			if(local_aqu_max < aqu_max):
				local_aqu_max = aqu_max
				local_x_max = x_max

		# Choose 10000 random designs
		rand_designs = \
			self.fg.generate_n_random_feature(10000)
		# print("Generated random designs ", rand_designs.shape)
		rand_designs_value = ac(rand_designs, reg, y_max)
		max_id = np.argmax(rand_designs_value)
		# if(rand_designs_value[max_id] > acq_max):
		# 	x_max = rand_designs[max_id]
		# 	acq_max = rand_designs_value[max_id]
		# print("GS RND ", rand_designs_value[max_id])
		rand_designs = rand_designs[max_id]

		# Choose the best of the both
		if(rand_designs_value[max_id] > local_aqu_max):
			global_x_max = rand_designs
			global_acq_max = rand_designs_value[max_id]
		else:
			global_x_max = local_x_max
			global_acq_max = local_aqu_max

		# Append the max value
		self.history.append(global_x_max)
		print("SMAC ", global_x_max, global_acq_max)
		return global_x_max, global_acq_max

	def random_local_search(self, ac, reg, y_max):
		N = 20#50

		global_x_max = []
		global_acq_max = -99999999
		for i in range(N):

			# for each iteration, choose a random starting point
			search_start_x = \
				self.fg.generate_n_random_feature(1)
			local_aqu_max, local_x_max, _ = \
				self.greedy_search_aqu_fn(ac, reg, 
					y_max, search_start_x)
			if(local_aqu_max > global_acq_max):
				global_x_max = local_x_max
				global_acq_max = local_aqu_max	
				#print("global i",global_acq_max)
		return global_x_max, global_acq_max

	def stage_algorithm(self, ac, reg, y_max, X):
		N = 10#20#30

		global_x_max = []
		global_acq_max = -99999999
		global_x_feature_list = []
		global_y_feature_list = []

		# for each iteration, choose a random starting point
		search_start_x = \
			self.fg.generate_n_random_feature(1)

		rand_designs = \
			self.fg.generate_n_random_feature(10)
		# print("Generated random designs ", rand_designs.shape)
		rand_designs_value = ac(rand_designs, reg, y_max)
		max_id = np.argmax(rand_designs_value)
		# if(rand_designs_value[max_id] > acq_max):
		# 	x_max = rand_designs[max_id]
		# 	acq_max = rand_designs_value[max_id]
		print("GS RND ", rand_designs_value[max_id])
		rand_designs = rand_designs[max_id]
		search_start_x = np.asarray([rand_designs])

		num_random = 0
		for i in range(N):
			
			local_aqu_max, local_x_max, x_feature_list = \
				self.greedy_search_aqu_fn(ac, reg, y_max, 
					search_start_x)
			if(local_aqu_max > global_acq_max and
			np.any((np.absolute(X - local_x_max)).sum(axis=1) != 0)):
				global_x_max = local_x_max
				global_acq_max = local_aqu_max
				#print("global i",global_acq_max)

			# Create the feature vector and max value
			for x_feature in x_feature_list:
				global_x_feature_list.append(x_feature)
				global_y_feature_list.append(local_aqu_max)
			# Do a STAGE search for the next starting point
			# search_start_x, search_start_aqu_max = \
			# 	self.gready_search_stage_reg(global_x_feature_list,
			# 		global_y_feature_list, local_x_max)

			# # Calculate the AF for the search_start_x
			# search_start_aqu_max = ac([search_start_x], reg, y_max)[0]
			# print("GS STG VAL ", search_start_aqu_max, local_aqu_max, global_acq_max)
			
			# if(search_start_aqu_max <= global_acq_max+0.001):
			if(True):
				# Randomly choose certain set of design points and find the min value for that
				rand_designs = \
					self.fg.generate_n_random_feature(10)
				# print("Generated random designs ", rand_designs.shape)
				rand_designs_value = ac(rand_designs, reg, y_max)
				max_id = np.argmax(rand_designs_value)
				# if(rand_designs_value[max_id] > acq_max):
				# 	x_max = rand_designs[max_id]
				# 	acq_max = rand_designs_value[max_id]
				print("GS RND ", rand_designs_value[max_id])
				rand_designs = rand_designs[max_id]
				search_start_x = np.asarray([rand_designs])
			
			# if(search_start_aqu_max <= local_aqu_max):
			# 	search_start_x = ld.generate_n_random_feature(1)
				
				# print("Random ", search_start_aqu_max, local_aqu_max,
				# 	ac(search_start_x, reg, y_max)[0], global_acq_max)
				num_random += 1
			else:
				search_start_x = np.asarray([search_start_x])
				# print("Not Random ", search_start_aqu_max, local_aqu_max,
				# 	global_acq_max)

		# print("Num random picks ", num_random)
		return global_x_max, global_acq_max

if __name__ == "__main__":
	print("No functions")

# Possible improvements:
# - Make the RF size to 15
# - Make RLS to 50 and STAGE to 30
# - Run for 500 iterations
# - EDP ratio calculated could be better than the actual ratios
