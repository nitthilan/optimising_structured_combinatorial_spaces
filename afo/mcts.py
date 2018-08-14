
import numpy as np
from collections import defaultdict
from sets import Set
from random import randint

import constants_tsv as CT

#file:///Users/kannappanjayakodinitthilan/Documents/myfolder/project_devan/aws_workspace/source/expensive_experiments/optimizing-combinatorial-structured-spaces/references/design_space/1608.06972_Design_space_exploration.pdf
#(M*N-N^2)
# https://mail-attachment.googleusercontent.com/attachment/u/0/?ui=2&ik=ff424d53fe&view=att&th=164289232b273060&attid=0.1&disp=inline&realattid=f_jiq995i20&safe=1&zw&saddbat=ANGjdJ9gDJ0HwA6fSc7t3Tdzu9Q6-X2hbevpbGLDzdtx7JWgFSJTXfnWKLuvKZLzqsS3P6qFfq3dOzbWnhLCJaYAEwvIkNboLq-4If--nFlbngbvdZDaA12Ax-XaFPwFSTKy5eOxjwXbr07GU64faYg1g5taWFSY5EUMZKFChF4jnFsWU051Wzp6Nbng_R80XD-C6ZKXBKQ5Iwuo68oqceebeBbj3z68XJ6I5NH_4z0nAEw9_DiE4e5lxfcQ3SIGfFjHzL7TSfEOa56PCwYEB87QEY9xobHmbJuf5xQTR9zIAMP_6Uha1jp_S5m5PIhIuCt6il_4bzTJlid9hTF37tb6xF8dg0nSrpfrF9A7tFwxoTnD6fm1Fo6gaH-yXsd4_uK8XaSKZMfusf0rMFkCFBUwE5kq_1lci8YVh2939IxxYyTNhRaYCEriVT3AUH9ACKh-zUADOfSWoXwEUH7XyCOo_QAqQc5ihMPINcf-Oo2cyy9F4syG48Qc9iU3gUzouEH5sUCam2zd57AUpHpQRzSYcmHwudGELnirfIU3cpw23EmjruxjqQit8W9D0Xa4pJi5982IeRCalLiZDBxonAlLNdTsb6Oo8uhd-PwA-w-qcIvNqz97FexHbVFWJ5w
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.172.9450&rep=rep1&type=pdf

#RESTRICT_SEARCH = CT.NUM_TSVS
#RESTRICT_SEARCH = CT.DESIGN_ORDERING[0]+CT.DESIGN_ORDERING[1]+CT.DESIGN_ORDERING[2]+CT.DESIGN_ORDERING[3]
RESTRICT_SEARCH = CT.DESIGN_ORDERING[0]+CT.DESIGN_ORDERING[1]

# When search all the (RESTRICT_SEARCH) 384 keep this value to 16+32+4*16+4*32
# When searching for 16+32+4*16+4*32 keep this value to 16+32
# When searching for 16+32+4 keep this value to zero
#DEPTH_BASED_BRANCH_RESTRICT = CT.DESIGN_ORDERING[0]+CT.DESIGN_ORDERING[1]+CT.DESIGN_ORDERING[2]+CT.DESIGN_ORDERING[3]
#DEPTH_BASED_BRANCH_RESTRICT = CT.DESIGN_ORDERING[0]+CT.DESIGN_ORDERING[1]
DEPTH_BASED_BRANCH_RESTRICT = 0


# Priority here is 10:2:1 - Center:Edge:corner, 6:1 Middle:Top/Bottom
def designScore(design, budget):
	total = 0

	middleTSVindexes = list()
	middleTSVindexes = [i for i in xrange(4,len(design),8)]
	secondPriorityIndexes = list()
	secondPriorityIndexes = [j for j in xrange(1, len(design), 2)]

	for i in range(len(design)):
		if i in middleTSVindexes:
			total = total + (10 * design[i])
		elif i in secondPriorityIndexes:
			total = total + (2 * design[i])
		else:
			total = total + design[i]
	secTotal = 0


	twoDimDesign = [design[i:i+8] for i in range(0, len(design), 8)]
	for i in range(len(twoDimDesign)):
		if i in [j for j in range(16,32)]:
			secTotal = secTotal + sum(twoDimDesign[i])
			secTotal = secTotal + sum(twoDimDesign[i])
			secTotal = secTotal + sum(twoDimDesign[i])
			secTotal = secTotal + sum(twoDimDesign[i])
			secTotal = secTotal + sum(twoDimDesign[i])
			secTotal = secTotal + sum(twoDimDesign[i])
		else:
			secTotal = secTotal + sum(twoDimDesign[i])


	return total + secTotal



# Random Probability distribution:
#	- Center:Edge:corner = 12:2:1, Middle:Top/Bottom = 2:1
#	- [0-16]-24, [16-48]-12, [48-]
#	- 16*24 + 32*12 + 16*4*4 + 32*4*2 + 16*3*2 + 32*3*1	= 1472
def fill_based_on_priority_old(random_design, total_alloc):
	# Priority of each index
	ctr_mid = 16*24
	ctr_topbot = ctr_mid + 32*12
	edg_mid = ctr_topbot + 16*4*4
	edg_topbot = edg_mid + 32*4*2
	cor_mid = edg_topbot + 16*3*2
	cor_topbot = cor_mid + 32*3*1
	# print(ctr_mid, ctr_topbot, edg_mid,
	# 	edg_topbot, cor_mid, cor_topbot)	

	while(total_alloc>0):
		#
		rand_idx = randint(0,1472-1)
		if(rand_idx < ctr_mid):
			dsg_idx = int(rand_idx/24)
		elif(rand_idx < ctr_topbot):
			rand_idx -= ctr_mid
			dsg_idx = 16+int(rand_idx/12)
		elif(rand_idx < edg_mid):
			rand_idx -= ctr_topbot
			dsg_idx = 48+int(rand_idx/4)
		elif(rand_idx < edg_topbot):
			rand_idx -= edg_mid
			dsg_idx = 48+64+int(rand_idx/2)
		elif(rand_idx < cor_mid):
			rand_idx -= edg_topbot
			dsg_idx = 48+64+128+int(rand_idx/2)
		elif(rand_idx < cor_topbot):
			rand_idx -= cor_mid
			dsg_idx = 48+64+128+48+int(rand_idx)

		random_design[dsg_idx] += 1
		total_alloc -= 1

	return

def fill_based_on_priority(random_design, start_idx, num_spare_tsv):
	while(num_spare_tsv > 0):
		dsg_idx, _, _ = RAND_TSV_IDX_GEN.rand_idx_based_on_priority()
		if(dsg_idx > start_idx):
			random_design[dsg_idx] += 1
			num_spare_tsv -= 1
	# print(start_idx, num_spare_tsv, random_design)
	return

# 2,3,4,4,5,8 for 9, 19, 28, 38, 57, 96
tsv_budget_thresholds_list = [9, 19, 28, 38, 57, 3000]
max_state_to_explore_list = [2, 3, 4, 4, 5, 8]
def get_next_node(node_state, idx, num_leaves_parsed,
	max_num_spare_tsv):
	# Calculate the already sealed allocation of links from the parent node
	curr_alloc_tsv = np.sum(node_state[:idx])
	if(np.sum(node_state[idx:])):
		print("Error in parent state. Cannot have allocation in this range ", \
			node_state, idx)

	# Based on TSV allocation
	max_state_to_explore = 0
	for idx, thres in enumerate(tsv_budget_thresholds_list):
		if(max_num_spare_tsv <= thres+1):
			# one added to inclue the threshold value into the search space
			max_state_to_explore = \
				max_state_to_explore_list[idx] + 1
			break
	# print("max_state_to_explore ", max_state_to_explore,
	# 	max_num_spare_tsv)
	#
	num_state_to_explore = max_num_spare_tsv - curr_alloc_tsv
	if(num_state_to_explore > max_state_to_explore):
		num_state_to_explore = max_state_to_explore
	# else:
	# 	print("num_state_to_explore ", num_state_to_explore, 
	# 		max_num_spare_tsv, curr_alloc_tsv)

	#
	if(num_leaves_parsed < num_state_to_explore):
		is_all_states_processed = False
		return is_all_states_processed, num_leaves_parsed
	else:
		return True, -1

def init_node(design_state, parent_node, node_depth, node_idx):
	node = {
		"node_idx": node_idx,
		"design_state": design_state, # the design state for this node. It is a 48x8 vector with possible design values
		"num_visited":0, # num ber of times this node is visited
		"total_val":0, # Accumulated value for calculating mean
		"num_leaves_parsed":0, # Assuming this to be the parent, the number of leave nodes processeds
		"parent_node": parent_node, # Parent node for backpropogation
		"leaves_list": [], # List of leaves from this node
		"node_depth": node_depth, # The node depth from the root

		# While updating this value conditions to take care:
		# If current leaf idx is same as min_leaf idx: two conditions
		#	- if value goes even higher nothing to change
		# 	- if value goes lower then search for the next best in the leaves and update it
		# If the current leaf idx is not the same as min_leaf idx two conditions
		#	- if value goes higher update the value with the current_leaf_idx
		#	- if the value is lower then nothing changes
		# Store a array instead of single value since there could be multiple simillar values
		# "max_leaf_idx_list": [],
	}
	return node

# Parse all the leaves and return the max values in a list
def get_max_leaf_value(current_node, num_iteration):
	max_q_val = -9999999
	max_leaf_idx_list = []
	for leaf in current_node["leaves_list"]:
		leaf_q_val = calc_q_value(leaf, num_iteration)
		# print(leaf_q_val, leaf["total_val"])
		if(leaf_q_val > max_q_val):
			max_q_val = leaf_q_val
			max_leaf_idx_list = [leaf["node_idx"]]
		elif(leaf_q_val == max_q_val):
			max_leaf_idx_list.append(leaf["node_idx"])
	return max_leaf_idx_list


# Using the given state_statistics dictionary
# parses the tree to get the next node to be processed
# Probably have another list which stores the min node index
def get_next_node_using_tree_policy(root_node, num_iterations,
	max_num_spare_tsv, max_tsv_to_explore):
	node_depth = 0
	current_node = root_node

	while(1):
		tsv_pos_idx = current_node["node_depth"]

		# Stop exploring once we hit the max tsv to explore value
		if(tsv_pos_idx >= max_tsv_to_explore):
			return current_node, True

		# print (current_node["node_depth"],
		# 	np.sum(current_node["design_state"][:tsv_pos_idx]))

		is_all_states_processed, tsv_alloc = \
			get_next_node(current_node["design_state"], 
				tsv_pos_idx, current_node["num_leaves_parsed"],
				max_num_spare_tsv)
		
		# print(is_all_states_processed, tsv_alloc)
		if(is_all_states_processed):
			# Calculate the list of max_leaf_idx_list 
			max_leaf_idx_list = get_max_leaf_value(current_node, num_iterations)
			
			# KJN
			# print("Min idx list ", current_node["node_idx"], 
			# 	current_node["node_depth"], max_leaf_idx_list)
			
			# If all the states are processed then find the next leaf with the maximum design
			rand_idx = randint(0, len(max_leaf_idx_list)-1)
			next_node = current_node["leaves_list"][max_leaf_idx_list[rand_idx]]
			current_node = next_node
		else:

			# create a new node and added it to the parrent list
			design_state = np.copy(current_node["design_state"])
			design_state[current_node["node_depth"]] = tsv_alloc
			new_node_depth = current_node["node_depth"]+1
			new_node_idx = current_node["num_leaves_parsed"]
			leaf_node = init_node(design_state, current_node, 
				new_node_depth, new_node_idx)

			# Update current node statistics
			current_node["num_leaves_parsed"] += 1
			current_node["leaves_list"].append(leaf_node)
			# return the leaf node and tsv_pos_idx for the rest of the design
			return leaf_node, False

	# return leaf_node, is_terminal

# Using the current expansion node as input, assign random values to the 
# rest of the design. Using this random design evaluate the q value
# the random desing is filled by iteratively chosing a value between non zero
# indexes and 1-(MAX_NUM_SPARE_TSV - curr_sum) until either of then goes to zero
def random_policy_rollout(expansion_node, ac, reg, y_max,
	max_num_spare_tsv):
	random_design = np.copy(expansion_node["design_state"])
	curr_alloc_tsv = np.sum(random_design)
	start_idx = expansion_node["node_depth"]

	if(curr_alloc_tsv > max_num_spare_tsv):
		print("Error in allocation of tsv ", curr_alloc_tsv, max_num_spare_tsv)
		exit(0)

	# Fill the rest of the design based on the priority of allocation
	# fill_based_on_priority(random_design, max_num_spare_tsv - curr_alloc_tsv)
	fill_based_on_priority(random_design, start_idx,
		max_num_spare_tsv - curr_alloc_tsv)

	
	if(np.sum(random_design) != max_num_spare_tsv):
		print("Error in random policy rollout ", random_design)

	q_value = ac(random_design.reshape(1, -1), reg, y_max)

	return q_value, random_design


def calc_q_value(current_node, total_num_iteration):
	# Assuming Cp = 1/sqrt(2)
	# UCT = mean + 2 * Cp * sqrt(2*ln(total_iteration)/num_visited)
	mean = 10*current_node["total_val"]/current_node["num_visited"]
	# Should not be scaled by -1. Since we are trying to maximise
	# The exploration should increase with num _itrations
	exploration = 1.0 * np.sqrt(np.log(total_num_iteration)/current_node["num_visited"])
	# print("mean exploration ", mean, exploration)
	# print(current_node["total_val"], current_node["num_visited"], total_num_iteration)

	return mean+exploration

# Start from the expansion node and update q_value to the root node
def back_propagate_q_value(expansion_node, q_value, total_num_iteration):
	current_node = expansion_node
	while(current_node != None):

		current_node["num_visited"] += 1
		current_node["total_val"] += q_value

		parent_node = current_node["parent_node"]
		# print(parent_node, current_node)
		if(parent_node == None):
			return
		
		current_node = current_node["parent_node"]
	return

def monte_carlo_tree_search(ac, reg, y_max,
	max_num_iterations, # Num of tries after which the algorithms has to terminate
	max_num_spare_tsv # the max number of spare TSV allocated
	):
	# initialise the root node
	root_node = init_node(np.zeros(CT.NUM_TSVS, dtype=np.int), 
		None, 0, 0)

	# Store init conditions
	num_iterations = 0
	max_design = np.zeros(CT.NUM_TSVS, dtype=np.int)
	max_design_value = -99999999

	EARLY_EXIT_ITER_THRESHOLD = 0.3*max_num_iterations
	early_exit_iter = 0


	# Start Iterations
	for num_iterations in range(max_num_iterations):
		# Selection and Expansion: 
		# Evaluate the node with the best Q value bases on Tree Policy
		# Ideally this node would be a terminal node - which case quit ???
		# Else would have some of the child nodes not evaluated. 
		# So get the set of child nodes to be evaluated or rather choose the next child node to be evaluated
		max_tsv_to_explore = \
			CT.DESIGN_ORDERING[0]+CT.DESIGN_ORDERING[1]
		expansion_node, is_terminal = \
			get_next_node_using_tree_policy(root_node, num_iterations, 
				max_num_spare_tsv, max_tsv_to_explore)
		
		# Simulation:
		# Do a random rollout using this child node.
		q_value, random_design = random_policy_rollout(expansion_node, 
			ac, reg, y_max, max_num_spare_tsv)

		# Probably store the min value obtained using random rollouts in a separate variable
		# along with the node which produced that value
		if(q_value > max_design_value):
			max_design_value = q_value
			max_design = random_design
			# If you find a better design reset early exit
			early_exit_iter = 0


		# Break only when terminal node is the minimum node else iterate back and explore
		# "is_terminal or" need not be used since we are just exploring only part of the search space
		if( early_exit_iter > EARLY_EXIT_ITER_THRESHOLD):
			print("Early Exit or Terminal node ", is_terminal, early_exit_iter)
			print(num_iterations, expansion_node["node_depth"], 
				expansion_node["node_idx"], 
				expansion_node["num_visited"],
				expansion_node["total_val"])
			print("Max value")
			print(max_design_value) #, max_design)
			# Probably store the min design obtained till now
			break
		

		# Backpropagation:
		# Add this node to the tree policy with values obtained using simulation
		# Using the value update the parents with the new Q value
		# tp_selection_node.leaves_list.append(expansion_node)
		# expansion_node.parent_node = tp_selection_node

		back_propagate_q_value(expansion_node, q_value, num_iterations)

		early_exit_iter += 1

	# There could be probability the min value obtained from terminal node need not 
	# be the better than the min value obtained by random roll out
	# Try storing this information and just return the absolute best if it terminates

	print("All iterations done ", max_design_value)
	return max_design, max_design_value

# List of queries:
# - Branching factor could be reduced by utilising the 
#	- Probabilities of center>edge>corner
#	- Node getting all B values is less likely so could use some kind of probability distribution
# - Using mean Q value for each node to expolit the tree search policy
#	- Can we use min instead of mean???
# - currently assume a value of 1/sqrt(2) 
#	- shown by Kocsis and Szepesvari [120] to satisfy the Hoeffding ineqality with rewards in the range)
# - What is the probable range of B? 20% of TSV 10,100??,1000x

# Some facts:
# - Thus the total number of initial states possible is upper bounded by 48 x 8 x (B)
#	- Its not B+1 because if zero is considered as a state until B is available no node will be left with zero value
#	- Its better to use B rather than not using it
# 

# List of options to try are:
#	- Reducing the tree branching factor using predetermined solution. Branch only the center cores, then edge then corners
# 	- B
# 	- Tweeking the exploration constant

# Questions:
#	- Random assignment of links seems to assign lot of links to a single node in one shot.
# Options to try out:
#	- Using the EI function instead of the exploration term
# 	- Try moving the terminal nodes position to lower tree levels to prevent earlier termination due to exploration
# 	- Can the scaling for TSV made x/x_mean instead of (x-xmean)/xmean

# References:
# - http://www.cameronius.com/cv/mcts-survey-master.pdf - Pretty good but not on the first read
# - https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/ - Good for a first read
# - Alan Fern ppt: ok read: file:///Users/kannappanjayakodinitthilan/Documents/myfolder/project_devan/aws_workspace/reference/jana_reference/mcts/icaps2010_fern_mcpbprp_01.pdf
# - http://delivery.acm.org/10.1145/3140000/3130701/p1366-das.pdf?ip=69.166.46.150&id=3130701&acc=ACTIVE%20SERVICE&key=B63ACEF81C6334F5%2E3B1D11B7501B70D8%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&CFID=976896827&CFTOKEN=18431532&__acm__=1503705200_1b7cf10bb151a85ab3673e9d4f9bedd6
# - file:///Users/kannappanjayakodinitthilan/Documents/myfolder/project_devan/aws_workspace/source/expensive_experiments/optimizing-combinatorial-structured-spaces/references/design_space/1608.06972_Design_space_exploration.pdf
# Not read but to be read
# - https://en.wikipedia.org/wiki/Minimax
# - Discussion of the exploitation term and how it is derived
# 	- http://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
# 	- https://www.lri.fr/~sebag/Examens_2008/UCT_ecml06.pdf

# Queries:
# - Mean lies between -1 to Zero and exploration is greater than zero. Should it be fine???
# 	print("mean exploration ", mean, exploration)
# 	print(current_node["total_val"], current_node["num_visited"], total_num_iteration)
# - Try changing the scale factor to consider the range of mean -1 to zero
# 	- or add 1 to the mean to change it between 0-1
# - No information on how much center values consume more that the rest
#	- file:///Users/kannappanjayakodinitthilan/Documents/myfolder/project_devan/aws_workspace/source/expensive_experiments/optimizing-combinatorial-structured-spaces/references/design_space/1608.06972_Design_space_exploration.pdf
#	- The google drive values just have % of tsv allocation and how it increases the time.
#		- It does not discuss the number of distributed between layers
# - Should the max be recalculated for every parsing. 
#	- Since num iteration changes with every iteration. Storing index may not be a good idea
# - What should be the number of iterations run for every iteration

# Things to handle:
# - Move constants from simulator folder
# - Handle condition where simulator and model and input type do not match
# 	- i.e. mcts works only with RF and not WL and etc

def dummy_aqu_func(design, reg, y_max):
	MAX_NUM_SPARE_TSV = 100
	point  = np.zeros(CT.NUM_TSVS)
	point[CT.NUM_TSVS-1] = MAX_NUM_SPARE_TSV-1
	point[CT.NUM_TSVS-2] = 1
	value = np.sqrt(np.sum((design - point)**2))/(CT.NUM_TSVS*MAX_NUM_SPARE_TSV)
	# print(value)
	return 1-(1*value)


if __name__=="__main__":
	# generate_random_design(5)
	# design = np.zeros(384)
	# design[10] = 4
	# num_comb = generate_all_alloc_combinations(design, 0)
	# print(len(num_comb), 97*383)

	# num_comb = generate_all_alloc_combinations(np.zeros(384), 1)
	# print(len(num_comb))

	# monte_carlo_tree_search(dummy_aqu_func, None, None)

	# for i in range(6):
	# 	random_design = np.zeros(384)
	# 	total_alloc = 9+i*10
	# 	fill_based_on_priority(random_design, total_alloc)
	# 	print(random_design)

	# for i in range(10):
	# 	random_design = np.zeros(384)
	# 	fill_based_on_priority(random_design, 9)
	# 	print(random_design)

	# random_design_list =  generate_random_design(10, 9)
	# print(random_design_list)
	print(RAND_TSV_IDX_GEN.tsv_map)
	# for i in range(10):
	# 	print(random_design_list[i])

	# 
	# for i in range(10):
	# 	print RAND_TSV_IDX_GEN.rand_idx_based_on_priority()


