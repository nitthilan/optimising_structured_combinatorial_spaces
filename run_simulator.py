import opt_comb_struct_space as ocss
import sys
import os
import datetime


# https://stackoverflow.com/questions/1432924/python-change-the-scripts-working-directory-to-the-scripts-own-directory
# Trick to make the file directory as the working directory
# This is used in aeolus where the script file is in a diferent folder
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

NUM_PROCESS_IDX=1
SYSTEM_OPT_IDX=2
SIM_LIST_IDX=3
SIM_TYPE_IDX=4
START_CONFIG_IDX=5
END_CONFIG_IDX=6
THREADING_IDX=7
BASE_FOLDER_NUM_IDX=8
# INPUT_TYPE=9
EXTRA_ARG_1=9
EXTRA_ARG_2=10
EXTRA_ARG_3=11
MAX_NUM_PARAMS=12

if(len(sys.argv) != MAX_NUM_PARAMS):
	print("Error: Not enough arguments ", len(sys.argv), sys.argv)
	print("<0-script> <1-num_process> <2-system_opt> <3-sim_list> \
		<4-sim_type> <5-start_config> <6-end_config>  \
		<7-threading:parallel/serial> <8-base_folder_num> \
		<9-extra_arg_1> <10-extra_arg_2> <11-extra_arg_3>")
	exit(-1)

print(datetime.datetime.now().strftime("%B %d, %Y"))


# The config object which is filled based on input
config = {}
config["NUM_PROCESS"] = int(sys.argv[NUM_PROCESS_IDX])

# If TSV, else: 
config["EXTRA_ARG_1"] = sys.argv[EXTRA_ARG_1]
config["EXTRA_ARG_2"] = sys.argv[EXTRA_ARG_2]
config["EXTRA_ARG_3"] = sys.argv[EXTRA_ARG_3]

if(sys.argv[SIM_TYPE_IDX] != "actual"
and sys.argv[SIM_TYPE_IDX] != "dummy"
and sys.argv[SIM_TYPE_IDX] != "dummy_tsv"
and sys.argv[SIM_TYPE_IDX] != "actual_tsv"
and sys.argv[SIM_TYPE_IDX] != "bocs"
and sys.argv[SIM_TYPE_IDX] != "borehole"
and sys.argv[SIM_TYPE_IDX] != "currinExponential"
and sys.argv[SIM_TYPE_IDX] != "univariate"
and sys.argv[SIM_TYPE_IDX] != "hosaki"
and sys.argv[SIM_TYPE_IDX] != "park2"):
	print("Error in simmulator type:actual/dummy/dummy_tsv/actual_tsv ", SIM_TYPE_IDX, sys.argv[SIM_TYPE_IDX])
	exit(-1)
config["SIM_TYPE"] = sys.argv[SIM_TYPE_IDX]

# if(sys.argv[INPUT_TYPE] != "normal"
# and sys.argv[INPUT_TYPE] != "tsv"):
# 	print("Error in input type: normal/tsv ", INPUT_TYPE, sys.argv[INPUT_TYPE])
# 	exit(-1)
# config["INPUT_TYPE"] = sys.argv[INPUT_TYPE]

if(sys.argv[THREADING_IDX] == "parallel"):
	config["RUN_PARALLEL"] = True
elif(sys.argv[THREADING_IDX] == "serial"):
	config["RUN_PARALLEL"] = False
else:
	print("Error in serial/parallel configuration", THREADING_IDX, sys.argv[THREADING_IDX])
	exit(-1)

# Based on the system the base folder changes
if(sys.argv[SYSTEM_OPT_IDX] == "my_laptop"):
	config["SIM_BASE_FOLDER"] = "../../../../reference/expensive_experiments/simulator/NoCsim_coreMapped_16thMay2017"
	output_base = "../../../../reference/expensive_experiments/simulator/2ndJune2018"
elif(sys.argv[SYSTEM_OPT_IDX] == "lab_machine"):
	if(config["INPUT_TYPE"] == "normal"):
		config["SIM_BASE_FOLDER"] ="../../../simulator/NoCsim_coreMapped_16thMay2017"
		output_base = "../../../simulator/31stJuly/"
	else: # config["INPUT_TYPE"] == "tsv"
		config["SIM_BASE_FOLDER"] ="F:/nitthilan/simulator/OneToMany27Oct"
		output_base = "F:/nitthilan/simulator/1stSept/"
elif(sys.argv[SYSTEM_OPT_IDX] == "aeolus"):
	config["SIM_BASE_FOLDER"] = "../../simulator/NoCsim_coreMapped_16thMay2017"
	output_base = "../../simulator/10thJune2018"
elif(sys.argv[SYSTEM_OPT_IDX] == "gpu_machine"):
	# config["SIM_BASE_FOLDER"] = "../../data/link_distribution/NoCsim_coreMapped_16thMay2017"
	config["SIM_BASE_FOLDER"] = "../../data/tsv/DUR250k"
	output_base = "../../data/tsv/results/09thAug2018"
elif(sys.argv[SYSTEM_OPT_IDX] == "kamiak"):
	config["SIM_BASE_FOLDER"] = "../../../simulator/NoCsim_coreMapped_16thMay2017"
	output_base = "../../../results/18thJune2018"
else:
	print("Error in system_opt ", SYSTEM_OPT_IDX, sys.argv[SYSTEM_OPT_IDX])
	exit(-1)

config["SIM_BASE_FOLDER"] += "_"+sys.argv[BASE_FOLDER_NUM_IDX] + "/"


# Choose between the 
if(sys.argv[SIM_LIST_IDX] == "base"):
	config["SIM_LIST"] = ['canneal', 'dedup', 'lu']
elif(sys.argv[SIM_LIST_IDX] == "full"):
	config["SIM_LIST"] = ['dedup', 'canneal', 'lu', 'fft','radix', 'water', 'vips', 'fluid']
elif(sys.argv[SIM_LIST_IDX] == "single"):
	config["SIM_LIST"] = ['canneal']
else:
	print("Error in sim list option ", SIM_LIST_IDX, sys.argv[SIM_LIST_IDX])
	exit(-1)

# OPT_ALGO: "DIRECT", "RANDOM", "RLS_ORDERED", 
#		"RLS_UNORDERED", "STAGE_ORDERED", "STAGE_UNORDERED"
# BO_MODEL: "GP_RBF", "GP_WL", "RF"
config_opt_list = [
{#0
	"MAX_VAL_FOLDER_TAG":"rf_rls_ord",
	"OPT_ALGO":"RLS_ORDERED",
	"BO_MODEL":"RF",
	"NUM_ITER":505
},
{#1
	"MAX_VAL_FOLDER_TAG":"rf_rls_unord",
	"OPT_ALGO":"RLS_UNORDERED",
	"BO_MODEL":"RF",
	"NUM_ITER":505
},
{#2
	"MAX_VAL_FOLDER_TAG":"rf_stg_ord",
	"OPT_ALGO":"STAGE_ORDERED",
	"BO_MODEL":"RF",
	"NUM_ITER":505
},
{#3
	"MAX_VAL_FOLDER_TAG":"rf_stg_unord",
	"OPT_ALGO":"STAGE_UNORDERED",
	"BO_MODEL":"RF",
	"NUM_ITER":505
},
{#4
	"MAX_VAL_FOLDER_TAG":"wl_rls_ord",
	"OPT_ALGO":"RLS_ORDERED",
	"BO_MODEL":"GP_WL",
	"NUM_ITER":505
},
{#5
	"MAX_VAL_FOLDER_TAG":"wl_rls_unord",
	"OPT_ALGO":"RLS_UNORDERED",
	"BO_MODEL":"GP_WL",
	"NUM_ITER":505
},
{#6
	"MAX_VAL_FOLDER_TAG":"wl_stg_ord",
	"OPT_ALGO":"STAGE_ORDERED",
	"BO_MODEL":"GP_WL",
	"NUM_ITER":505
},
{#7
	"MAX_VAL_FOLDER_TAG":"wl_stg_unord",
	"OPT_ALGO":"STAGE_UNORDERED",
	"BO_MODEL":"GP_WL",
	"NUM_ITER":505
},
{#8
	"MAX_VAL_FOLDER_TAG":"rf_mcts",
	"OPT_ALGO":"MCTS",
	"BO_MODEL":"RF",
	"NUM_ITER":505
},
{#9
	"MAX_VAL_FOLDER_TAG":"dp_mcts",
	"OPT_ALGO":"MCTS",
	"BO_MODEL":"GP_DP",
	"NUM_ITER":505
},
{#10
	"MAX_VAL_FOLDER_TAG":"jac_rls_ord",
	"OPT_ALGO":"RLS_ORDERED",
	"BO_MODEL":"GP_JAC",
	"NUM_ITER":505
},
{#11
	"MAX_VAL_FOLDER_TAG":"dp_stg_unord",
	"OPT_ALGO":"STAGE_UNORDERED",
	"BO_MODEL":"GP_DP",
	"NUM_ITER":505
},
{#12
	"MAX_VAL_FOLDER_TAG":"rbf_stg_ord",
	"OPT_ALGO":"STAGE_ORDERED",
	"BO_MODEL":"GP_RBF",
	"NUM_ITER":505
},
{#13
	"MAX_VAL_FOLDER_TAG":"rbf_stg_unord",
	"OPT_ALGO":"STAGE_UNORDERED",
	"BO_MODEL":"GP_RBF",
	"NUM_ITER":505
},
{#14
	"MAX_VAL_FOLDER_TAG":"rfk_stg_unord",
	"OPT_ALGO":"STAGE_UNORDERED",
	"BO_MODEL":"GP_RFK",
	"NUM_ITER":505
},
{#15
	"MAX_VAL_FOLDER_TAG":"fck_stg_unord",
	"OPT_ALGO":"STAGE_UNORDERED",
	"BO_MODEL":"GP_FCK",
	"NUM_ITER":505
},
{#16
	"MAX_VAL_FOLDER_TAG":"rfk_stg_unord_nsw",
	"OPT_ALGO":"STAGE_UNORDERED_NSW",
	"BO_MODEL":"GP_RFK",
	"NUM_ITER":505
},
{#17
	"MAX_VAL_FOLDER_TAG":"fck_stg_unord_nsw",
	"OPT_ALGO":"STAGE_UNORDERED_NSW",
	"BO_MODEL":"GP_FCK",
	"NUM_ITER":505
},
{#18
	"MAX_VAL_FOLDER_TAG":"rfk_smac_unord_nsw",
	"OPT_ALGO":"SMAC_UNORDERED_NSW",
	"BO_MODEL":"GP_RFK",
	"NUM_ITER":505
},
{#19
	"MAX_VAL_FOLDER_TAG":"fck_smac_unord_nsw",
	"OPT_ALGO":"SMAC_UNORDERED_NSW",
	"BO_MODEL":"GP_FCK",
	"NUM_ITER":505
},
{#20
	"MAX_VAL_FOLDER_TAG":"rfk_smac_unord",
	"OPT_ALGO":"SMAC_UNORDERED",
	"BO_MODEL":"GP_RFK",
	"NUM_ITER":505
},
{#21
	"MAX_VAL_FOLDER_TAG":"fck_smac_unord",
	"OPT_ALGO":"SMAC_UNORDERED",
	"BO_MODEL":"GP_FCK",
	"NUM_ITER":505
},
{#22
	"MAX_VAL_FOLDER_TAG":"rbf_bamlogo",
	"OPT_ALGO":"BAMLOGO",
	"BO_MODEL":"GP_RBF",
	"NUM_ITER":505
},
{#23
	"MAX_VAL_FOLDER_TAG":"rbf_structpred",
	"OPT_ALGO":"STRUCTPRED",
	"BO_MODEL":"GP_RBF",
	"NUM_ITER":505
},
{#24
	"MAX_VAL_FOLDER_TAG":"rf_structpred",
	"OPT_ALGO":"STRUCTPRED",
	"BO_MODEL":"RF",
	"NUM_ITER":505
},
{#25
	"MAX_VAL_FOLDER_TAG":"rf_smac_unord",
	"OPT_ALGO":"SMAC_UNORDERED",
	"BO_MODEL":"RF",
	"NUM_ITER":505
}]

start_config = int(sys.argv[START_CONFIG_IDX])
end_config = int(sys.argv[END_CONFIG_IDX])+1
if(start_config > end_config or end_config > len(config_opt_list)):
	print("Error in start and end config", start_config, end_config, len(config_opt_list))
	exit(-1)


def create_max_val_folder(base_path, max_val_folder_tag, sim_type,
	base_folder_num_idx):
	folder_name = "max_val"+datetime.datetime.now().strftime("_%d%B")
	folder_name += "_"+sim_type+"_sim"
	folder_name += "_"+max_val_folder_tag
	folder_name += "_"+str(base_folder_num_idx)

	folder_path = os.path.join(base_path, folder_name)

	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

	return folder_path

# Run for all the configuration
for i in range(start_config, end_config):
	# Copy the config to be run into the master configuration
	for key, value in config_opt_list[i].iteritems():
		config[key] = config_opt_list[i][key]

	# Create output folder
	config["MAX_VAL_FOLDER"] = create_max_val_folder(output_base, 
		config["MAX_VAL_FOLDER_TAG"], config["SIM_TYPE"], 
		sys.argv[BASE_FOLDER_NUM_IDX])

	# Run the configuration
	ocss.run_all_benchmark(config)



NUM_PROCESS = 3 # Set it to 1 less than the num cores in the system
SIM_BASE_FOLDER = "../../../../../reference/expensive_experiments/simulator/NoCsim_coreMapped_16thMay2017/"
# SIM_BASE_FOLDER = "../simulator/NoCsim_coreMapped_16thMay2017/"
# SIM_BASE_FOLDER = "../../../simulator/NoCsim_coreMapped_16thMay2017/"
# MAX_VAL_FOLDER_RLS_ORD = "../../../../../reference/expensive_experiments/simulator/max_val_rls_ord/"
# MAX_VAL_FOLDER_RLS_UNORD = "../../../../../reference/expensive_experiments/simulator/max_val_rls_unord/"

# MAX_VAL_FOLDER_STAGE_ORD = "../../../../../reference/expensive_experiments/simulator/max_val_stage_ord/"
# MAX_VAL_FOLDER_STAGE_UNORD = "../../../../../reference/expensive_experiments/simulator/max_val_stage_unord/"
# MAX_VAL_FOLDER_DIRECT = "../../../../../reference/expensive_experiments/simulator/max_val_direct/"

MAX_VAL_FOLDER_RLS_ORD = "../../../simulator/max_val_act_sim_rls_ord/"
MAX_VAL_FOLDER_RLS_UNORD = "../../../simulator/max_val_act_sim_rls_unord/"

MAX_VAL_FOLDER_STAGE_ORD = "../../../simulator/max_val_act_sim_stage_ord/"
MAX_VAL_FOLDER_STAGE_UNORD = "../../../simulator/max_val_act_sim_stage_unord/"

MAX_VAL_FOLDER_RLS_ORD = "../../../simulator/dummy_sim_11thJuly/max_val_rls_ord"
MAX_VAL_FOLDER_RLS_UNORD = "../../../simulator/dummy_sim_11thJuly/max_val_rls_unord"
MAX_VAL_FOLDER_STAGE_ORD = "../../../simulator/dummy_sim_11thJuly/max_val_stage_ord"
MAX_VAL_FOLDER_STAGE_UNORD = "../../../simulator/dummy_sim_11thJuly/max_val_stage_unord"

MAX_VAL_FOLDER_RLS_ORD = "../../../../../reference/expensive_experiments/simulator/dummy_sim_24thJuly/max_val_rls_ord"
MAX_VAL_FOLDER_RLS_UNORD = "../../../../../reference/expensive_experiments/simulator/dummy_sim_24thJuly/max_val_rls_unord"
MAX_VAL_FOLDER_STAGE_ORD = "../../../../../reference/expensive_experiments/simulator/dummy_sim_24thJuly/max_val_stage_ord"
MAX_VAL_FOLDER_STAGE_UNORD = "../../../../../reference/expensive_experiments/simulator/dummy_sim_124thJuly/max_val_stage_unord"

# MAX_VAL_FOLDER = "../output/max_val_3/"
# MAX_VAL_FOLDER = "../../../simulator/max_val_act_sim"

# sim_list = get_sim_list(SIM_BASE_FOLDER)
# sim_list =['dedup', 'canneal', 'lu', 'fft','radix', 'water', 'vips', 'fluid']
# sim_list =['dedup', 'canneal', 'lu']
# ocss.run_all_benchmark(SIM_BASE_FOLDER, sim_list, NUM_PROCESS, MAX_VAL_FOLDER_RLS_ORD, 
# 	run_dummy_sim = True, algo="RLS ORDERED")
# ocss.run_all_benchmark(SIM_BASE_FOLDER, sim_list, NUM_PROCESS, MAX_VAL_FOLDER_STAGE_ORD, 
# 	run_dummy_sim = True, algo="STAGE_ORDERED")
# ocss.run_all_benchmark(SIM_BASE_FOLDER, sim_list, NUM_PROCESS, MAX_VAL_FOLDER_RLS_UNORD, 
# 	run_dummy_sim = True, algo="RLS_UNORDERED")
# ocss.run_all_benchmark(SIM_BASE_FOLDER, sim_list, NUM_PROCESS, MAX_VAL_FOLDER_STAGE_UNORD, 
# 	run_dummy_sim = True, algo="STAGE_UNORDERED")

# ocss.run_all_benchmark(SIM_BASE_FOLDER, sim_list, NUM_PROCESS, MAX_VAL_FOLDER_DIRECT, 
#	run_dummy_sim = True, algo="DIRECT")


# 
# Make EI return zero when std is less than zero
# Should EI be maximised instead of minimised as shown in the code???

# Configurations to run:
# RLS: 20 random Restarts
# STAGE: 10 Iterations
# Num Iterations: 180

# Current run for GP(WL): RLS 10 itraton and h 3

# Queries: Should stage use GP for its model

# Combinations to run
# Optimiser: RLS/STAGE/DIRECT
# 	RLS: NumRandPoints (20), Search: Ordered/Unordered
#	STAGE: NumIterations (10), Search: Ordered/Unordered, RF (10 trees and min split 2)
#	DIRECT:
# Aquisition Function (AF): EI/UCB

# BO Model: RF(STRUCT)/GP(STRUCT)/GP(RBF)
#	RF(Struct): 10 trees and min split 2
#	GP(Struct): h (5)
#	GP(RBF): 

# Sim: Dummy/Real
# Num Iterations: 205 (RLS/STAGE), 505 (DIRECT)

# List of tasks:
# - Make sure the GP(Graph Kernel) is working fine
# - Make all the configuration working using commandline
# - Fixing simuntaneous runs of different algorithms
# - Ability to plot graphs from dumps


# List of control parameters
# acq function: 
# ucb, kappa, acq='ucb', kappa=2.576, xi=0.0,
# ei, kappa, Looks like xi is used for ei and kappa used for ucb
# Kernel:
# kernel=Matern(nu=2.5), n_restarts_optimizer=25,
# 



# Debug using different kernels for gaussian
# The tility does not utilise a hill climbing algorithm. Try doing a refinement on graph search
# What are the init functions like STAGE and DIRECT to be added to this frame work
# Store X the vector which gives the least value
# Call the actual simulator instead of the dummy_simulator 

# Make a class for 
#   - approximate_sw_feature <=> get_connection_idx_list_based_on_link_distribution
#   - get_next_connection_idx_1 <=> get_next_connection_idx

# acq_max_DIRECT requires "utility" to be modified as 
# x = x.reshape(1, -1)

# acq_max_DIRECT/acq_max_random is changed in two places

# The min value obtained via acq optimisation is very less 
# compared to the value after approximation
# Also After few iteration the approximation seems to match 
# the already existing value and so goes to random feature generation
# Probably this could be reduced by introducing randomness while
# choosing a probable next point while approximating since many values 
# seems to have same value
# The predicted X_feature seems to have all values near 0.5. Why is this?


# Facing this error intermittently
# /Users/kannappanjayakodinitthilan/.local/lib/python2.7/site-packages/sklearn/gaussian_process/gpr.py:427: 
# UserWarning: fmin_l_bfgs_b terminated abnormally with the  state: {'warnflag': 2, 'task': 'ABNORMAL_TERMINATION_IN_LNSRCH', 'grad': array([ -1.58650843e-05]), 'nit': 7, 'funcalls': 54}

# List of queries:
# - What should be the init number of points
# - How many iterations whould we go?
# - What is the aquisition function one should use?
# 
# Indexing a tuple
# tup2 = (1, 2, 3, 4, 5 );
# tup2[0] = 1, and tup2[0] != 5
# tup2[1] = 2 and not 4. REMEMBER IMPORTANT BUG. 
# Better to use (x,y,z) instead of index 0,1,2

# - Should the value be scaled for RegressionValues? Scaled by mesh values

# - What is the mean and std deviation for a RF model? How is this plotted in the graph?
# STAGE requires a monotonic functions. Further, randomness in the gready search does not make sure the same states would give the same minima
# The AquFunc in GP has a mean and std which RF lacks. So after evaluating the actual value it may not change as much as a GP does
#   - in Case of GP, the next best predicted value would be usually near zero and once it gets evaluated it becomes non-zero since std becomes zero there
#   - This may not be the case with RF. Probably this could be done with Graph kernals
# Just to clarify, there are two RegressionLearners one for STAGE init modeling and one for moduleing the actual simulator values

# currin
# [ 7.41376986  8.35990933  9.65773804 10.78109537 11.00050397 11.33199709
#  11.82277026 11.82277026 12.06919987 12.43579752 12.90632387 12.90632387
#  12.90632387 13.07526747 13.07526747 13.07526747 13.07526747 13.07526747
#  13.07526747 13.07526747 13.0921289  13.0921289  13.0921289  13.13242015
#  13.13242015 13.13753172 13.71048592 13.71048592 13.71048592 13.71048592
#  13.71048592 13.71048592 13.71048592 13.71048592 13.71058899 13.71058899
#  13.71886045 13.71886045 13.72456899 13.72456899 13.73853739 13.74640899
#  13.74897217 13.74897217 13.74897217 13.74925697 13.76258102 13.77350792
#  13.77350792 13.77350792 13.77350792 13.77477698]

# [ 7.41497499  8.44338424  8.68353425  9.32655021 11.27838056 11.30104223
#  11.75371824 11.82062387 11.82580589 11.93150701 11.95198842 11.98326978
#  12.19515792 12.785404   12.785404   12.87468191 12.89037118 12.89037118
#  12.9373597  12.98413787 13.03794176 13.11434601 13.12498355 13.12498355
#  13.15660548 13.1662158  13.23153564 13.24146111 13.25995562 13.25995562
#  13.25995562 13.25995562 13.25995562 13.25995562 13.25995562 13.25995562
#  13.32182535 13.32182535 13.32182535 13.32182535 13.32240765 13.32759754
#  13.32774366 13.32774366 13.32789188 13.32789188 13.32789188 13.32789188
#  13.32789344 13.36730193 13.37723464 13.37723464]

# Univariate

# [0.12810786 0.87807294 1.1107544  1.11148967 1.11148967 1.11148967
#  1.15394274 1.15394274 1.18304835 1.19523139 1.2090376  1.20991478
#  1.33528027 1.36369235 1.43291155 1.43291155 1.43291155 1.43291155
#  1.43291155 1.43291155 1.43291155 1.43291155 1.43291155 1.43291155
#  1.43291155 1.43291155 1.43291155 1.46377733 1.46377733 1.47493727
#  1.47493727 1.47493727 1.48023285 1.48023285 1.48023285 1.48023285
#  1.48023285 1.48023285 1.48023285 1.48023285 1.48023285 1.48023285
#  1.48023285 1.48023285 1.48023285 1.48023285 1.48023285 1.48023285
#  1.48073315 1.48073315 1.48073315 1.48073315]



# [0.00418652 0.37626933 0.64884033 0.67128993 0.74840625 1.06539885
#  1.2082647  1.2082647  1.21190135 1.21400225 1.21444032 1.21444032
#  1.21444032 1.21444032 1.22941327 1.25961811 1.26252622 1.26252622
#  1.26252622 1.26252622 1.26252622 1.26252622 1.26571543 1.26571543
#  1.26571543 1.28249337 1.28579724 1.30965866 1.33535839 1.33535839
#  1.33535839 1.33603049 1.36932211 1.37043779 1.37043779 1.37043779
#  1.40782081 1.40782081 1.40782081 1.40782081 1.40782081 1.46178744
#  1.47436852 1.47436852 1.47689396 1.47689396 1.47689396 1.47689396
#  1.47689396 1.47954175 1.47954175 1.47954175]

# hosaki
# [-99.42348172  -1.83348663   0.32786489   0.56460677   0.69686927
#    0.84702503   1.05440219   1.05440219   1.16603538   1.16628404
#    1.35401171   1.42229535   1.52356596   1.52356596   1.6436865
#    1.6436865    1.6436865    1.67468228   1.67468228   1.67468228
#    1.67468228   1.71064959   1.71064959   1.71064959   1.71389522
#    1.7227031    1.83438857   1.83438857   1.83438857   1.83438857
#    1.86312591   1.87139278   1.87139278   1.88287128   1.88287128
#    1.88287128   1.95398189   1.95398189   2.01728545   2.01728545
#    2.01728545   2.01728545   2.01728545   2.01728545   2.01728545
#    2.01728545   2.01728545   2.01728545   2.03187096   2.03187096
#    2.08536132   2.08536132]

# [-41.26854751  -0.7832383    0.28267744   0.78312742   1.03491712
#    1.104034     1.38280581   1.38280581   1.38873941   1.38873941
#    1.38873941   1.46165345   1.47288757   1.47288757   1.47288757
#    1.47288757   1.47288757   1.47288757   1.47288757   1.71847577
#    1.71847577   1.71847577   1.72697652   1.80386853   1.80386853
#    1.86108996   1.86108996   1.86108996   1.86108996   1.86108996
#    1.86108996   1.86108996   1.86108996   1.86108996   1.86525038
#    1.91568751   1.91568751   1.91804731   1.92313382   1.92313382
#    1.92417983   1.92417983   2.03352921   2.03352921   2.03352921
#    2.03352921   2.03352921   2.03352921   2.03352921   2.03352921
#    2.06295646   2.06295646]

# SMAC
# [-0.01265358  0.32525596  0.46044238  1.06029423  1.26396704  1.29173623
#   1.61418979  1.61418979  1.63932959  1.74732143  1.74735451  1.74735451
#   1.8811083   1.9484674   2.05514574  2.05514574  2.05514574  2.05514574
#   2.05514574  2.20286384  2.20286384  2.20774973  2.20774973  2.20774973
#   2.20774973  2.20774973  2.20774973  2.2100501   2.2100501   2.2100501
#   2.2100501   2.2100501   2.2100501   2.2100501   2.2100501   2.2100501
#   2.2100501   2.2100501   2.2100501   2.2100501   2.2100501   2.2100501
#   2.2100501   2.21088923  2.21088923  2.21088923  2.21088923  2.21088923
#   2.21088923  2.21132205  2.21272469  2.21272469]

# # park2
# [2.33238363 2.61672634 2.86248282 2.86248282 3.06402973 3.26945872
#  3.26945872 3.67989716 3.7583184  4.01162092 4.42180021 4.6852648
#  4.6852648  4.6852648  4.69953206 4.69953206 4.69953206 4.69953206
#  4.69953206 4.69953206 4.73358914 4.7661053  4.7661053  4.78912058
#  4.84812215 4.8561716  4.97738499 4.97738499 4.97738499 5.07594133
#  5.07594133 5.07594133 5.07594133 5.0817728  5.0980842  5.0980842
#  5.0980842  5.0980842  5.0980842  5.0980842  5.0980842  5.0980842
#  5.10123703 5.10511145 5.10511145 5.21068826 5.21068826 5.21068826
#  5.24005672 5.24005672 5.24005672 5.31435974]
# smac
# [1.82490886 2.52741896 2.96191989 2.96334464 3.09534866 3.21084873
#  3.21084873 3.42956857 4.18137148 4.30493355 4.44052908 4.45619182
#  4.48168559 4.48168559 4.63877758 4.63877758 4.63877758 4.63877758
#  4.7687357  4.7687357  4.76902953 4.81707725 4.81707725 4.81707725
#  4.88374636 4.88374636 4.88383104 4.88383104 4.88383104 5.07034967
#  5.13760543 5.13760543 5.18514692 5.18514692 5.18514692 5.2006883
#  5.2006883  5.30358916 5.30358916 5.30358916 5.32913475 5.32913475
#  5.39683469 5.39683469 5.39683469 5.49755189 5.49755189 5.49755189
#  5.49755189 5.49755189 5.49755189 5.49755189]

# bocs 10 features
# [0.37561285 1.35118206 2.12122957 3.33530645 4.01169492 4.08654404
#  4.09966179 4.09966179 4.16441388 4.27932498 4.45611506 4.79789496
#  4.79789496 4.99327322 5.52576856 5.53480308 5.67499394 5.78286169
#  5.78286169 6.2437926  6.4256621  6.444528   7.04837705 7.04837705
#  7.05337945 7.16018278 7.16018278 7.16018278 7.16018278 7.17904869
#  7.17904869 7.17904869 7.17904869 7.17904869 7.17904869 7.17904869
#  7.17904869 7.36091819 7.36091819 7.36091819 7.36091819 7.36091819
#  7.36091819 7.36091819 7.36091819 7.36091819 7.36091819 7.36091819
#  7.36091819 7.36091819 7.37978409 7.37978409]

# smac
# [1.98402632 3.37459548 3.62372603 3.62372603 4.39306094 4.57113124
#  5.1563618  5.1563618  5.1563618  5.1563618  5.36739404 5.36739404
#  5.36739404 5.36739404 5.45087564 5.82167631 5.92950174 5.92950174
#  5.97370157 5.99256747 5.99256747 6.08004567 6.30231965 6.30231965
#  6.63052726 6.63052726 6.63052726 6.63052726 6.63052726 6.63052726
#  6.63052726 6.63052726 6.72471847 6.72471847 6.72471847 6.72471847
#  6.72471847 6.72471847 6.72471847 6.76974075 6.76974075 6.77392635
#  6.77392635 6.99959448 6.99959448 6.99959448 6.99959448 6.99959448
#  6.99959448 6.99959448 7.00878384 7.00878384]

# bocs 16 features 100 iterations
# [ 2.77146657  3.54860095  4.21525877  4.23395086  4.48641428  5.91874928
#   5.91874928  5.91874928  6.0994214   6.67091382  6.83971071  7.04063649
#   7.04063649  7.11786434  7.13546939  7.84720444  7.84720444  7.9443249
#   7.9443249   8.04293744  8.23289622  8.6819167   8.6819167   8.6819167
#   8.6819167   9.18593738  9.18593738  9.22206822  9.25714638  9.25714638
#   9.25714638  9.46657763  9.68927268  9.68927268  9.80198243 10.07902657
#  10.07902657 10.41261989 10.47377726 10.48450526 10.48450526 10.96136392
#  10.96136392 10.96136392 10.96136392 11.16968456 11.16968456 11.16968456
#  11.16968456 11.16968456 11.16968456 11.16968456]
# smac
# [ 3.32125773  3.78396259  4.32179795  4.74161564  6.35466663  6.35466663
#   6.57440312  6.57440312  6.57440312  6.6182641   6.67244984  6.67244984
#   6.67244984  6.90736315  7.09770825  7.47317001  7.51823087  8.06025855
#   8.87002921  8.87002921  9.27168832  9.46006536  9.58627157  9.58627157
#   9.58627157  9.6765344   9.6765344   9.6765344  10.02763805 10.09124171
#  10.09124171 10.12958085 10.12958085 10.12958085 10.12958085 10.12958085
#  10.12958085 10.19416824 10.19416824 10.19416824 10.32056171 10.32056171
#  10.32056171 10.32056171 10.38038582 10.65460606 10.65460606 10.65460606
#  10.65460606 10.65460606 10.65460606 10.65460606]

