import bayesian_optimization as bo
from app.noc import simulator_interface as si
from app.tsv import simulator_tsv_decay as std

from multiprocessing import Pool
import os
import signal
from kernel import graph_kernel as gk

from kernel import gp_kernel as gpk
from sklearn.gaussian_process.kernels import RBF, Matern
from kernel import randomised_kernel as rk
from kernel import fast_cluster_kernel as fck
from app.syn_func import simulator as asfsim
from app.bocs import simulator as absim




def run_single_benchmark(config, benchmark_base, 
	output_base_name, benchmark):
	
	sim_type = config["SIM_TYPE"]
	opt_algo = config["OPT_ALGO"]
	bo_model = config["BO_MODEL"]
	n_iter = config["NUM_ITER"]
	# input_type = config["INPUT_TYPE"]
	if(sim_type == "dummy_tsv" or sim_type == "actual_tsv"):
		max_num_spare_tsv = int(config["EXTRA_ARG_1"])
		max_num_levels = int(config["EXTRA_ARG_2"])
		max_num_iterations = int(config["EXTRA_ARG_3"])
	elif(sim_type == "bocs"):
		num_features = int(config["EXTRA_ARG_1"]) # num features
		bocs_lamda = int(config["EXTRA_ARG_2"])
		bocs_alpha = float(config["EXTRA_ARG_3"])
	else:
		is_discrete = int(config["EXTRA_ARG_1"]) # 0-continous, 1-discrete,
		num_levels = int(config["EXTRA_ARG_2"])
		num_bits_per_dim = int(config["EXTRA_ARG_3"])
	
	if(sim_type == "dummy"):
		# Initialise the Dummy Simulator
		# input_benchmark = os.path.join(benchmark_base, "input_benchmark.txt")
		sim = si.Dummy_Simulator(benchmark_base)
	elif(sim_type == "actual"):
		# Initialise the actual simulator
		sim = si.Simulator(benchmark_base, benchmark, opt_val_idx=2) #0-Latency, 1-Energy, 2-EDP
	elif(sim_type == "dummy_tsv"):
		sim = std.Dummy_Simulator()
	elif(sim_type == "actual_tsv"):
		sim = std.Simulator(benchmark_base, benchmark)
	elif(sim_type == "bocs"):
		sim = absim.BPQ_Sim(num_features, bocs_lamda, bocs_alpha)
	else:
		sim = asfsim.Simulator(sim_type, is_discrete, num_levels, num_bits_per_dim)

	if(sim_type == "dummy_tsv" or sim_type == "actual_tsv"):
		output_file = output_base_name + "_output_"+sim_type+"_"+str(max_num_spare_tsv)+"_"+"_sim.csv"
	else:
		output_file = output_base_name + "_output_"+sim_type+"_sim.csv"
	
	# Initialise the BO with the simulator run function
	# kernel is None it is RBF kernel with 1.0 * RBF(1.0)
	bayes_opt = bo.BayesianOptimization(sim)

	if (bo_model == "GP_RBF"):
		# print("KErnal used has no impact")
		kernel = Matern(nu=0.5)
		bayes_opt.init_gp_params(kernel=kernel, acq='ucb')
	elif(bo_model == "GP_WL"):
		kernel = gk.GraphKernel(h=1)
		bayes_opt.init_gp_params(kernel=kernel, acq='ucb')
	elif(bo_model == "GP_JAC"):
		kernel = gpk.Jaccard()
		bayes_opt.init_gp_params(kernel=kernel, acq='ucb')
	elif(bo_model == "GP_DP"):
		kernel = gpk.DotProduct()
		bayes_opt.init_gp_params(kernel=kernel, acq='ucb')
	elif(bo_model == "GP_RFK"):
		kernel = rk.RandForrestKernel()
		# print("Random forest kernal")
		bayes_opt.init_gp_params(kernel=kernel, acq='ucb')
	elif(bo_model == "GP_FCK"):
		kernel = fck.FastClusterKernel()
		bayes_opt.init_gp_params(kernel=kernel, acq='ucb')
	elif(bo_model == "RF"):
		bayes_opt.init_random_forest(acq='ucb')

	if(sim_type == "dummy_tsv" or sim_type == "actual_tsv"):
		bayes_opt.tsv_init(max_num_spare_tsv, max_num_iterations,
			max_num_levels)
		

	# when acq is ei, parameter xi is 0.0
	bayes_opt.find_average(n_iter=n_iter,
		init_points=2,
		opt_algo=opt_algo,
		dump_max_every_n = 10,
		output_file_name = output_file,
		sim_type = sim_type)
	bayes_opt.points_to_csv(output_file)

	return

# https://stackoverflow.com/questions/34473069/multiprocessing-keyboardinterrupt-handling
def keyboard_interrupt_run_single(config, benchmark_base, 
	output_base_name, benchmark):
	try:
		return run_single_benchmark(config, benchmark_base, 
			output_base_name, benchmark)
	except KeyboardInterrupt:
		print("exited keyboard_interupt_run_single")
        os.exit(2)

def run_all_benchmark(config):

	print("Configuration:")
	print("==============")
	for key, value in config.iteritems():
		print(key,":",value)
	print("==============================")
	sim_base_folder = config["SIM_BASE_FOLDER"]
	sim_list = config["SIM_LIST"]
	num_process = config["NUM_PROCESS"]
	output_base_folder = config["MAX_VAL_FOLDER"]
	run_parallel = config["RUN_PARALLEL"]

	# https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
	if(run_parallel):
		pool = Pool(processes=num_process)
	
	results = {}
	for benchmark in sim_list:
		benchmark_base = os.path.join(sim_base_folder, benchmark)
		output_base_name = os.path.join(output_base_folder, benchmark)
		print(benchmark_base, output_base_name, benchmark)
		
		if(run_parallel):
			results[benchmark] = pool.apply_async(keyboard_interrupt_run_single, 
				[config, benchmark_base, output_base_name, benchmark])
		else:
			run_single_benchmark(config, benchmark_base, 
				output_base_name, benchmark)

			# pool.terminate()
	if(run_parallel):
		pool.close()	
		pool.join()
		print({benchmark: result.get() for benchmark, result in results.items()})
	return {benchmark: result.get() for benchmark, result in results.items()}

# https://stackoverflow.com/questions/34473069/multiprocessing-keyboardinterrupt-handling
def keyboard_interrupt_run_act_sim(sim, max_val_output_file, act_sim_output_file):
	try:
		return sim.run_sim_for_max(max_val_output_file, act_sim_output_file)
	except KeyboardInterrupt:
		print("exited keyboard_interrupt_run_act_sim")
        os.exit(2)

def run_all_simulator_for_max(all_sim_base_folder, max_val_folder, sim_list, num_process):
	pool = Pool(processes=num_process)
	results = {}
	for sim_name in sim_list:
		sim_base_folder = os.path.join(all_sim_base_folder, sim_name)
		print(sim_base_folder)
		max_val_output_file = os.path.join(max_val_folder, sim_name+"_output_actual_sim_max.csv")
		print(max_val_output_file)
		act_sim_output_file = os.path.join(max_val_folder, sim_name+"_sim_max_full.csv")
		print(act_sim_output_file)
		# Initialise the actual simulator
		sim = si.Simulator(sim_base_folder, sim_name)
		sim.run_sim_for_max(max_val_output_file, act_sim_output_file)
		# results[sim_name] = pool.apply_async(keyboard_interrupt_run_act_sim, 
	#		[sim, max_val_output_file, act_sim_output_file])

	pool.close()
	pool.join()
	print({benchmark: result.get() for benchmark, result in results.items()})

	return

def get_sim_list(sim_base_folder):
	dir_list = []
	for directory in os.listdir(sim_base_folder):
		if(directory != ".DS_Store" and directory != "contains.txt"):
			dir_list.append(directory)
	print  dir_list
	return dir_list

