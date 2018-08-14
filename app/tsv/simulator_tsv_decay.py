
# Details regarding the simulator:
# BO_NoC_MultiFrame.exe - Exe to be called
#	- Internally it calls BO_NoC.exe	
# sTSVsData.txt - 48 lines each having 8 entries. 
#	- Ordering of TSV top left to bottom right raster scan
# 	- 3x3 - so 1-9 ideally but the bottom right is ignored
# timeVsEDP_lifetime.txt - Left column last entry gives the time for that particular combination
# Reference Paper:
#	- DATE 2017: http://dl.acm.org/citation.cfm?id=3130701
# 	- file:///Users/kannappanjayakodinitthilan/Documents/myfolder/project_devan/aws_workspace/source/expensive_experiments/optimizing-combinatorial-structured-spaces/references/design_space/1608.06972_Design_space_exploration.pdf
import os, errno
import shutil
import stat
import subprocess
import numpy as np
import constants_tsv as CT
import os
import bayesian_helpers as bh
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from bayesian_helpers import UtilityFunction, unique_rows, PrintLog

from kernel import gp_kernel as gpk
from sklearn.gaussian_process import GaussianProcessRegressor


reference_tsv_time = {
	"canneal": 63528.4,
	"dedup": 373570, 
	"fft": 267211, 
	"radix":537988,
	"vips":1098150, 
	"water":781459,
	# "fluid":{"latency": 18.184, "energy": 2.24E-09, "edp": 4.07E-08},
	# "lu":{"latency": 17.269, "energy": 2.17E-09, "edp": 3.75E-08},
}

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred


# Using a dummy simulator for evaluating the cost without calling the actual simulator
class Dummy_Simulator(object):
	def __init__(self):
		kernel = gpk.DotProduct()
		self.reg = bh.ModelRegressor("gp",
            GaussianProcessRegressor(kernel=kernel,
                    n_restarts_optimizer=25))
		# Hack to debug MCTS
		feature_list = pd.read_csv("./app/tsv/tsv_dummy_input.txt", delim_whitespace=True, header=None, comment='#').as_matrix()
		self.X = feature_list[:,1:]
		self.Y = feature_list[:,0]
		# Find unique rows of X to avoid GP from breaking
		ur = unique_rows(self.X)
		# print(self.X.shape)
		self.reg.fit(self.X[ur], self.Y[ur])

		return

	def run(self, feature_vector):
		mean, std = self.reg.predict(feature_vector.reshape(1, -1))

		return mean[0]
		# return communication_cost


class Simulator(object):
	def __init__(self, sim_path, 
		sim_name
		):
		self.sim_path = sim_path
		# self.opt_val_idx = opt_val_idx
		self.sim_name = sim_name
		return
	
	# Generate the sTSVsData.txt file require for the simulator
	# Based on the information in CT.DESIGN_ORDERING update the necessary information
	def generate_tsv_data(self, feature_vector, path):
		
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

		def fill_edge(tsv_matrix, feature_vector, 
			tsv_off, fea_off):
			tsv_matrix[tsv_off+i, edg_idx[0]] =\
				feature_vector[fea_off+0+4*i]
			tsv_matrix[tsv_off+i, edg_idx[1]] =\
				feature_vector[fea_off+1+4*i]
			tsv_matrix[tsv_off+i, edg_idx[2]] =\
				feature_vector[fea_off+2+4*i]
			tsv_matrix[tsv_off+i, edg_idx[3]] =\
				feature_vector[fea_off+3+4*i]
			return

		def fill_corner(tsv_matrix, feature_vector, 
			tsv_off, fea_off):
			tsv_matrix[tsv_off+i, cor_idx[0]] =\
				feature_vector[fea_off+0+3*i]
			tsv_matrix[tsv_off+i, cor_idx[1]] =\
				feature_vector[fea_off+1+3*i]
			tsv_matrix[tsv_off+i, cor_idx[2]] =\
				feature_vector[fea_off+2+3*i]
			return
		
		# Feature vector ordering
		# [center-16]
		for i in range(16):
			
			# center
			tsv_matrix[mid_off+i, cen_idx] =\
				feature_vector[i] # Middle-center
			tsv_matrix[top_off+i, cen_idx] =\
				feature_vector[16+i] # Top-center
			tsv_matrix[bot_off+i, cen_idx] =\
				feature_vector[32+i] # Bottom-center
			# edge
			fill_edge(tsv_matrix, feature_vector, 
				mid_off, 48) # Middle-edge
			fill_edge(tsv_matrix, feature_vector, 
				top_off, 48+16*4) # Top-edge
			fill_edge(tsv_matrix, feature_vector, 
				bot_off, 48+2*16*4) # Bottom-edge
			# corner
			fill_corner(tsv_matrix, feature_vector, 
				mid_off, 48+3*16*4) # Middle-corner
			fill_corner(tsv_matrix, feature_vector, 
				top_off, 48+3*16*4+16*3) # Top-corner
			fill_corner(tsv_matrix, feature_vector, 
				bot_off, 48+3*16*4+2*16*3) # Bottom-corner

		np.savetxt(path, tsv_matrix, fmt='%d', delimiter='\t', newline='\n', header='', footer='', comments='# ')
		#print("Savetxt completed", sw_matrix.shape, path)
		return

	def _run(self, feature_vector):
		# Remove "sTSVsData.txt" if it exists
		tsv_matrix_path = os.path.join(self.sim_path, "sTSVsData.txt")
		silentremove(tsv_matrix_path)
		self.generate_tsv_data(feature_vector, tsv_matrix_path)
		print(tsv_matrix_path, feature_vector.shape)

		p = subprocess.Popen(["wine BO_NoC_MultiFrame.exe"], shell=True, 
				cwd=self.sim_path, 
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE)
		# p.wait()
		out, err = p.communicate()
		# if(err):
		# 	print("Error in running simulator ", p.returncode, err)
		# 	return 0
		# sprint ("Communicate output ", out, err, p.returncode)
		
		# print("Sim Path", self.sim_path)
		# output = p.stdout.read()
		# print ("Exe output ", output)

		# exit_status = os.system(nocsim_path)
		# print(nocsim_path, exit_status)

		# Read the output.txt for result
		output_file = os.path.join(self.sim_path, "timeVsEDP_lifetime.txt")
		with open(output_file) as f:
			content = f.readlines()

		content = [float(x.strip().split()[0]) for x in content][-1]
		# Normalisation
		content = (content - reference_tsv_time[self.sim_name])/reference_tsv_time[self.sim_name]
		print content
		return content

		# ds = Dummy_Simulator(os.path.join(self.sim_path, "input_benchmark.txt"))
		# return ds.run(feature_vector)

	# Run simulator and return the particular value for Baysian Optimisation
	def run(self, feature_vector):
		return self._run(feature_vector)

# python -m simulator.simulator_tsv_decay - use this to run the simulator alone
if __name__=="__main__":
	SIM_BASE_FOLDER = "F:/nitthilan/simulator/OneToMany27Oct/Canneal2/"
	sim = Simulator(SIM_BASE_FOLDER, "canneal")
	feature_vector = np.ones(CT.NUM_TSVS)
	for i in range((CT.NUM_TSVS/16)-3):
		feature_vector[i*16:(i+1)*16] = (i+1)*np.ones(16)

	offset = 16*18
	print(offset, CT.NUM_TSVS)
	for i in range(8):
		feature_vector[offset+i*12:offset+(i+1)*12] = (18+i+1)*np.ones(12)
	
	sim.generate_tsv_data(feature_vector, "sTSVsData.txt")

	sim.run(feature_vector)

