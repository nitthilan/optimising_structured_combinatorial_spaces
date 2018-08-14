#!/bin/bash
### vcea, free_gpu, test
#SBATCH --partition=vcea        ### Partition (like a queue in PBS)
#SBATCH --job-name=kjn_sim      ### Job Name
#SBATCH --output=../../../output/kjn_sim.out         ### File in which to store job output
#SBATCH --error=../../../output/kjn_sim.err          ### File in which to store job error messages
#SBATCH --time=0-01:01:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1               ### Node count required for the job
#SBATCH --ntasks-per-node=5     ### Nuber of tasks to be launched per Node
###SBATCH --gres=gpu:1          ### General REServation of gpu:number of GPUs

# python mnist_cnn.py
python ../run_simulator.py 10 aeolus base actual 15 15 parallel 0 normal 100 10000
