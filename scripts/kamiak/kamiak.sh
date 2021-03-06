#!/bin/bash
##SBATCH --partition=free_gpu        ### Partition (like a queue in PBS)
### vcea, free_gpu, test
#SBATCH --partition=vcea        ### Partition (like a queue in PBS)
#SBATCH --job-name=HiWorld      ### Job Name
#SBATCH --output=Hi.out         ### File in which to store job output
#SBATCH --error=Hi.err          ### File in which to store job error messages
#SBATCH --time=0-01:01:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1               ### Node count required for the job
#SBATCH --ntasks-per-node=1     ### Nuber of tasks to be launched per Node
###SBATCH --gres=gpu:1          ### General REServation of gpu:number of GPUs

python mnist_cnn.py
