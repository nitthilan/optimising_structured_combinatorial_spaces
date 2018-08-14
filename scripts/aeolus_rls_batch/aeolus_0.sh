#!/bin/bash
#PBS -V
#PBS -N kjn_0
#PBS -q doppa
#PBS -l nodes=1:ppn=5,mem=12gb,walltime=144:00:00
#PBS -e /home/njayakodi/myfolder/output/error_0.txt
#PBS -o /home/njayakodi/myfolder/output/out_0.txt
##PBS -M n.kannappanjayakodi@wsu.edu
#PBS -m abe

cd /home/njayakodi/myfolder/nitthilan/scripts/aeolus_rls_batch/
python ../../run_simulator.py 5 aeolus base actual 2 3 parallel 0 normal 100 10000
# python ../../run_simulator.py 5 aeolus base actual 0 1 serial 0 normal 100 10000
