#!/bin/bash
#PBS -V
#PBS -N kjn_dummy
#PBS -q doppa
#PBS -l nodes=1:ppn=12,mem=12gb,walltime=72:00:00
#PBS -e /home/njayakodi/myfolder/output/error_dummy.txt
#PBS -o /home/njayakodi/myfolder/output/out_dummy.txt
##PBS -M n.kannappanjayakodi@wsu.edu
#PBS -m abe

cd /home/njayakodi/myfolder/scripts/aeolus/
# python run_simulator.py 10 aeolus full dummy 0 1 parallel 0
python ../run_simulator.py 10 aeolus full actual 0 1 parallel 0
