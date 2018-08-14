#!/bin/bash
#PBS -V
#PBS -N kjn_2
#PBS -q doppa
#PBS -l nodes=1:ppn=5,mem=12gb,walltime=144:00:00
#PBS -e /home/njayakodi/myfolder/output/error_2.txt
#PBS -o /home/njayakodi/myfolder/output/out_2.txt
##PBS -M n.kannappanjayakodi@wsu.edu
#PBS -m abe

cd /home/njayakodi/myfolder/nitthilan/scripts/aeolus_rls_batch/
python ../../run_simulator.py 5 aeolus base actual 2 3 parallel 2 normal 100 10000
