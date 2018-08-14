#!/bin/bash
#PBS -V
#PBS -N kjn_5
#PBS -q doppa
#PBS -l nodes=1:ppn=5,mem=12gb,walltime=144:00:00
#PBS -e /home/njayakodi/myfolder/output/error_5.txt
#PBS -o /home/njayakodi/myfolder/output/out_5.txt
##PBS -M n.kannappanjayakodi@wsu.edu
#PBS -m abe

cd /home/njayakodi/myfolder/nitthilan/scripts/aeolus_rls_batch/
python ../../run_simulator.py 5 aeolus base actual 2 3 parallel 5 normal 100 10000
