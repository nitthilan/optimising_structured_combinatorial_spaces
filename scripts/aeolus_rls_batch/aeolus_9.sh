#!/bin/bash
#PBS -V
#PBS -N kjn_9
#PBS -q doppa
#PBS -l nodes=1:ppn=5,mem=12gb,walltime=288:00:00
#PBS -e /home/njayakodi/myfolder/output/error_9.txt
#PBS -o /home/njayakodi/myfolder/output/out_9.txt
##PBS -M n.kannappanjayakodi@wsu.edu
#PBS -m abe

cd /home/njayakodi/myfolder/nitthilan/scripts/aeolus_rls_batch/
python ../../run_simulator.py 5 aeolus base actual 12 13 parallel 9 normal 100 10000
