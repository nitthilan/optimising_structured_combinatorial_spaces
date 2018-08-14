#!/bin/bash
#PBS -V
#PBS -N kjn_4
#PBS -q doppa
#PBS -l nodes=1:ppn=5,mem=12gb,walltime=144:00:00
#PBS -e /home/njayakodi/myfolder/output/error_4.txt
#PBS -o /home/njayakodi/myfolder/output/out_4.txt
##PBS -M n.kannappanjayakodi@wsu.edu
#PBS -m abe

cd /home/njayakodi/myfolder/nitthilan/scripts/aeolus_rls_batch/
python ../../run_simulator.py 5 aeolus base actual 2 3 parallel 4 normal 100 10000
