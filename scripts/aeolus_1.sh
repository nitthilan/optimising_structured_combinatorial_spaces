#!/bin/bash
#PBS -V
#PBS -N kjn_1
#PBS -q doppa
#PBS -l nodes=1:ppn=6,mem=12gb,walltime=240:00:00
#PBS -e /home/njayakodi/myfolder/output/error_1.txt
#PBS -o /home/njayakodi/myfolder/output/out_1.txt
##PBS -M n.kannappanjayakodi@wsu.edu
#PBS -m abe

cd /home/njayakodi/myfolder/nitthilan/bo/scripts/
python ../run_simulator.py 4 aeolus base actual 17 17 parallel 1 100 10000
