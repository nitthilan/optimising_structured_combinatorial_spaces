#!/bin/bash
#PBS -V
#PBS -N kjn_mesh
#PBS -q doppa
#PBS -l nodes=1:ppn=5,mem=12gb,walltime=72:00:00
#PBS -e /home/njayakodi/myfolder/output/error_mesh.txt
#PBS -o /home/njayakodi/myfolder/output/out_mesh.txt
##PBS -M n.kannappanjayakodi@wsu.edu
#PBS -m abe

cd /home/n.kannappanjayakodi/myfolder/expensive_experiments/nitthilan/tutorials
python ../tutorials/run_simulator_mesh.py

