#!/bin/bash
##SBATCH --time=14:00:00
#SBATCH --partition=short
#SBATCH --job-name=vhacd
#SBATCH --output=collisions.out
#SBATCH --mem=4GB

source ~/msc-thesis/jobs/.ml_cpu.sh
source ~/msc-thesis/.cpu/bin/activate
cd ~/msc-thesis

#python ~/msc-thesis/simulator/generate_vhacd.py $DATA/remaining $DATA/remaining
python simulator/generate_vhacd.py $DATA/shapenetsem40 $DATA/shapenetsem40
