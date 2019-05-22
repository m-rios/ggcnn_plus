#!/bin/bash
##SBATCH --time=14:00:00
#SBATCH --partition=short
#SBATCH --job-name=vhacd
#SBATCH --output=collisions.out
#SBATCH --mem=4GB

source $HOME/.modules
ml load CMake

#python ~/msc-thesis/simulator/generate_vhacd.py $DATA/remaining $DATA/remaining
python ~/msc-thesis/simulator/generate_vhacd.py ~/obj_test ~/obj_test
