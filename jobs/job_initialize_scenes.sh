#!/bin/bash
##SBATCH --time=00:01:00
#SBATCH --partition=short
#SBATCH --job-name=initialize_scenes
#SBATCH --output=initialize_scenes.out
#SBATCH --mem=4GB
##SBATCH --mail-type=ALL
##SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/.ml_cpu
source ~/msc-thesis/.cpu/bin/activate
cd ~/msc-thesis

python simulator/initialize_scenes.py ~/obj_test ~/scenes