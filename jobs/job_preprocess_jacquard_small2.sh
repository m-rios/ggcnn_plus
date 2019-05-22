#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --partition=regular
#SBATCH --job-name=preprocess_jacquard
#SBATCH --output=preprocess_jacquard_small.out
#SBATCH --mem=4200MB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/.ml_cpu
source ~/msc-thesis/.cpu/bin/activate
cd ~/msc-thesis

python preprocess_jacquard.py --mean --fraction 0.1
