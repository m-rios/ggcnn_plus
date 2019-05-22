#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --partition=regular
#SBATCH --job-name=pack
#SBATCH --output=pack.out
#SBATCH --mem=2000MB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/.ml_cpu
source ~/msc-thesis/.cpu/bin/activate
cd ~/msc-thesis
python pack_dataset.py -o $DATA/raw
