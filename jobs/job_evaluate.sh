#!/bin/bash
#SBATCH --time=3-20:00:00
#SBATCH --partition=regular
#SBATCH --job-name=evaluate
#SBATCH --output=evaluate.out
#SBATCH --mem=4GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/.ml_cpu
source ~/msc-thesis/.cpu/bin/activate
cd ~/msc-thesis
python evaluate.py ~/DATA/ggcnn/data/networks/B_D --scenes ~/DATA/scenes/
