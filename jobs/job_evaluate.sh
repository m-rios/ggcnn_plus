#!/bin/bash
#SBATCH --time=3-20:00:00
#SBATCH --partition=regular
#SBATCH --job-name=evaluate
#SBATCH --output=evaluate.out
#SBATCH --mem=20GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/msc-thesis/jobs/.ml_cpu.sh
source ~/msc-thesis/.cpu/bin/activate
cd ~/msc-thesis
python core/evaluate.py ~/DATA/results/beam_search_190606_1311 --scenes ~/DATA/scenes/shapenet_1.hdf5
