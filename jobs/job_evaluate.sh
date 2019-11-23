#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --partition=regular
#SBATCH --job-name=evaluate-shallow
#SBATCH --output=evaluate-no-lookahead.out
#SBATCH --mem=20GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/msc-thesis/jobs/.ml_cpu.sh
source ~/msc-thesis/.cpu/bin/activate
cd ~/msc-thesis
python core/evaluate.py /home/s3485781/DATA/results/beam_search_190926_2111 --scenes ~/DATA/scenes/shapenetsem40_5.hdf5 --nolog --models ~/DATA/shapenetsem40
