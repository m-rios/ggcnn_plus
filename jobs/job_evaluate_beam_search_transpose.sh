#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --partition=regular
#SBATCH --job-name=evaluate-beam-search-transpose
#SBATCH --output=evaluate-beam-search-transpose.out
#SBATCH --mem=20GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/msc-thesis/jobs/.ml_cpu.sh
source ~/msc-thesis/.cpu/bin/activate
cd ~/msc-thesis
python core/evaluate.py /home/s3485781/DATA/networks/beam_search_last --scenes ~/DATA/scenes/shapenetsem40_5.hdf5 --nolog --models ~/DATA/shapenetsem40
