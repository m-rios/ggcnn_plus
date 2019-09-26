#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=beam_search_transpose_narrow
#SBATCH --output=beam_search_transpose_narrow.out
#SBATCH --mem=100GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/msc-thesis/jobs/.ml_gpu.sh
source ~/msc-thesis/.gpu/bin/activate
cd ~/msc-thesis
python scripts/optimize_network.py /home/s3485781/DATA/results/beam_search_190625_2057/arch_C9x9x8_C5x5x4_C3x3x2_T3x3x2_T5x5x4_T5x5x4_T9x9x8_T9x9x8_depth_1_model.hdf5 /home/s3485781/DATA/ggcnn/data/datasets/D.hdf5 --epochs 2 --depth 4 --expand_transpose
