#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=beam_search_transpose_ggcnn
#SBATCH --output=beam_search_transpose_ggcnn.out
#SBATCH --mem=80GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/msc-thesis/jobs/.ml_gpu.sh
source ~/msc-thesis/.gpu/bin/activate
cd ~/msc-thesis
python scripts/optimize_network.py /home/s3485781/DATA/results/beam_search_190703_1226/arch_C9x9x32_C5x5x16_C5x5x16_C3x3x8_T3x3x8_T5x5x16_T9x9x32_depth_0_model.hdf5 /home/s3485781/DATA/ggcnn/data/datasets/D.hdf5 --epochs 2 --depth 4 --expand_transpose
