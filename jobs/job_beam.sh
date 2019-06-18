#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=beam_search_10
#SBATCH --output=beam_search_10.out
#SBATCH --mem=20GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/msc-thesis/jobs/.ml_gpu.sh
source ~/msc-thesis/.gpu/bin/activate
cd ~/msc-thesis
python scripts/optimize_network.py /home/s3485781/DATA/results/beam_search_190608_1434/depth_0_arch_9x9x32_5x5x16_3x3x8_3x3x8.hdf5 /home/s3485781/DATA/ggcnn/data/datasets/D.hdf5 --epochs 10 --depth 4
