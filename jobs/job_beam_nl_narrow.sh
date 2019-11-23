#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=beam_nl_transpose
#SBATCH --output=beam_nl_transpose.out
#SBATCH --mem=50GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/msc-thesis/jobs/.ml_gpu.sh
source ~/msc-thesis/.gpu/bin/activate
cd ~/msc-thesis
python scripts/optimize_network.py /home/s3485781/DATA/networks/narrow/ /home/s3485781/DATA/ggcnn/data/datasets/D.hdf5 --epochs 2 --depth 4 --no_lookahead --heuristic iou
