#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=beam_search_no_lookahead
#SBATCH --output=beam_search_no_lookahead.out
#SBATCH --mem=50GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/msc-thesis/jobs/.ml_gpu.sh
source ~/msc-thesis/.gpu/bin/activate
cd ~/msc-thesis
python scripts/optimize_network_no_lookahead.py /home/s3485781/DATA/networks/ggcnn_rss/epoch_29_model.hdf5 /home/s3485781/DATA/ggcnn/data/datasets/D.hdf5 --epochs 5 --depth 4
#python scripts/optimize_network_no_lookahead.py /home/s3485781/DATA/networks/ggcnn_rss/epoch_29_model.hdf5 /home/s3485781/DATA/ggcnn/data/datasets/jacquard_samples.hdf5 --epochs 2 --depth 2
