#!/bin/bash
#SBATCH --time=02:30:00
#SBATCH --partition=regular
##SBATCH --partition=gpu
##SBATCH --gres=gpu:1
#SBATCH --job-name=evaluate_ggcnn_rss
#SBATCH --output=evaluate_ggcnn_rss.out
#SBATCH --mem=48GB

source $HOME/.modules

cd $HOME/msc-thesis
deactivate
source .cpu/bin/activate


#virtualenv .venv
#pip install -r requirements.txt

python evaluate_jacquard.py -o $HOME/ggcnn2/data/networks/ggcnn_rss $DATA/ggcnn/data/datasets/dataset_190124_1816.hdf5
