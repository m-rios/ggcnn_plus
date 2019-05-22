#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=ggcnn
#SBATCH --output=train.out
#SBATCH --mem=4GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/.ml_gpu
source ~/msc-thesis/.gpu/bin/activate
cd ~/msc-thesis

python train.py $DATA/preprocessed/190401_1122.hdf5
