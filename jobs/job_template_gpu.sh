#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=gpu_template
#SBATCH --output=gpu_template.out
#SBATCH --mem=200MB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/.ml_gpu
source ~/msc-thesis/.gpu/bin/activate
cd ~/msc-thesis
python -c "import keras"
