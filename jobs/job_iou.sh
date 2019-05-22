#!/bin/bash
#SBATCH --time=12:01:00
#SBATCH --partition=himem
#SBATCH --job-name=iou
#SBATCH --output=iou.out
#SBATCH --mem=512GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/.ml_cpu
source ~/msc-thesis/.cpu/bin/activate
cd ~/msc-thesis
python iou_evaluation.py ~/DATA/ggcnn/data/networks/190503_1946__ggcnn_9_5_3__32_16_8 ~/DATA/preprocessed/190421_1806.hdf5 --miniou 0.4 --max_scenes 1000
