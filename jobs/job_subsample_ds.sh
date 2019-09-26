#!/bin/bash
##SBATCH --time=3-20:00:00
#SBATCH --partition=short
#SBATCH --job-name=subsample_ds
#SBATCH --output=subsample_ds.out
#SBATCH --mem=40GB
##SBATCH --mail-type=ALL
##SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/msc-thesis/jobs/.ml_cpu.sh
source ~/msc-thesis/.cpu/bin/activate
cd ~/msc-thesis
python scripts/subsample_dataset_images.py /home/s3485781/DATA/ggcnn/data/datasets/D.hdf5 0.5
