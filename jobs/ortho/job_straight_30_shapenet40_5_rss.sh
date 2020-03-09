#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=short
#SBATCH --job-name=straight_30_shapenet40_5_rss
#SBATCH --output=straight_30_shapenet40_5_rss.out
#SBATCH --mem=4GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/msc-thesis/jobs/.ml_cpu.sh
source ~/msc-thesis/.cpu/bin/activate
cd ~/msc-thesis
export MODELS_PATH=/home/s3485781/DATA/shapenetsem40
python core/evaluate_straight_in_approach.py \
          --network /home/s3485781/DATA/networks/ggcnn_rss/epoch_29_model.hdf5 \
          --scenes /home/s3485781/DATA/scenes/shapenetsem40_5.hdf5 \
          --angle 30 \
          --output-file $SLURM_JOB_ID \
          --save-grasps \
          --gui 0 \
          --debug 0 \
          --omit-results 0