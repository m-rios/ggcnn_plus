#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=short
#SBATCH --job-name=entropy_baseline
#SBATCH --output=entropy_baseline.out
#SBATCH --mem=4GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/msc-thesis/jobs/.ml_cpu.sh
source ~/msc-thesis/.cpu/bin/activate
cd ~/msc-thesis
export MODELS_PATH=/home/s3485781/DATA/shapenetsem40
python core/evaluate_orthographic_pipeline.py \
          --network /home/s3485781/DATA/networks/beam_search_2/depth_3_arch_9x9x32_5x5x16_3x3x16_3x3x16_3x3x8_3x3x8_epoch_3_model.hdf5 \
          --scenes /home/s3485781/DATA/scenes/shapenetsem40_5.hdf5 \
          --angle 45 \
          --cam-resolution 500 \
          --scoring entropy \
          --output-file entropy \
          --gui 0 \
          --debug 0 \
          --omit-results 0