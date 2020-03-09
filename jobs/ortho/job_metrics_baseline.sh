#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=regular
#SBATCH --job-name=metrics_baseline
#SBATCH --output=metrics_baseline.out
#SBATCH --mem=4GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/msc-thesis/jobs/.ml_cpu.sh
source ~/msc-thesis/.cpu/bin/activate
cd ~/msc-thesis
export MODELS_PATH=/home/s3485781/DATA/shapenetsem40
python core/compare_view_selection_metrics.py \
          --network /home/s3485781/DATA/networks/beam_search_2/depth_3_arch_9x9x32_5x5x16_3x3x16_3x3x16_3x3x8_3x3x8_epoch_3_model.hdf5 \
          --scenes /home/s3485781/DATA/scenes/shapenetsem40_5.hdf5 \
          --cam-resolution 500 \
          --output-file $SLURM_JOB_ID \
          --gui 0 \
          --debug 0 \
          --omit-results 0