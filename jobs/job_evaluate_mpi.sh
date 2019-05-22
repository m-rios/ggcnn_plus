#!/bin/bash
##SBATCH --time=10:00:00
#SBATCH --time=00:30:00
#SBATCH --partition=short
#SBATCH --job-name=evaluate
#SBATCH --output=evaluate.out
#SBATCH --mem=12GB
#SBATCH --ntasks=3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source .ml_cpu.sh
source ~/msc-thesis/.cpu/bin/activate
cd ~/msc-thesis
python core/evaluate_mpi.py --models ~/DATA/adversarial --scenes ~/DATA/scenes/adversarial_1.hdf5 ~/msc-thesis/ggcnn/data/networks/ggcnn_rss
