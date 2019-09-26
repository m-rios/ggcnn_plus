#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --partition=regular
#SBATCH --job-name=vhacd_mpi
#SBATCH --output=vhacd_mpi.out
#SBATCH --mem=100GB
#SBATCH --ntasks=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/msc-thesis/jobs/.ml_cpu.sh
source ~/msc-thesis/.cpu/bin/activate
cd ~/msc-thesis

mpirun python scripts/vhacd_mpi.py $DATA/to_vhacd
