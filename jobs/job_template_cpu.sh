#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --partition=short
#SBATCH --job-name=cpu_template
#SBATCH --output=cpu_template.out
#SBATCH --mem=200MB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.rios.munoz@student.rug.nl

source ~/.ml_cpu
source ~/msc-thesis/.cpu/bin/activate
cd ~/msc-thesis
python -c "import keras"
