#!/bin/bash

#SBATCH --job-name=unzip
#SBATCH --partition=day
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=20G
#SBATCH --mail-type=ALL

### For Misha
module purge
module load miniconda
module load CUDA/12.2.2 cuDNN/8.9.2.26-CUDA-12.2.2
module load GCC/12.2.0
module load git/2.38.1-GCCcore-12.2.0-nodocs

conda activate brainfm

cd /gpfs/radev/project/krishnaswamy_smita/cl2482/BrainFM/data_download
python unzip_HBN.py