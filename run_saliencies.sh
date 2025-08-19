#!/bin/bash
#SBATCH --partition=gpu_min11gb
#SBATCH --qos=gpu_min11gb
#SBATCH --job-name=SSL
#SBATCH --output=/nas-ctm01/homes/bfcoelho/CIBISA/outputs.out
#SBATCH --error=/nas-ctm01/homes/bfcoelho/CIBISA/errors.err

echo "Running metrics"

python run_imageNET_saliencies.py -m db.model='ViT'

echo "Finished"