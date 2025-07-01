#!/bin/bash
#SBATCH --partition=gpu_min11gb
#SBATCH --qos=gpu_min11gb
#SBATCH --job-name=CIBISA
#SBATCH --output=/nas-ctm01/homes/bfcoelho/CIBISA/outputs.out
#SBATCH --error=/nas-ctm01/homes/bfcoelho/CIBISA/errors.err

echo "Running metrics"

python run_imageNET.py -m param.beta=1,5 db.model='ResNET','VGG','ViT'

echo "Finished"
