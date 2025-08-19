#!/bin/bash
#SBATCH --partition=gpu_min11gb
#SBATCH --qos=gpu_min11gb
#SBATCH --job-name=CIBISA_VGG
#SBATCH --output=/nas-ctm01/homes/bfcoelho/CIBISA/outputsVGG.out
#SBATCH --error=/nas-ctm01/homes/bfcoelho/CIBISA/errorsVGG.err

echo "Running metrics"

python run_imageNET.py -m db.model='VGG'

echo "Finished"
