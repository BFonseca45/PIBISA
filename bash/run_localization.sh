#!/bin/bash
#SBATCH --partition=gpu_min11gb
#SBATCH --qos=gpu_min11gb
#SBATCH --job-name=SSL
#SBATCH --output=/nas-ctm01/homes/bfcoelho/CIBISA/outputs.out
#SBATCH --error=/nas-ctm01/homes/bfcoelho/CIBISA/errors.err

echo "Running metrics"

python run_encoder_metrics.py -m db.model='ResNET', 'DINOv1_ResNET','Barlow Twins', 'SwAV', 'CLIP_ResNET'

echo "Finished"

# 'ViT', 'DINOv1_ViT','DINOv2', 'iBOT', 'CLIP_ViT'