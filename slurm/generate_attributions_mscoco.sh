#!/bin/bash
#SBATCH --partition=gpu_min8gb
#SBATCH --qos=gpu_min8gb
#SBATCH --job-name=gen_att_mscoco         # Job name
#SBATCH -o gen_att_mscoco.out             # STDOUT
#SBATCH -e gen_att_mscoco.err             # STDERR

echo "m2ib"
python NIB/generate_attributions.py --dataset MSCocoDataset --method m2ib

echo "nib"
python NIB/generate_attributions.py --dataset MSCocoDataset --method nib

echo "rise"
python NIB/generate_attributions.py --dataset MSCocoDataset --method rise