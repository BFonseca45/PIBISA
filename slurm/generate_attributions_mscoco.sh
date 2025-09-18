#!/bin/bash
#SBATCH --partition=gpu_min32gb
#SBATCH --qos=gpu_min32gb
#SBATCH --job-name=gen_att_mscoco         # Job name
#SBATCH -o gen_att_mscoco.out             # STDOUT
#SBATCH -e gen_att_mscoco.err             # STDERR

echo "m2ib"
python NIB/generate_attributions.py --dataset MSCocoDataset --method m2ib --batch_size 32

echo "nib"
python NIB/generate_attributions.py --dataset MSCocoDataset --method nib --batch_size 32

echo "rise"
python NIB/generate_attributions.py --dataset MSCocoDataset --method rise --batch_size 32