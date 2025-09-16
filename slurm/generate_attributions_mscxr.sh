#!/bin/bash
#SBATCH --partition=gpu_min8gb
#SBATCH --qos=gpu_min8gb
#SBATCH --job-name=gen_att_mscxr         # Job name
#SBATCH -o gen_att_mscxr.out             # STDOUT
#SBATCH -e gen_att_mscxr.err             # STDERR

# python NIB/generate_attributions.py --dataset MSCXRDataset --method rise
python NIB/generate_attributions.py --dataset MSCXRDataset --method m2ib --batch_size 1
python NIB/generate_attributions.py --dataset MSCXRDataset --method nib --batch_size 1