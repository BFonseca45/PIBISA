#!/bin/bash
#SBATCH --partition=gpu_min32gb
#SBATCH --qos=gpu_min32gb
#SBATCH --job-name=gen_att_mscxr         # Job name
#SBATCH -o gen_att_mscxr.out             # STDOUT
#SBATCH -e gen_att_mscxr.err             # STDERR


echo "m2ib"
python NIB/generate_attributions.py --dataset MSCXRDataset --method m2ib --batch_size 32

echo "nib"
python NIB/generate_attributions.py --dataset MSCXRDataset --method nib --batch_size 32

echo "rise"
python NIB/generate_attributions.py --dataset MSCXRDataset --method rise --batch_size 32