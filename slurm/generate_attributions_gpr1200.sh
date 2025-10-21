#!/bin/bash
#SBATCH --partition=gpu_min8gb
#SBATCH --qos=gpu_min8gb
#SBATCH --job-name=gen_att_g12k         # Job name
#SBATCH -o gen_att_g12k.out             # STDOUT
#SBATCH -e gen_att_g12k.err             # STDERR

echo "m2ib"
python NIB/generate_attributions.py --dataset GPR1200Dataset --method m2ib --batch_size 32

echo "nib"
python NIB/generate_attributions.py --dataset GPR1200Dataset --method nib --batch_size 32

echo "rise"
python NIB/generate_attributions.py --dataset GPR1200Dataset --method rise --batch_size 32