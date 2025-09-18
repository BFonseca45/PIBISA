#!/bin/bash
#SBATCH --partition=gpu_min8gb
#SBATCH --qos=gpu_min8gb
#SBATCH --job-name=gen_att_f8         # Job name
#SBATCH -o gen_att_f8.out             # STDOUT
#SBATCH -e gen_att_f8.err             # STDERR

echo "m2ib"
python NIB/generate_attributions.py --dataset Flickr8kDataset --method m2ib --batch_size 32

echo "nib"
python NIB/generate_attributions.py --dataset Flickr8kDataset --method nib --batch_size 32

echo "rise"
python NIB/generate_attributions.py --dataset Flickr8kDataset --method rise --batch_size 32