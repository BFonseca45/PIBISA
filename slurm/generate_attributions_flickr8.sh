#!/bin/bash
#SBATCH --partition=gpu_min12gb
#SBATCH --qos=gpu_min12gb
#SBATCH --job-name=gen_att_f8         # Job name
#SBATCH -o gen_att_f8.out             # STDOUT
#SBATCH -e gen_att_f8.err             # STDERR

python NIB/generate_attributions.py --dataset Flickr8kDataset --method rise
python NIB/generate_attributions.py --dataset Flickr8kDataset --method m2ib
python NIB/generate_attributions.py --dataset Flickr8kDataset --method nib