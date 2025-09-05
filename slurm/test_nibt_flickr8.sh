#!/bin/bash
#SBATCH --partition=gpu_min12gb
#SBATCH --qos=gpu_min12gb
#SBATCH --job-name=test_nibt_f8         # Job name
#SBATCH -o test_nibt_f8.out             # STDOUT
#SBATCH -e test_nibt_f8.err             # STDERR

python NIB/generate_attributions.py --dataset Flickr8kDataset