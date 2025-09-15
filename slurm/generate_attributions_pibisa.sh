#!/bin/bash
#SBATCH --partition=gpu_min12gb
#SBATCH --qos=gpu_min12gb
#SBATCH --job-name=pibisa_CLIP         # Job name
#SBATCH -o pibisa_CLIP.out             # STDOUT
#SBATCH -e pibisa_CLIP.err             # STDERR

python ../generate_attributions.py --dataset Flickr8kDataset --method cibisa --beta=2 --learn False 
python ../generate_attributions.py --dataset MSCXRDataset --method cibisa --beta=2 --learn False  