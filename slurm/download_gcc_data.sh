#!/bin/bash
#SBATCH --partition=cpu_8cores
#SBATCH --qos=cpu_8cores
#SBATCH --job-name=download_gcc_data         # Job name
#SBATCH -o download_gcc_data.out             # STDOUT
#SBATCH -e download_gcc_data.err             # STDERR

python NIB/generate_attributions --dataset GoogleAIConceptualCations