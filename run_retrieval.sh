#!/bin/sh
#SBATCH --job-name=retrieval_oxford
#SBATCH --output=oxford_resnet.txt
#SBATCH --error=oxford_resnet_err.txt
#SBATCH --gres=gpu:1

# do not forget to activate any environment here
source ~/envs/pytorch/bin/activate
python run_retrieval.py -m db.model='ResNET','VGG','ViT'

echo "Finished"