#!/bin/sh
#SBATCH --job-name=retrieval_paris
#SBATCH --output=paris_resnet.txt
#SBATCH --error=paris_resnet_err.txt
#SBATCH --gres=gpu:1

# do not forget to activate any environment here
source ~/envs/pytorch/bin/activate
python3 run_retrieval.py -m db.model='ResNET','VGG','ViT' db.dataset_name='paris'

echo "Finished"
