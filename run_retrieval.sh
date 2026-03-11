#!/bin/sh
#SBATCH --job-name=PIBISAvit_INSTRES1
#SBATCH --output=PIBISAvit_INSTRES1.txt
#SBATCH --error=PIBISAvit_INSTRES1_err.txt
#SBATCH --gres=gpu:1

# do not forget to activate any environment here
source ~/envs/pytorch/bin/activate
python3 run_retrieval.py -m db.model='ViT' db.dataset_name='instre' param.method='PIBISA'

echo "Finished"
