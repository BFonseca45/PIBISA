#!/bin/sh
#SBATCH --job-name=CLIPret
#SBATCH --output=CLIPretOx.txt
#SBATCH --error=CLIPretOX_err.txt
#SBATCH --gres=gpu:1

# do not forget to activate any environment here
source ~/envs/pytorch/bin/activate
python3 run_retrieval.py -m db.model='CLIP' db.dataset_name='oxford' param.method='PIBISA','SBSM'

echo "Finished"
