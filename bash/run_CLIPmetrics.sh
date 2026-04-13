#!/bin/sh
#SBATCH --job-name=CLIP
#SBATCH --output=gradECLIPt.txt
#SBATCH --error=gradECLIP_errt.txt
#SBATCH --gres=gpu:1

# do not forget to activate any environment here
source ~/envs/pytorch/bin/activate
wandb login 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

python3 run_metricsCLIP.py -m param.learnable=True param.beta=3 run.method="gradeclip"

echo "Finished"
