#!/bin/sh
#SBATCH --job-name=EncFlickrLearn
#SBATCH --output=test.txt
#SBATCH --error=test_err.txt
#SBATCH --gres=gpu:1

# do not forget to activate any environment here
source ~/envs/pytorch/bin/activate
#python3 generate_attributions.py --dataset Flickr8kDataset --method cibisa_enc --beta=1 --learn True
python3 test.py

echo "Finished"