#!/bin/bash

echo "Running metrics"

python3 generate_attributions.py --dataset Flickr8kDataset --beta=1 --learn=True> Flicker_beta_learn.out 

echo "Finished"
