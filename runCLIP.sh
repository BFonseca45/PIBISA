#!/bin/bash

echo "Running metrics"

python run_imageNET.py -m param.beta=1,5 db.model='ResNET','VGG','ViT'

echo "Finished"
