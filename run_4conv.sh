#!/bin/bash

python ./main.py --folder=/home/osilab7/hdd/ml_dataset \
                 --dataset=cars \
                 --output-folder=./output_conv_abal \
                 --model=4conv \
                 --hidden-size=64 \
                 --device=cuda:0 \
                 --num-ways=5 \
                 --num-shots=5 \
                 --batch-iter=300 \
                 --inner-update-num=1 \
                 --extractor-step-size=0.5 \
                 --classifier-step-size=0.5 \
                 --meta-lr=1e-3 \
                 --download \
                 --save-name=MAML

echo "finished"
