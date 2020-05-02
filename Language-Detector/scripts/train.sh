#!/usr/bin/env bash

DATASET=Multi30k
TRAIN=../data/${DATASET}/Training/
TEST=../data/${DATASET}/Testing/

echo "Training the model"

source bin/activate
cd src

python3.7 train.py "${TRAIN}" \
    "../models/${DATASET}/vocab.gz" \
    "../models/${DATASET}/model.pt" \
    --langs en fr \
    --epochs 10 \
    --train-val-ratio 0.75 \
    --batch-size 32 \
    --seed 1 \
    --device cpu \
    --resume-from-checkpoint "../models/${DATASET}/checkpoint.pt" \
    --save-checkpoint-to "../models/${DATASET}/checkpoint.pt" \

echo "Finished training"