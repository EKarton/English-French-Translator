#!/usr/bin/env bash

DATASET=Multi30k
TRAIN=../data/${DATASET}/Training/
TEST=../data/${DATASET}/Testing/

echo "Training the model"

source bin/activate
cd src

python3.7 train.py "${TRAIN}" \
    en "../models/${DATASET}/vocab.english.gz" \
    fr "../models/${DATASET}/vocab.french.gz" \
    "../models/${DATASET}/model.en.fr.pt" \
    --source-word-embedding-size 256 \
    --target-word-embedding-size 256 \
    --encoder-num-layers 3 \
    --encoder-num-attention-heads 8 \
    --encoder-pf-size 512 \
    --encoder-dropout 0.1 \
    --decoder-num-layers 3 \
    --decoder-num-attention-heads 8 \
    --decoder-pf-size 512 \
    --decoder-dropout 0.1 \
    --epochs 10 \
    --train-val-ratio 0.75 \
    --batch-size 32 \
    --seed 1 \
    --device cpu \
    --resume-from-checkpoint "../models/${DATASET}/checkpoint.en.fr.pt" \
    --save-checkpoint-to "../models/${DATASET}/checkpoint.en.fr.pt" \

echo "Finished training"