#!/usr/bin/env bash

DATASET=Multi30k
TRAIN=../data/${DATASET}/Training/
TEST=../data/${DATASET}/Testing/

cd src

echo "Making the vocabulary"

python3.7 build-vocabs.py $TRAIN ../models/${DATASET}/vocab.gz --langs en fr --min-frequency 2

echo "Finished making the vocabulary"