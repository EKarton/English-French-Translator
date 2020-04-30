#!/usr/bin/env bash

DATASET=Multi30k
TRAIN=../data/${DATASET}/Training/
TEST=../data/${DATASET}/Testing/

cd src
source bin/activate

echo "Making the vocabulary"

python3.7 build-vocabs.py vocab $TRAIN en ../models/${DATASET}/vocab.english.gz --min-frequency 2
python3.7 build-vocabs.py vocab $TRAIN fr ../models/${DATASET}/vocab.french.gz --min-frequency 2

echo "Finished making the vocabulary"