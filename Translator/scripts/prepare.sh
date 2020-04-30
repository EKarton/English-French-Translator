#!/usr/bin/env bash

cd src
source bin/activate

python3.7 -m spacy download en_core_web_sm
python3.7 -m spacy download fr_core_news_sm