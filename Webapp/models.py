import sys
sys.path.append("../Translator/src")

from collections import deque
import string

import gzip

import spacy
import torch

from dataloader import vocabs
from dataloader import datasets
from dataloader.utils import get_spacy_instance, get_tokens_from_line


class Tokenizer:
    def __init__(self, lang, vocabs_path):
        """ Creates the tokenizer for the model

            Parameters
            ----------
            lang : str
                The language
            vocabs_path : str
                A path to the list of vocabs for that particular language
        """
        self.spacy_instance = get_spacy_instance(lang)
        self.vocabs = vocabs.load_vocabs_from_file(vocabs_path)
        self.punctuations = set([w for w in string.punctuation])

    def tokenize(self, text):
        """ Tokenizes the raw text into input word ID tokens for the ML model to consume

            Parameters
            ----------
            text : str
                A set of input strings
            
            Returns
            -------
            tokens : list(int)
                A list of word IDs
            num_vals : list(str)
                A list of tokens from left to right in `text` that were marked as numbers 
                and were replaced with the NUM token
            unk_vals : list(str)
                A list of tokens from left to right in `text` that were not in the vocab list
        """
        word2id = self.vocabs.get_word2id()
        source_unk_id = len(word2id)

        words, num_vals = get_tokens_from_line(text, self.spacy_instance)

        tokens = []
        unk_vals = []
        for word in words:
            if word not in word2id:
                tokens.append(source_unk_id)
                unk_vals.append(word)

            else:
                tokens.append(word2id[word])

        return tokens, num_vals, unk_vals

    def detokenize(self, tokens, num_vals, unk_vals):
        """ Translates a set of tokens from the ML model to human-readable text

            Parameters
            ----------
            tokens : list(int)
                A list of tokens
            num_vals : list(str)
                A list of tokens from left to right in the original text that were replaced with NUM tokens
            unk_vals : list(str)
                A list of tokens from left to right in the original text that were replaced with NAN tokens
        """
        id2word = self.vocabs.get_id2word()

        num_vals = deque(num_vals)
        unk_vals = deque(unk_vals)

        # Convert word IDs to words
        words = []
        for word_id in tokens:
            word = id2word.get(word_id, "NAN")
            
            if word == "NUM" and len(num_vals) > 0:
                word = num_vals.popleft()

            elif word == "NAN" and len(unk_vals) > 0:
                word = unk_vals.popleft()

            words.append(word)

        # Join the list of words to one string
        translated_text = ""
        for word in words:
            if word in self.punctuations:
                translated_text += word
            else:
                translated_text += " " + word

        return translated_text.strip()
