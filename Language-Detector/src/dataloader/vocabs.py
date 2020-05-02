import locale
import os
import re
from string import punctuation
from collections import Counter
import gzip

import numpy as np

import torch

import random

from tqdm import tqdm

from dataloader import utils


class VocabDataset:
    def __init__(self, word2id={}, id2word={}):
        """
            Initializes the Vocab dataset

            Parameters
            ----------
            word2id : { str : int }, optional
                A mapping of words to their ID

            id2word : { str : int }, optional
                A mapping of IDs to their words
        """

        self.word2id = word2id
        self.id2word = id2word

    def get_id2word(self):
        """ Returns a dictionary mapping an ID to a word

            Returns
            -------
            dictionary : { int : str }
                key is the ID of the word, and its value is the word itself
        """
        return self.id2word

    def get_word2id(self):
        """ Returns a dictionary mapping a word to an ID

            Returns
            -------
            dictionary : { str : int }
                key is the word itself, and its value is the ID of the word
        """
        return self.word2id


def load_vocabs_from_file(file_):
    """ Read self.word2id map from a file

        Parameters
        ----------
        file_ : str or file
            A file to read `word2id` from. If a path that ends with ``.gz``, it
            will be de-compressed via gzip.
    """
    if isinstance(file_, str):
        if file_.endswith(".gz"):
            with gzip.open(file_, mode="rt", encoding="utf8") as file_:
                return load_vocabs_from_file(file_)
        else:
            with open(file_, encoding="utf8") as file_:
                return load_vocabs_from_file(file_)

    word2id = dict()
    id2word = dict()

    for line in file_:
        line = line.strip()
        if not line:
            continue

        tokens = line.split()

        if len(tokens) == 2:
            word, id_ = tokens[0], tokens[1]
            id_ = int(id_)

            if id_ in id2word:
                raise ValueError(f"Duplicate id {id_}")
            if word in word2id:
                raise ValueError(f"Duplicate word {word}")

            word2id[word] = id_
            id2word[id_] = word

        else:
            raise Exception("Illegal vocab:", line)

    print("Loaded", len(word2id), "words")

    return VocabDataset(word2id, id2word)


def save_vocabs_to_file(vocabs, file_):
    """ Write vocabs.word2id map to a file

        Parameters
        ----------
        vocabs : VocabDataset
            The dataset to save
        file_ : str or file
            A file to write `word2id` to. If a path that ends with ``.gz``, it will be gzipped.
    """
    if isinstance(file_, str):
        if file_.endswith(".gz"):
            with gzip.open(file_, mode="wt", encoding="utf8") as file_:
                return save_vocabs_to_file(vocabs, file_)
        else:
            with open(file_, "w", encoding="utf8") as file_:
                return save_vocabs_to_file(vocabs, file_)

    id2word = vocabs.get_id2word()
    for i in range(len(id2word)):
        line = "{} {}\n".format(id2word[i], i)
        file_.write(line)


def build_vocabs_from_dir(
    train_dir_, lang, max_vocab=float("inf"), min_freq=0
):
    """ Build a vocabulary (words->ids) from transcriptions in a directory

        Parameters
        ----------
        train_dir_ : str
            A path to the transcription directory. ALWAYS use the training
            directory, not the test, directory, when building a vocabulary.

        lang : {'en', 'fr'}
            Whether to build the English vocabulary ('en') or the French one ('fr').

        max_vocab : int, optional
            The size of your vocabulary. Words with the greatest count will be
            retained.

        min_freq : int, optional
            The minimum frequency each word in the vocabulary needs to be
    """

    # Get the language corpus
    transcriptions = utils.get_parallel_text(train_dir_, [lang])
    filepaths = [os.path.join(train_dir_, trans[0]) for trans in transcriptions]

    print("Building", lang, "vocab from", len(transcriptions), "transcriptions")

    # Build a counter of tokens
    word2count = Counter()

    corpus = utils.read_transcription_files(filepaths)
    corpus_size = utils.get_size_of_corpus(filepaths)

    for tokenized, _, _ in tqdm(corpus, total=corpus_size):
        word2count.update(list(set(tokenized)))

    # Filter those that meet the frequency
    word2count = list(
        filter(lambda word_count: word_count[1] >= min_freq, word2count.items())
    )

    # Sort tokens in dec. frequency and cap it by max_vocab
    word2count = sorted(word2count, key=lambda kv: (kv[1], kv[0]), reverse=True)

    # Cap the number of words in the corpus
    if len(word2count) > max_vocab:
        word2count = word2count[0:max_vocab]

    word2id = dict((v[0], i) for i, v in enumerate(word2count))
    id2word = dict((i, v[0]) for i, v in enumerate(word2count))

    print("Built", len(word2id), "vocabs")

    return VocabDataset(word2id, id2word)


def combine_vocabs(vocab1: VocabDataset, vocab2: VocabDataset):
    """ Combines two vocabs together

        Parameters
        ----------
        vocab1 : VocabDataset
            The first vocab dataset
        vocab2 : VocabDataset
            The second vocab dataset

        Returns
        -------
        VocabDataset
            A new vocab dataset
    """
    combined_words = set()

    for key in vocab1.get_word2id():
        combined_words.add(key)

    for key in vocab2.get_word2id():
        combined_words.add(key)

    word2id = dict((word, index) for index, word in enumerate(combined_words))
    id2word = dict((index, word) for index, word in enumerate(combined_words))

    print("Built", len(word2id), "vocabs")

    return VocabDataset(word2id, id2word)
