import os
from tqdm import tqdm

import torch
import numpy as np

from dataloader import utils
from dataloader import vocabs


class Seq2VecDataset(torch.utils.data.Dataset):
    def __init__(self, dir_: str, vocab: vocabs.VocabDataset, langs: list):
        """ Reads in the sentences in `dir_`, convert each word into its numerical token, 
            and tags it with its language

            Parameters
            ----------
            dir_ : str
                The directory with the training data
            vocab : vocabs.VocabDataset
                The vocab
            lang : [ str ]
                The set of languages to capture in `dir_`
        """

        pairs = []
        word2index = vocab.get_word2id()

        for lang_index, lang in enumerate(langs):

            # Get all the filenames with that language
            transcriptions = utils.get_parallel_text(dir_, [lang])

            # Get all the filepaths with that language
            filepaths = [os.path.join(dir_, trans[0]) for trans in transcriptions]

            # Get the iterator that will read and tokenize all the content in filepaths
            iterator = utils.read_transcription_files(filepaths)

            # Get the number of sentences in entire corpus with that language
            corpus_size = utils.get_size_of_corpus(filepaths)

            for ((f, f_fn, _),) in tqdm(iterable=zip(iterator), total=corpus_size):
                if not f:
                    continue

                # Ignore sentences with no words in vocabs
                has_known_word = sum([1 if word in word2index else 0 for word in f]) > 0
                if not has_known_word:
                    continue

                pairs.append((f, lang_index))

        self.langs = langs
        self.pairs = pairs
        self.word2index = word2index

    def __len__(self):
        """ Returns the number of sentences in this dataset """

        return len(self.pairs)

    def __getitem__(self, i):
        """ Returns the i-th sentence in this dataset """

        f, lang_index = self.pairs[i]

        # Get the bag of words for f where vec[i] = 1 if word i exists; else 0
        F = torch.zeros(len(self.word2index))
        for word in f:
            if word in self.word2index:
                index = self.word2index[word]
                F[index] = 1

        Y = torch.tensor(lang_index)

        return F, Y

