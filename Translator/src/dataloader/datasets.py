import os
from tqdm import tqdm

import torch
import numpy as np

from dataloader import utils
from dataloader import vocabs


class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dir_: str,
        source_vocabs: vocabs.VocabDataset,
        target_vocabs: vocabs.VocabDataset,
        source_lang: str,
        target_lang: str,
    ):
        """ Initialize the Hansard dataset from a directory of parallel texts
            Note:
                The parallel text in 'dir_' must contain source and target transcriptions
                with the file extension 'source_lang' and 'target_lang'

            Parameters
            ----------
            dir_ : str
                A path to the directory of parallel text
            source_vocabs : VocabDataset
                A vocab dataset of the source text
            target_vocabs : VocabDataset
                A vocab dataset of the target text
            source_lang : { 'en', 'fr' }
                The source language
            target_lang : { 'en', 'fr' }
                The target language
        """
        # Get the spacy instances
        source_spacy = utils.get_spacy_instance(source_lang)
        target_spacy = utils.get_spacy_instance(target_lang)

        # Get all the files with the text
        transcriptions = utils.get_parallel_text(dir_, [source_lang, target_lang])

        source_filepaths = [os.path.join(dir_, trans[0]) for trans in transcriptions]
        target_filepaths = [os.path.join(dir_, trans[1]) for trans in transcriptions]

        source_l = utils.read_transcription_files(source_filepaths, source_spacy)
        target_l = utils.read_transcription_files(target_filepaths, target_spacy)

        source_word2id = source_vocabs.get_word2id()
        target_word2id = target_vocabs.get_word2id()

        src_unk, src_pad = range(len(source_word2id), len(source_word2id) + 2)
        trg_unk, trg_sos, trg_eos, trg_pad = range(
            len(target_word2id), len(target_word2id) + 4
        )

        corpus_iterator = zip(target_l, source_l)
        corpus_size = utils.get_size_of_corpus(source_filepaths)

        pairs = []
        src_lens = []
        trg_lens = []

        for (trg, trg_filename, _), (src, src_filename, _) in tqdm(
            corpus_iterator, total=corpus_size
        ):
            assert trg_filename[:-2] == src_filename[:-2]

            if not src or not trg:
                continue

            # Skip sentences > 50 words
            if len(src) > 50 or len(trg) > 50:
                continue

            # Skip sentences with no words
            if len(src) <= 0 or len(trg) <= 0:
                print("Found a sentence with no words in it!")
                continue

            src_tensor = torch.tensor(
                [source_word2id.get(word, src_unk) for word in src]
            )
            trg_tensor = torch.tensor(
                [trg_sos]
                + [target_word2id.get(word, trg_unk) for word in trg]
                + [trg_eos]
            )

            # Validate the contents of E and F
            if torch.any(src_tensor < 0) or torch.any(src_tensor > src_unk):
                print("src_unk:", src_unk)
                print("src_tensor:", src_tensor)
                raise ValueError(
                    "Contents of src_tensor should be <= src_unk and >= 0!"
                )

            if torch.any(trg_tensor < 0) or torch.any(trg_tensor > trg_eos):
                print("trg_eos:", trg_eos)
                print("trg_tensor:", trg_tensor)
                raise ValueError(
                    "Contents of trg_tensor should be <= trg_eos and >= 0!"
                )

            # Skip sentences that don't have any words in the vocab
            if torch.all(src_tensor == src_unk) and torch.all(
                trg_tensor[1:-1] == trg_unk
            ):
                continue

            pairs.append((src_tensor, trg_tensor))
            src_lens.append(src_tensor.size()[0])
            trg_lens.append(trg_tensor.size()[0])

        print("Number of sentence pairs:", len(pairs))

        print("Avg. num words in source text:", np.mean(src_lens))
        print("Std. num words in source text:", np.std(src_lens))
        print("Max. num words in source text:", np.max(src_lens))
        print("Min. num words in source text:", np.min(src_lens))

        print("Avg. num words in target text:", np.mean(trg_lens))
        print("Std. num words in target text:", np.std(trg_lens))
        print("Max. num words in target text:", np.max(trg_lens))
        print("Min. num words in target text:", np.min(trg_lens))

        self.source_unk = src_unk
        self.source_pad_id = src_pad
        self.source_vocab_size = len(source_word2id) + 2  # pad id and unk

        self.target_unk = trg_unk
        self.target_sos = trg_sos
        self.target_eos = trg_eos
        self.target_pad_id = trg_pad
        self.target_vocab_size = len(target_word2id) + 4  # unk, sos, eos, and pad id

        self.dir_ = dir_
        self.pairs = pairs

    def __len__(self):
        """ Returns the number of parallel texts in this dataset """
        return len(self.pairs)

    def __getitem__(self, i):
        """ Returns the i-th parallel texts in this dataset """
        return self.pairs[i]


class Seq2SeqDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self, dataset, source_pad_id, target_pad_id, batch_first=False, **kwargs
    ):
        """ Loads the dataset for the model
            It can load the dataset in parallel by setting 'num_workers' param > 0

            Parameters
            ----------
            dataset : Seq2SeqDataset
                The parallel text dataset
            source_pad_id : int
                An ID used to pad the source text for batching
            target_pad_id : int
                An ID used to pad the target text for batching
        """
        super().__init__(dataset, collate_fn=self.collate, **kwargs)

        self.source_pad_id = source_pad_id
        self.target_pad_id = target_pad_id
        self.batch_first = batch_first

    def collate(self, batch):
        """ Given a batch of source and target texts, it will pad it
            Specifically, it pads F with self.source_pad_id and E with self.target_eos

            Parameters
            ----------
            batch : A set of sequences F and E where F is torch.tensor and E is torch.tensor

            Returns
            -------
            (F, F_lens, E, E_lens) : tuple
                F is a torch.tensor of size (S, N)
                E is a torch.tensor of size (S, N)
        """
        src_batch, trg_batch = zip(*batch)
        src_lens = torch.tensor([src_seq.size()[0] for src_seq in src_batch])
        trg_lens = torch.tensor([trg_seq.size()[0] for trg_seq in trg_batch])

        src = torch.nn.utils.rnn.pad_sequence(
            src_batch, batch_first=self.batch_first, padding_value=self.source_pad_id
        )
        trg = torch.nn.utils.rnn.pad_sequence(
            trg_batch, batch_first=self.batch_first, padding_value=self.target_pad_id
        )

        return src, src_lens, trg, trg_lens
