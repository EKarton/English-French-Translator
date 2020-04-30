import sys
sys.path.append("../Translator/src")

from collections import deque
import string

import gzip

import torch

from dataloader import vocabs
from dataloader import datasets
from dataloader.utils import get_spacy_instance, get_tokens_from_line

from model.transformer import Transformer
from model.encoder import Encoder
from model.decoder import Decoder


class Translator:
    def __init__(
        self,
        source_vocabs_path,
        target_vocabs_path,
        model_weights_path,
        word_embedding_size: int = 256,
        num_encoder_layers: int = 3,
        num_encoder_heads: int = 8,
        encoder_pf_dim: int = 512,
        encoder_dropout: int = 0.1,
        num_decoder_layers: int = 3,
        num_decoder_heads: int = 8,
        decoder_pf_dim: int = 512,
        decoder_dropout: int = 0.1,
    ):
        """ Sets up the translator

            Parameters
            ----------
            source_lang : str
                The source language. Currently supports one of: { 'en', 'fr' }
            source_vocabs_path : str
                The file path to the source vocabs
            target_lang : str
                The target language. Currently supports one of: { 'en', 'fr' }
            target_vocabs_path : str
                The file path to the target vocabs
            model_weights_path : str
                The file path to the model weights 
        """
        # Set the model params
        self.word_embedding_size = word_embedding_size
        self.num_encoder_layers = num_encoder_layers
        self.num_encoder_heads = num_encoder_heads
        self.encoder_pf_dim = encoder_pf_dim
        self.encoder_dropout = encoder_dropout
        self.num_decoder_layers = num_decoder_layers
        self.num_decoder_heads = num_decoder_heads
        self.decoder_pf_dim = decoder_pf_dim
        self.decoder_dropout = decoder_dropout

        # Get the source and target vocab word mappings
        source_vocab = vocabs.load_vocabs_from_file(source_vocabs_path)
        target_vocab = vocabs.load_vocabs_from_file(target_vocabs_path)

        source_word2id = source_vocab.get_word2id()
        target_word2id = target_vocab.get_word2id()

        # Set up the sizes, unk, padding, eos, sos
        self.source_vocab_size = len(source_word2id) + 2
        self.target_vocab_size = len(target_word2id) + 4

        self.source_unk_id = len(source_word2id)
        self.source_pad_id = len(source_word2id) + 1

        self.target_unk_id = len(target_word2id)
        self.target_sos_id = len(target_word2id) + 1
        self.target_eos_id = len(target_word2id) + 2
        self.target_pad_id = len(target_word2id) + 3

        # Set up the model
        self.model = self.build_model()
        self.model.load_state_dict(torch.load(model_weights_path))
        self.model.eval()

        del source_vocab, target_vocab, source_word2id, target_word2id

    def build_model(self):
        device = torch.device("cpu")

        encoder = Encoder(
            self.source_vocab_size,
            self.word_embedding_size,
            self.num_encoder_layers,
            self.num_encoder_heads,
            self.encoder_pf_dim,
            self.encoder_dropout,
            device,
        )

        decoder = Decoder(
            self.target_vocab_size,
            self.word_embedding_size,
            self.num_decoder_layers,
            self.num_decoder_heads,
            self.decoder_pf_dim,
            self.decoder_dropout,
            device,
        )

        model = Transformer(
            encoder,
            decoder,
            self.source_pad_id,
            self.target_sos_id,
            self.target_eos_id,
            self.target_pad_id,
            device,
        )

        return model

    def predict(self, tokens):
        """ Translates a text from one language to another language

            Parameters
            ----------
            tokens : list(int)
                A list of tokens

            Returns
            -------
            translated_text : str
                The translated text
        """
        src = [torch.tensor(tokens)]
        src = torch.nn.utils.rnn.pad_sequence(
            src, batch_first=True, padding_value=self.source_pad_id
        )
        src_lens = torch.tensor([len(src_seq) for src_seq in src])

        tokens = None
        with torch.no_grad():

            # Get the logits (size: (1, S', self.target_vocab_size))
            logits = self.model(src, src_lens)

            # Get the candidate words (size: (1, S'))
            tokens = logits.argmax(2)

            # Change size from (1, S') to (S', ), and remove the EOS
            tokens = tokens[0, :-1].tolist()

        return tokens
