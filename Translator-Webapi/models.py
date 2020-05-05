import sys

sys.path.append("../Translator/src")

from collections import deque
import string

import gzip

import torch
import spacy

from dataloader import vocabs
from dataloader import datasets
from dataloader.utils import get_spacy_instance, get_tokens_from_line

from model.transformer import Transformer
from model.encoder import Encoder
from model.decoder import Decoder


class Translator:
    def __init__(
        self,
        source_lang,
        target_lang,
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
            target_lang : str
                The target language. Currently supports one of: { 'en', 'fr' }
            source_vocabs_path : str
                The file path to the source vocabs
            target_vocabs_path : str
                The file path to the target vocabs
            model_weights_path : str
                The file path to the model weights 
        """
        # Set up Spacy
        self.spacy_instance = get_spacy_instance(source_lang)

        # A set of punctuations
        self.punctuations = set([w for w in string.punctuation])

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
        self.source_vocab = vocabs.load_vocabs_from_file(source_vocabs_path)
        self.target_vocab = vocabs.load_vocabs_from_file(target_vocabs_path)

        source_word2id = self.source_vocab.get_word2id()
        target_word2id = self.target_vocab.get_word2id()

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
        self.model = self.__build_model__()
        self.model.load_state_dict(torch.load(model_weights_path))
        self.model.eval()

        del source_word2id, target_word2id

    def __build_model__(self):
        """ Create the Translator model for the CPU

            Returns
            -------
            model : Transformer
                The transformer
        """
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

    def __tokenize_sentence__(self, sentence):
        """ Tokenizes the raw text into input word ID tokens for the ML model to consume

            Parameters
            ----------
            sentence : str
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
        word2id = self.source_vocab.get_word2id()
        source_unk_id = len(word2id)

        words, num_vals = get_tokens_from_line(sentence, self.spacy_instance)

        tokens = []
        unk_vals = []
        for word in words:
            if word not in word2id:
                tokens.append(source_unk_id)
                unk_vals.append(word)

            else:
                tokens.append(word2id[word])

        return tokens, num_vals, unk_vals

    def __detokenize_translated_sentence__(
        self, sentence_tokens, num_vals, unk_vals
    ):
        """ Detokenizes a set of translated sentence tokens to human-readable text

            Parameters
            ----------
            sentence_tokens : list(int)
                A list of tokens
            num_vals : list(str)
                A list of tokens from left to right in the original text that were replaced with NUM tokens
            unk_vals : list(str)
                A list of tokens from left to right in the original text that were replaced with NAN tokens

            Returns
            -------
            translated_text : str
                A string of words and symbols from the sentence tokens
        """
        id2word = self.target_vocab.get_id2word()

        num_vals = deque(num_vals)
        unk_vals = deque(unk_vals)

        # Convert word IDs to words
        words = []
        for word_id in sentence_tokens:
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

    def __translate_sentence_tokens__(self, sentence_tokens):
        """ Translates a sentence tokens from one language to another language

            Parameters
            ----------
            sentence_tokens : list(int)
                A list of tokens from a sentence

            Returns
            -------
            translated_tokens : str
                A list of translated tokens
        """
        src = [torch.tensor(sentence_tokens)]
        src = torch.nn.utils.rnn.pad_sequence(
            src, batch_first=True, padding_value=self.source_pad_id
        )
        src_lens = torch.tensor([len(src_seq) for src_seq in src])

        translated_tokens = None
        with torch.no_grad():

            # Get the logits (size: (1, S', self.target_vocab_size))
            logits = self.model(src, src_lens)

            # Get the candidate words (size: (1, S'))
            translated_tokens = logits.argmax(2)

            # Change size from (1, S') to (S', ), and remove the EOS
            translated_tokens = translated_tokens[0, :-1].tolist()

        return translated_tokens

    def __translate_sentence__(self, sentence):
        """ Translates a sentence

            Parameters
            ----------
            sentence : str
                The sentence to translate

            Returns
            -------
            translated_sentence : str
                The translated sentence
        """
        
        # Tokenize the sentence
        tokens, num_vals, unk_vals = self.__tokenize_sentence__(sentence)

        # Translate the tokens to the target language
        translated_tokens = self.__translate_sentence_tokens__(tokens)

        # Translate the translated tokens back to a sentence
        translated_sentence = self.__detokenize_translated_sentence__(
            translated_tokens, num_vals, unk_vals
        )

        return translated_sentence

    def translate_text(self, text):
        """ Translates text (a string concatenation of one or more sentences)

            Parameters
            ----------
            text : str
                The text to translate

            Returns
            -------
            translated_text : str
                The translated text
        """
        sentences = self.spacy_instance(text).sents
        translated_text = " ".join(
            [self.__translate_sentence__(sentence.text) for sentence in sentences]
        )

        return translated_text
