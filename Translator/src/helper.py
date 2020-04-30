import sys

# sys.path.append("dataloader")
# sys.path.append("ml")

import argparse
import os
import gzip

import torch
from torch import nn
import torch.nn.functional as F

from model.decoder import Decoder
from model.encoder import Encoder
from model.transformer import Transformer


class ModelCommandLineParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=__doc__)
        self.add_common_model_options(self.parser)

    def get_options(self, args):
        return self.parser.parse_args(args)

    def add_common_model_options(self, parser):
        parser.add_argument(
            "--source-word-embedding-size",
            metavar="W",
            type=self.lower_bound,
            default=256,
            help="The size of word embeddings in the encoder",
        )
        parser.add_argument(
            "--target-word-embedding-size",
            metavar="W",
            type=self.lower_bound,
            default=256,
            help="The size of word embeddings in the encoder",
        )

        parser.add_argument(
            "--encoder-num-layers",
            metavar="H",
            type=self.lower_bound,
            default=3,
            help="The number of encoder layers in the encoder",
        )
        parser.add_argument(
            "--encoder-num-attention-heads",
            metavar="L",
            type=self.lower_bound,
            default=8,
            help="The number of heads in the encoder's attention layers",
        )
        parser.add_argument(
            "--encoder-pf-size",
            metavar="L",
            type=self.lower_bound,
            default=512,
            help="The size of the Position-Wise Feed Forward Layer",
        )
        parser.add_argument(
            "--encoder-dropout",
            metavar="p",
            type=self.proportion,
            default=0.1,
            help="The probability of dropping an encoder hidden state during training",
        )

        parser.add_argument(
            "--decoder-num-layers",
            metavar="H",
            type=self.lower_bound,
            default=3,
            help="The number of decoder layers in the decoder",
        )
        parser.add_argument(
            "--decoder-num-attention-heads",
            metavar="L",
            type=self.lower_bound,
            default=8,
            help="The number of heads in the decoder's attention layers",
        )
        parser.add_argument(
            "--decoder-pf-size",
            metavar="L",
            type=self.lower_bound,
            default=512,
            help="The size of the Position-Wise Feed Forward Layer",
        )
        parser.add_argument(
            "--decoder-dropout",
            metavar="p",
            type=self.proportion,
            default=0.1,
            help="The probability of dropping an decoder hidden state during training",
        )

    def lower_bound(self, v, low=1):
        v = int(v)
        if v < low:
            raise argparse.ArgumentTypeError(f"{v} must be at least {low}")
        return v

    def possible_gzipped_file(self, path, mode="r", encoding="utf8"):
        if path.endswith(".gz"):
            open_ = gzip.open
            if mode[-1] != "b":
                mode += "t"
        else:
            open_ = open
        try:
            f = open_(path, mode=mode, encoding=encoding)
        except OSError as e:
            raise argparse.ArgumentTypeError(f"can't open '{path}': {e}")
        return f

    def proportion(self, v, inclusive=False):
        v = float(v)
        if inclusive:
            if v < 0.0 or v > 1.0:
                raise argparse.ArgumentTypeError(f"{v} must be between [0, 1]")
        else:
            if v <= 0 or v >= 1:
                raise argparse.ArgumentTypeError(f"{v} must be between (0, 1)")
        return v


class ArgparseReadableDirAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(
                f"ArgparseReadableDirAction:{prospective_dir} is not a valid path"
            )

        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)

        else:
            raise argparse.ArgumentTypeError(
                f"ArgparseReadableDirAction:{prospective_dir} is not a readable dir"
            )


def build_model(
    opts,
    source_vocab_size: int,
    target_vocab_size: int,
    source_pad_id: int,
    target_sos_id: int,
    target_eos_id: int,
    target_pad_id: int,
    device: torch.device,
):
    """ Builds our Transformer model

        Parameters
        ----------
        opts
            The options from the command line
        source_vocab_size : int
            The vocab size in the source language
        target_vocab_size : int
            The vocab size in the target language
        source_pad_id : int
            The ID of a padding token in the source language
        target_sos_id : int
            The ID of a start-of-sequence token in the target language
        target_eos_id : int
            The ID of an end-of-sequence token in the target language
        target_pad_id : int
            The ID of a padding token in the target language
        device : torch.device
            The device to run the model on
            
        Returns
        -------
        model : Transformer
            The model
    """

    encoder = Encoder(
        source_vocab_size,
        opts.source_word_embedding_size,
        opts.encoder_num_layers,
        opts.encoder_num_attention_heads,
        opts.encoder_pf_size,
        opts.encoder_dropout,
        device,
    )

    decoder = Decoder(
        target_vocab_size,
        opts.target_word_embedding_size,
        opts.decoder_num_layers,
        opts.decoder_num_attention_heads,
        opts.decoder_pf_size,
        opts.decoder_dropout,
        device,
    )

    model = Transformer(
        encoder,
        decoder,
        source_pad_id,
        target_sos_id,
        target_eos_id,
        target_pad_id,
        device,
    )
    model.to(device)

    return model
