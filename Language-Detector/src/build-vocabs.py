import sys
import os
import argparse
import gzip
import random

import torch

from dataloader import vocabs


class CommandLineParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=__doc__)
        self.build_vocab_parser(self.parser)

    def get_options(self, args):
        return self.parser.parse_args(args)

    def build_vocab_parser(self, parser):
        parser.add_argument(
            "training_dir",
            action=ArgparseReadableDirAction,
            help="Where the training data is located",
        )
        parser.add_argument(
            "out",
            type=lambda p: self.possible_gzipped_file(p, "w"),
            nargs="?",
            default=sys.stdout,
            help="Where to output the vocab file to. Defaults to stdout. If the "
            'path ends with ".gz", will gzip the file.',
        )
        parser.add_argument(
            "--langs", type=str, nargs="+", help="List of languages to parse"
        )
        parser.add_argument(
            "--max-vocab",
            metavar="V",
            type=self.lower_bound,
            default=float("inf"),
            help="The maximum size of the vocabulary. Words with lower frequency "
            "will be cut first. By default, it is infinity.",
        )
        parser.add_argument(
            "--min-frequency",
            metavar="V",
            type=self.lower_bound,
            default=0,
            help="The minimum frequency of words in the corpus. By default, it is 0.",
        )
        return parser

    def lower_bound(self, v, low=1):
        v = int(v)
        if v < low:
            raise argparse.ArgumentTypeError(f"{v} must be at least {low}")
        return v

    def possible_gzipped_file(self, path, mode="r"):
        if path.endswith(".gz"):
            open_ = gzip.open
            if mode[-1] != "b":
                mode += "t"
        else:
            open_ = open
        try:
            f = open_(path, mode=mode, encoding="utf8")
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


def build_vocab(opts):

    # Get vocabs for all languages
    vocabs_list = []
    for lang in opts.langs:
        vocab_dataset = vocabs.build_vocabs_from_dir(
            opts.training_dir,
            lang,
            max_vocab=opts.max_vocab,
            min_freq=opts.min_frequency,
        )

        vocabs_list.append(vocab_dataset)
    
    # Merge the vocabs
    new_vocab = None
    if len(vocabs_list) > 0:
        new_vocab = vocabs.combine_vocabs(vocabs_list[0], vocabs_list[1])
        for i in range(2, len(vocabs_list)):
            new_vocab = vocabs.combine_vocabs(new_vocab, vocabs_list[i])
    else:
        new_vocab = vocabs_list[0]

    # Save vocabs to a file
    vocabs.save_vocabs_to_file(new_vocab, opts.out)


if __name__ == "__main__":
    parser = CommandLineParser()
    opts = parser.get_options(sys.argv[1:])

    build_vocab(opts)
    