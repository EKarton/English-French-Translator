import sys

import argparse
import os
import gzip

import torch
from torch import nn
import torch.nn.functional as F


class ModelCommandLineParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=__doc__)

    def get_options(self, args):
        return self.parser.parse_args(args)

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
