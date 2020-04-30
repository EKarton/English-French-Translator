import sys
import os
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from dataloader import vocabs
from dataloader.datasets import Seq2SeqDataset, Seq2SeqDataLoader

import helper
import test


class CommandLineParser(helper.ModelCommandLineParser):
    def __init__(self):
        super().__init__()
        self.build_training_parser()

    def build_training_parser(self):
        self.parser.add_argument(
            "training_dir",
            action=helper.ArgparseReadableDirAction,
            help="Where the training data is located",
        )
        self.parser.add_argument("source_lang", type=str, help="Source language")
        self.parser.add_argument(
            "source_vocab",
            type=self.possible_gzipped_file,
            help="Source vocabulary file",
        )
        self.parser.add_argument("target_lang", type=str, help="Target language")
        self.parser.add_argument(
            "target_vocab",
            type=self.possible_gzipped_file,
            help="Target vocabulary file",
        )
        self.parser.add_argument(
            "model_path", type=str, help="Where to store the resulting model",
        )

        stopping = self.parser.add_mutually_exclusive_group()
        stopping.add_argument(
            "--epochs",
            type=self.lower_bound,
            metavar="E",
            default=5,
            help="The number of epochs to run in total. Mutually exclusive with "
            "--patience. Defaults to 5.",
        )
        stopping.add_argument(
            "--patience",
            type=self.lower_bound,
            metavar="P",
            default=None,
            help="The number of epochs with no BLEU improvement after which to "
            "call it quits. If unset, will train until the epoch limit instead.",
        )

        self.parser.add_argument(
            "--learning-rate",
            metavar="N",
            type=float,
            default=0.001,
            help="The initial learning rate for Adam optimizer",
        )
        self.parser.add_argument(
            "--train-val-ratio",
            metavar="p",
            type=self.proportion,
            default=0.75,
            help="The % of the training data devoted to training, and 1 - % devoted to testing",
        )
        self.parser.add_argument(
            "--batch-size",
            metavar="N",
            type=self.lower_bound,
            default=100,
            help="The number of sequences to process at once",
        )
        self.parser.add_argument(
            "--device",
            metavar="DEV",
            type=torch.device,
            default=torch.device("cpu"),
            help='Where to do training (e.g. "cpu", "cuda")',
        )
        self.parser.add_argument(
            "--use-auto-mixed-precision",
            action="store_true",
            default=False,
            help="If true, it will use auto-mixed precision",
        )
        self.parser.add_argument(
            "--seed",
            type=int,
            metavar="S",
            default=0,
            help="The random seed, for reproducibility",
        )
        self.parser.add_argument(
            "--resume-from-checkpoint",
            type=str,
            default=None,
            help="Resume training from checkpoint",
        )
        self.parser.add_argument(
            "--save-checkpoint-to",
            type=str,
            default=None,
            help="Path to save epoch checkpoints",
        )


def load_checkpoint(checkpoint_filepath, model, optimizer):

    # Load the saved data
    checkpoint = torch.load(checkpoint_filepath)

    # Get the model
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Get the stats
    best_val_loss = checkpoint["best_val_loss"]
    num_poor = checkpoint["num_poor"]
    epoch = checkpoint["epoch"] + 1

    return best_val_loss, num_poor, epoch


def save_checkpoint(
    checkpoint_filepath, model, optimizer, best_val_loss, num_poor, epoch
):
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "model": model.state_dict(),
            "best_val_loss": best_val_loss,
            "num_poor": num_poor,
            "epoch": epoch,
        },
        checkpoint_filepath,
    )


def train(opts):
    """ Trains the model """
    torch.manual_seed(opts.seed)

    source_vocab = vocabs.load_vocabs_from_file(opts.source_vocab)
    target_vocab = vocabs.load_vocabs_from_file(opts.target_vocab)

    dataset = Seq2SeqDataset(
        opts.training_dir,
        source_vocab,
        target_vocab,
        opts.source_lang,
        opts.target_lang,
    )

    num_training_data = int(len(dataset) * opts.train_val_ratio)
    num_val_data = len(dataset) - num_training_data

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [num_training_data, num_val_data]
    )

    train_dataloader = Seq2SeqDataLoader(
        train_dataset,
        dataset.source_pad_id,
        dataset.target_pad_id,
        batch_first=True,
        batch_size=opts.batch_size,
        shuffle=True,
        pin_memory=(opts.device.type == "cuda"),
        num_workers=4,
    )
    val_dataloader = Seq2SeqDataLoader(
        val_dataset,
        dataset.source_pad_id,
        dataset.target_pad_id,
        batch_first=True,
        batch_size=opts.batch_size,
        shuffle=True,
        pin_memory=(opts.device.type == "cuda"),
        num_workers=4,
    )

    model = helper.build_model(
        opts,
        dataset.source_vocab_size,
        dataset.target_vocab_size,
        dataset.source_pad_id,
        dataset.target_sos,
        dataset.target_eos,
        dataset.target_pad_id,
        opts.device,
    )

    patience = opts.patience
    num_epochs = opts.epochs

    if opts.patience is None:
        patience = float("inf")
    else:
        num_epochs = float("inf")

    best_val_loss = float("inf")

    num_poor = 0
    epoch = 1

    optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)

    if opts.resume_from_checkpoint and os.path.isfile(opts.resume_from_checkpoint):
        print("Loading from checkpoint")
        best_val_loss, num_poor, epoch = load_checkpoint(
            opts.resume_from_checkpoint, model, optimizer
        )
        print(
            f"Previous state > Epoch {epoch}: Val loss={best_val_loss}, num_poor={num_poor}"
        )

    while epoch <= num_epochs and num_poor < patience:

        # Train
        loss_function = nn.CrossEntropyLoss(ignore_index=dataset.target_pad_id)
        train_loss = train_for_one_epoch(
            model, loss_function, optimizer, train_dataloader, opts.device
        )

        # Evaluate the model
        val_loss = test.evaluate_model_by_loss_function(
            model, loss_function, val_dataloader, opts.device
        )

        print(f"Epoch {epoch}: Train loss={train_loss}, Val loss={val_loss}")

        model.cpu()
        if val_loss > best_val_loss:
            num_poor += 1
        else:
            num_poor = 0
            best_val_loss = val_loss

            print("Saved model")
            torch.save(model.state_dict(), opts.model_path)

        save_checkpoint(
            opts.save_checkpoint_to, model, optimizer, best_val_loss, num_poor, epoch,
        )
        print("Saved checkpoint")

        model.to(opts.device)

        epoch += 1

    if epoch > num_epochs:
        print(f"Finished {num_epochs} epochs")
    else:
        print(f"Loss did not improve after {patience} epochs")

    val_bleu_score = test.evaluate_model_by_bleu_score(
        model,
        val_dataloader,
        opts.device,
        dataset.target_sos,
        dataset.target_eos,
        dataset.target_pad_id,
        target_vocab.get_id2word(),
    )
    print(f"Final BLEU score: {val_bleu_score}. Done.")


def train_for_one_epoch(model, loss_function, optimizer, train_dataloader, device):
    """ Trains the model on the training set

        Parameters
        ----------
        model : Seq2Seq
            The model
        loss_function : torch.nn.LossFunction
            The loss function (ex: torch.nn.CrossEntropyLoss)
        optimizer : torch.nn.Optimizer
            The optimizer (ex: SGD, AdamOptimizer, etc)
        train_dataloader : Seq2SeqDataLoader
            The dataloader for the training set
        device : torch.device
            The device to run predictions on

        Returns
        -------
        loss : float
            The loss for the training set
    """

    model.train()
    train_loss = 0.0

    for _, (src, src_lens, trg, trg_lens) in tqdm(
        enumerate(train_dataloader), total=len(train_dataloader)
    ):

        # Send the data to the specified device
        src = src.to(device)
        src_lens = src_lens.to(device)
        trg = trg.to(device)
        trg_lens = trg_lens.to(device)

        # Zeros out the model's previous gradient
        optimizer.zero_grad()

        # Get the logits
        logits = model(src, src_lens, trg=trg[:, :-1], trg_lens=trg_lens)

        # Flatten the logits so that it is (b * (t - 1), target_vocab_size)
        flattened_logits = logits.contiguous().view(-1, logits.shape[2])

        # Remove the SOS and flatten it so that it is (b * (t - 1))
        flattened_trg = trg[:, 1:].contiguous().view(-1)

        # Compute loss
        loss = loss_function(flattened_logits, flattened_trg)
        train_loss += loss.item()

        # Backward prop
        loss.backward()
        optimizer.step()

        del src, src_lens, trg, trg_lens, loss, logits

    return train_loss / len(train_dataloader)


if __name__ == "__main__":
    parser = CommandLineParser()
    opts = parser.get_options(sys.argv[1:])
    train(opts)
