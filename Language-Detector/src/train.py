import sys
import os
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from dataloader import vocabs
from dataloader.datasets import Seq2VecDataset

from model.classifier import Seq2VecNN

import test
import helper


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
        self.parser.add_argument(
            "vocab", type=self.possible_gzipped_file, help="Vocabulary file",
        )
        self.parser.add_argument(
            "model_path", type=str, help="Where to store the resulting model",
        )
        self.parser.add_argument(
            "--langs", type=str, nargs="+", help="List of languages to parse"
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


def train_for_one_epoch(model, loss_function, optimizer, train_dataloader, device):
    model.train()

    train_loss = 0.0
    train_accuracy = 0.0

    for F, Y in tqdm(train_dataloader, total=len(train_dataloader)):

        # Send data to device
        F = F.to(device)
        Y = Y.to(device)

        # Forward-prop with the model
        optimizer.zero_grad()
        logits = model(F)

        # Compute the loss
        batch_loss = loss_function(logits, Y)
        train_loss += batch_loss.item()

        # Compute the accuracy
        _, predictions = torch.max(torch.round(torch.sigmoid(logits)), 1)
        batch_accuracy = predictions.eq(Y).sum().float().item() / Y.shape[0]
        train_accuracy += batch_accuracy

        batch_loss.backward()
        optimizer.step()

        del F, Y, logits, batch_loss

    train_loss /= len(train_dataloader)
    train_accuracy /= len(train_dataloader)

    return train_loss, train_accuracy


def train(opts):
    """ Trains the model """
    torch.manual_seed(opts.seed)

    vocab = vocabs.load_vocabs_from_file(opts.vocab)

    dataset = Seq2VecDataset(opts.training_dir, vocab, opts.langs)

    num_training_data = int(len(dataset) * opts.train_val_ratio)
    num_val_data = len(dataset) - num_training_data

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [num_training_data, num_val_data]
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        pin_memory=(opts.device.type == "cuda"),
        num_workers=2,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        pin_memory=(opts.device.type == "cuda"),
        num_workers=2,
    )

    model = Seq2VecNN(len(vocab.get_word2id()), 2, num_neurons_per_layer=[100, 25])
    model = model.to(opts.device)

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

        loss_function = torch.nn.CrossEntropyLoss()

        # Train
        train_loss, train_accuracy = train_for_one_epoch(
            model, loss_function, optimizer, train_dataloader, opts.device
        )

        # Evaluate the model
        eval_loss, eval_accuracy = test.evaluate_model(
            model, loss_function, val_dataloader, opts.device
        )

        print(
            f"Epoch={epoch} Train-Loss={train_loss} Train-Acc={train_accuracy} Test-Loss={eval_loss} Test-Acc={eval_accuracy} Num-Poor={num_poor}"
        )

        model.cpu()
        if eval_loss >= best_val_loss:
            num_poor += 1

        else:
            num_poor = 0
            best_eval_loss = eval_loss

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


if __name__ == "__main__":
    parser = CommandLineParser()
    opts = parser.get_options(sys.argv[1:])
    train(opts)
