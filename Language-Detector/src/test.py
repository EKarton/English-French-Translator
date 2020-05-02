import sys
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from dataloader import vocabs
from dataloader.datasets import Seq2VecDataset

from model.classifier import Seq2VecNN

import test
import helper

class TestCommandLineParser(helper.ModelCommandLineParser):
    def __init__(self):
        super().__init__()
        self.build_testing_parser()

    def build_testing_parser(self):
        self.parser.add_argument(
            "testing_dir",
            action=helper.ArgparseReadableDirAction,
            help="Where the test data is located",
        )
        self.parser.add_argument(
            "vocab", type=self.possible_gzipped_file, help="Vocabulary file",
        )
        self.parser.add_argument(
            "model_path", type=str, help="Where the model was stored after training",
        )
        self.parser.add_argument(
            "--langs", type=str, nargs="+", help="List of languages to parse"
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


def test(opts):
    torch.manual_seed(opts.seed)

    vocab = vocabs.load_vocabs_from_file(opts.vocab)

    test_dataset = Seq2VecDataset(opts.testing_dir, vocab, opts.langs)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        pin_memory=(opts.device.type == "cuda"),
        num_workers=4,
    )

    model = Seq2VecNN(len(vocab.get_word2id()), 2, num_neurons_per_layer=[100, 25])
    model = model.to(opts.device)
    model.load_state_dict(torch.load(opts.model_path))
    model.eval()

    # The loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # Evaluate the model
    test_loss, test_accuracy = evaluate_model(
        model, loss_function, test_dataloader, opts.device
    )

    print(f"Test loss={test_loss}, Test Accuracy={test_accuracy}")


def evaluate_model(model, loss_function, test_dataloader, device):

    model.eval()

    eval_loss = 0
    eval_accuracy = 0
    num_sequences = 0

    with torch.no_grad():
        for F, Y in tqdm(test_dataloader, total=len(test_dataloader)):

            # Send data to device
            F = F.to(device)
            Y = Y.to(device)

            # Get predictions
            logits = model(F)

            # Compute the loss
            batch_loss = loss_function(logits, Y)
            eval_loss += batch_loss.item()

            # Compute the accuracy
            _, predictions = torch.max(torch.round(torch.sigmoid(logits)), 1)
            batch_accuracy = predictions.eq(Y).sum().float().item() / Y.shape[0]
            eval_accuracy += batch_accuracy

            del F, Y, logits, batch_loss

    eval_loss /= len(test_dataloader)
    eval_accuracy /= len(test_dataloader)

    return eval_loss, eval_accuracy


if __name__ == "__main__":
    parser = TestCommandLineParser()
    opts = parser.get_options(sys.argv[1:])
    test(opts)
