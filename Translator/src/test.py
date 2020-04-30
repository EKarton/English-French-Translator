import sys
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torchtext.data.metrics import bleu_score

from dataloader import vocabs
from dataloader.datasets import Seq2SeqDataset, Seq2SeqDataLoader

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
            "model_path", type=str, help="Where the model was stored after training",
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


def test(opts):
    source_vocab = vocabs.load_vocabs_from_file(opts.source_vocab)
    target_vocab = vocabs.load_vocabs_from_file(opts.target_vocab)

    test_dataset = Seq2SeqDataset(
        opts.testing_dir, source_vocab, target_vocab, opts.source_lang, opts.target_lang
    )
    test_dataloader = Seq2SeqDataLoader(
        test_dataset,
        test_dataset.source_pad_id,
        test_dataset.target_pad_id,
        batch_first=True,
        batch_size=opts.batch_size,
        shuffle=True,
        pin_memory=(opts.device.type == "cuda"),
        num_workers=4,
    )

    model = helper.build_model(
        opts,
        test_dataset.source_vocab_size,
        test_dataset.target_vocab_size,
        test_dataset.source_pad_id,
        test_dataset.target_sos,
        test_dataset.target_eos,
        test_dataset.target_pad_id,
        opts.device,
    )
    model.load_state_dict(torch.load(opts.model_path))
    model.eval()

    # The loss function
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=test_dataset.target_pad_id)

    # Evaluate the model
    test_loss = evaluate_model_by_loss_function(
        model, loss_function, test_dataloader, opts.device
    )
    test_bleu = evaluate_model_by_bleu_score(
        model,
        test_dataloader,
        opts.device,
        test_dataset.target_sos,
        test_dataset.target_eos,
        test_dataset.target_pad_id,
        target_vocab.get_id2word(),
    )

    print(f"Test loss={test_loss}, Test Bleu={test_bleu}")


def evaluate_model_by_bleu_score(
    model,
    test_dataloader,
    device,
    target_sos,
    target_eos,
    target_pad_id,
    target_id2word,
):
    """ Evaluates the model on the test set by computing its BLEU score

        Parameters
        ----------
        model : Seq2Seq
            The model
        test_dataloader : Seq2SeqDataLoader
            The dataloader for the test set
        device : torch.device
            The device to run predictions on
        target_sos : int
            The ID of a SOS token in the target language
        target_eos : int
            The ID of an EOS token in the target language
        target_pad_id : int
            The ID of a padding token in the target language
        target_id2word : { int : str }
            A mapping of word IDs in the target language to its string representative

        Returns
        -------
        bleu_score : float
            The bleu score on the test set
    """

    model.eval()
    bleu = 0

    with torch.no_grad():
        for _, (src, src_lens, trg, trg_lens) in tqdm(
            enumerate(test_dataloader), total=len(test_dataloader)
        ):

            # Send the data to the specified device
            src = src.to(device)
            src_lens = src_lens.to(device)
            trg = trg.to(device)

            # Compute BLEU score
            bleu += compute_batch_bleu_score(
                model,
                src,
                src_lens,
                trg,
                target_id2word,
                target_sos,
                target_eos,
                target_pad_id,
                device,
            )

            del src, src_lens, trg

    return bleu / len(test_dataloader)


def evaluate_model_by_loss_function(model, loss_function, test_dataloader, device):
    """ Evaluates the model on the test set by computing its loss score

        Parameters
        ----------
        model : Seq2Seq
            The model
        loss_function : torch.nn.LossFunction
            The loss function (ex: torch.nn.CrossEntropyLoss)
        test_dataloader : Seq2SeqDataLoader
            The dataloader for the test set
        device : torch.device
            The device to run predictions on

        Returns
        -------
        loss : float
            The loss score on the test set
    """
    model.eval()
    eval_loss = 0

    with torch.no_grad():
        for _, (src, src_lens, trg, trg_lens) in tqdm(
            enumerate(test_dataloader), total=len(test_dataloader)
        ):

            # Send the data to the specified device
            src = src.to(device)
            src_lens = src_lens.to(device)
            trg = trg.to(device)
            trg_lens = trg_lens.to(device)

            # Get the logits
            logits = model(src, src_lens, trg=trg[:, :-1], trg_lens=trg_lens)

            # Flatten the logits so that it is (b * (t - 1), model.target_vocab_size)
            flattened_logits = logits.contiguous().view(-1, logits.shape[2])

            # Remove the SOS and flatten it so that it is (b * (t - 1))
            flattened_trg = trg[:, 1:].contiguous().view(-1)

            # Compute loss
            loss = loss_function(flattened_logits, flattened_trg)
            eval_loss += loss.item()

            del src, src_lens, trg, trg_lens, loss, logits

    return eval_loss / len(test_dataloader)


def compute_batch_bleu_score(
    model,
    src,
    src_lens,
    trg,
    target_id2word,
    target_sos,
    target_eos,
    target_pad_id,
    device,
):
    """ Computes the BLEU score on a batch of sequences

      Parameters
      ----------
      src : torch.tensor(N, S)
        A batch of source sequences
      src_lens : torch.LongTensor(N, )
        The lengths of each source sequence in the current batch
      trg : torch.tensor(N, S')
        A batch of expected target sequences
      target_id2word : { int : str }
        A mapping of word IDs in the target language to its string representative
      target_sos : int
        The ID of a SOS token in the target language
      target_eos : int
        The ID of an EOS token in the target language
      target_pad_id : int
        The ID of a padding token in the target language
      device : torch.device
        The device to run predictions on

      Returns
      -------
      bleu_score : float
        The bleu score for the batch of sequences
  """
    # Get predicted output and add EOS to the end of each sequence (in case any seq doesn't have an EOS)
    logits = model(src, src_lens)
    predicted_trg = logits.argmax(2)

    # Remove SOS token
    expected_trg = trg[:, 1:]

    # Move to the CPU
    predicted_trg = predicted_trg.cpu().tolist()
    expected_trg = expected_trg.cpu().tolist()

    # Populate lst for bleu score
    predicted_seqs = []
    expected_seqs = []

    for i in range(len(predicted_trg)):

        predicted_seq = predicted_trg[i]
        expected_seq = expected_trg[i]

        # Remove the EOS
        if target_eos in predicted_seq:
            predicted_seq = predicted_seq[: predicted_seq.index(target_eos)]
        if target_eos in expected_seq:
            expected_seq = expected_seq[: expected_seq.index(target_eos)]

        # Convert IDs to words
        predicted_seq = [target_id2word.get(id_, "NAN") for id_ in predicted_seq]
        expected_seq = [target_id2word.get(id_, "NAN") for id_ in expected_seq]

        predicted_seqs.append(predicted_seq)
        expected_seqs.append([expected_seq])

    return bleu_score(predicted_seqs, expected_seqs)


if __name__ == "__main__":
    parser = TestCommandLineParser()
    opts = parser.get_options(sys.argv[1:])
    test(opts)
