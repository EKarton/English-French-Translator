import sys

import spacy
import torch

from tqdm import tqdm

from dataloader import vocabs
from dataloader import utils

import helper


class PredictCommandLineParser(helper.ModelCommandLineParser):
    def __init__(self):
        super().__init__()
        self.build_predict_parser(self.parser)

    def build_predict_parser(self, parser):
        self.parser.add_argument("source_lang", type=str, help="Source language")
        self.parser.add_argument(
            "source_vocab",
            type=self.possible_gzipped_file,
            help="Source vocabulary file",
        )
        self.parser.add_argument(
            "target_vocab",
            type=self.possible_gzipped_file,
            help="Target vocabulary file",
        )
        parser.add_argument(
            "model_path",
            type=lambda p: self.possible_gzipped_file(p, "rb", encoding=None),
            help="Where the model was stored after training. Model parameters "
            "passed via command line should match those from training",
        )
        parser.add_argument(
            "--batch-size",
            metavar="N",
            type=self.lower_bound,
            default=100,
            help="The number of sequences to process at once",
        )
        parser.add_argument(
            "--device",
            metavar="DEV",
            type=torch.device,
            default=torch.device("cpu"),
            help='Where to do training (e.g. "cpu", "cuda")',
        )
        parser.add_argument(
            "--input-text", type=str, default="Hello", help="Text to translate to"
        )


def predict(opts):

    # Get our current version of spacy
    spacy_instance = utils.get_spacy_instance(opts.source_lang)

    # Make the text lowercase and no EOF
    input_text = opts.input_text.lower().strip()

    # Parse input into tokens with spacy
    input_tokens = [token.text for token in spacy_instance.tokenizer(input_text)]

    print("Input:", " ".join(input_tokens))

    # Get the vocabs
    # TODO: Handle the case of translating from fr to en
    source_vocab = vocabs.load_vocabs_from_file(opts.source_vocab)
    target_vocab = vocabs.load_vocabs_from_file(opts.target_vocab)

    # Get the mappings
    source_word2id = source_vocab.get_word2id()
    target_word2id = target_vocab.get_word2id()

    source_id2word = source_vocab.get_id2word()
    target_id2word = target_vocab.get_id2word()

    source_vocab_size = len(source_word2id) + 2
    target_vocab_size = len(target_word2id) + 4

    src_unk, src_pad = range(len(source_word2id), source_vocab_size)
    trg_unk, trg_sos, trg_eos, trg_pad = range(len(target_word2id), target_vocab_size)

    model = helper.build_model(
        opts,
        source_vocab_size,
        target_vocab_size,
        src_pad,
        trg_sos,
        trg_eos,
        trg_pad,
        opts.device,
    )
    model.load_state_dict(torch.load(opts.model_path))
    model.eval()

    src = [torch.tensor([source_word2id[word] for word in input_tokens])]
    src_lens = torch.tensor([len(input_tokens)])
    src = torch.nn.utils.rnn.pad_sequence(src, padding_value=src_pad)

    predicted_words = None
    with torch.no_grad():

        # Get the output
        logits = model(src, src_lens)
        predicted_trg = logits.argmax(2)[0, :]

        # Remove the EOS and SOS
        predicted_trg = predicted_trg[1:-1]

        # Get the resultant sequence of words
        predicted_words = [
            target_id2word.get(word_id.item(), "NAN") for word_id in predicted_trg
        ]

    return predicted_words


if __name__ == "__main__":
    parser = PredictCommandLineParser()
    opts = parser.get_options(sys.argv[1:])
    print("Predicted word:", " ".join(predict(opts)))
