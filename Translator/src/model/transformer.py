import torch
from torch import nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        source_pad_idx: int,
        target_sos: int,
        target_eos: int,
        target_pad_idx: int,
        device,
    ):
        """ Constructs the Seq2Seq model

            Parameters
            ----------
            encoder : nn.Module
                The encoder
            decoder : nn.Module
                The decoder
            source_pad_idx : int
                The padding ID in the source language
            target_sos : int
                The SOS token in the target language
            target_eos : int
                The EOS token in the target language
            target_pad_idx : int
                The padding ID in the target language
            device: torch.device
                The device to run the model on
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.source_pad_idx = source_pad_idx

        self.target_sos = target_sos
        self.target_eos = target_eos
        self.target_pad_idx = target_pad_idx

        self.device = device

    def get_source_padding_mask(self, src, src_lens):
        """ Makes a mask for our source sequences, where src_mask[i, 0, 0, j] = 1 
            if src[i, j] is not a padding token; else 0

            Parameters
            ----------
            src : torch.Longtensor(N, S)
                A batch of source sequences
            src_lens : torch.LongTensor(N, )
                The lengths of each source sequence in the current batch

            Returns
            -------
            src_mask : torch.LongTensor(N, 1, 1, S)
                The masks of each source sequence in the current batch
        """
        batch_size, seq_len = src.size()

        # Extend src_lens from (N, ) to (N, S)
        src_lens = src_lens.unsqueeze(1).repeat(1, seq_len)

        # Make a mask where src_mask[i, j] = 1 if src[i, j] == src padding token; else 0
        src_mask = torch.arange(seq_len).to(self.device)
        src_mask = src_mask.unsqueeze(0).repeat(batch_size, 1)
        src_mask = src_mask < src_lens

        # Extend src_mask from (N, S) to (N, 1, 1, S)
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)

        batch_size, seq_len = src.size()
        assert src_mask.size() == (batch_size, 1, 1, seq_len)

        return src_mask

    def get_target_padding_mask(self, trg, trg_lens):
        """ Makes a mask for our target sequences, where trg_mask[i, j] = 1 
            if a word exists in sequence i; else 0

            It relies on self.trg_pad_idx to see if the j-th spot of sequence i 
            in trg[] is a valid word or not

            Parameters
            ----------
            trg : torch.LongTensor(N, S')
            A batch of target sequences
            trg_lens : torch.LongTensor(N, )
            The lengths of each target sequence in the current batch

            Returns
            -------
            trg_mask : torch.LongTensor(N, 1, S', S')
            The mask of each target sequence in the current batch
        """
        batch_size, seq_len = trg.size()

        # Extend src_lens from (N, ) to (N, S')
        trg_lens = trg_lens.unsqueeze(1).repeat(1, seq_len)

        # Make a mask where trg_pad_mask[i, j] = 1 if trg[i, j] == trg padding token; else 0
        trg_pad_mask = torch.arange(seq_len).to(self.device)
        trg_pad_mask = trg_pad_mask.unsqueeze(0).repeat(batch_size, 1)
        trg_pad_mask = trg_pad_mask < trg_lens

        # Extend the mask from (N, S') to (N, 1, S', 1)
        trg_pad_mask = trg_pad_mask.unsqueeze(1).unsqueeze(3)

        # Apply a sub mask on the lower triangular matrix (Size: (S', S'))
        trg_sub_mask = torch.tril(
            torch.ones((seq_len, seq_len), device=self.device)
        ).bool()

        assert trg_sub_mask.size() == (seq_len, seq_len)

        # Apply a mask on the inputs (Size: (N, 1, S', S'))
        trg_mask = trg_pad_mask & trg_sub_mask

        assert trg_mask.size() == (batch_size, 1, seq_len, seq_len)

        return trg_mask

    def forward(self, src, src_lens, trg=None, trg_lens=None):
        """ Performs forward propogation of our model
            If trg == None, it will get the logits from the greedy search

            Parameters
            ----------
            src : torch.Longtensor(N, S)
            A batch of source sequences
            src_lens : torch.LongTensor(N, )
            The lengths of each source sequence in the current batch
            trg : torch.Longtensor(N, S')
            A batch of target sequences
            trg_lens : torch.LongTensor(N, )
            The lengths of each target sequence in the current batch

            Returns
            -------
            logits : torch.LongTensor(N, S', target_vocab_size)
            The set of logits for the target sequence
        """
        if self.training and trg is None:
            raise ValueError("Expected target sequence (trg) must be set!")

        if trg is not None:
            return self.get_logits_from_teacher_forcing(src, src_lens, trg, trg_lens)

        else:
            return self.get_logits_from_greedy_search(src, src_lens)

    def get_logits_from_teacher_forcing(self, src, src_lens, trg, trg_lens):
        """ Translates a batch of sequences in the source language to a target language
            with the help of its expected translated sequences.

            NOTE: This should only be used in training

            Parameters
            ----------
            src : torch.Longtensor(N, S)
            A batch of source sequences
            src_lens : torch.LongTensor(N, )
            The lengths of each source sequence in the current batch
            trg : torch.Longtensor(N, S')
            A batch of target sequences
            trg_lens : torch.LongTensor(N, )
            The lengths of each target sequence in the current batch

            Returns
            -------
            logits : torch.LongTensor(N, S', target_vocab_size)
            The set of logits for the predicted target sequence
        """
        assert src.size()[0] == trg.size()[0]

        src_mask = self.get_source_padding_mask(src, src_lens)
        encoded_src = self.encoder(src, src_mask)

        trg_mask = self.get_target_padding_mask(trg, trg_lens)
        logits = self.decoder(trg, encoded_src, trg_mask, src_mask)

        return logits

    def get_logits_from_greedy_search(self, src, src_lens, max_len=50):
        """ Translates a batch of sentences in a source language to a target language

            Parameters
            ----------
            src : torch.Longtensor(N, S)
            A batch of source sequences
            src_lens : torch.LongTensor(N, )
            The lengths of each source sequence in the current batch
            max_len : int (optional)
            The max length of the target sequence

            Returns
            -------
            logits : torch.LongTensor(N, S', target_vocab_size)
            The set of logits for the target sequence
        """
        batch_size = src.size()[0]

        # All inputs to the decoder starts with SOS
        trg = torch.tensor([[self.target_sos] for _ in range(batch_size)]).to(
            self.device
        )
        logits = None

        # Encode the source sequences
        src_mask = self.get_source_padding_mask(src, src_lens)
        encoded_src = self.encoder(src, src_mask)

        # The lengths of each target sequence
        # where trg_lens[i] = the length of sequence i from index 0 to its first EOS inclusive
        trg_lens = torch.tensor([max_len + 100 for _ in range(batch_size)]).to(
            self.device
        )

        # Decode the target sequences
        cur_len = 1
        while cur_len < max_len and torch.any(trg_lens == max_len + 100):

            # Get logits for target seq (Size: (N, cur_len, self.target_vocab_size))
            trg_mask = self.get_target_padding_mask(trg, trg_lens * 0 + cur_len)
            cur_logits = self.decoder(
                trg, encoded_src, trg_mask, src_mask
            )

            # Get the last logit in the sequences (Size: (N, self.target_vocab_size))
            last_logits = cur_logits[:, cur_len - 1, :]  

            # Get the best tokens (Size: (N, ))
            best_tokens = last_logits.argmax(1)  

            # See which has newly ended and update trg_lens (Size: (N, ))
            new_trg_lens = (best_tokens == self.target_eos).long() * cur_len
            condition = (new_trg_lens < trg_lens) & (best_tokens == self.target_eos)
            trg_lens = torch.where(condition, new_trg_lens, trg_lens)

            trg = torch.cat([trg, best_tokens.unsqueeze(-1)], dim=1)
            logits = cur_logits
            cur_len += 1

            assert trg.size() == (batch_size, cur_len)

        return logits

    def get_logits_from_beam_search(self, src, src_lens, max_len=50, beam_width=3):
        """ Translates a batch of sentences in a source language to a target language

            Parameters
            ----------
            src : torch.Longtensor(N, S)
            A batch of source sequences
            src_lens : torch.LongTensor(N, )
            The lengths of each source sequence in the current batch
            max_len : int (optional)
            The max length of the target sequence
            beam_width : int (optional)
            The beam width

            Returns
            -------
            logits : torch.LongTensor(N, beam_width, S', target_vocab_size)
            The set of logits for the target sequence per beam width
        """
        raise NotImplementedError()
