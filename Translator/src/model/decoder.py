import torch
from torch import nn
import torch.nn.functional as F

from .attention import MultiHeadedSelfAttention
from .positionwise import PositionwiseFeedforwardLayer


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        pf_dim: int,
        dropout_value: float,
        device: torch.device,
    ):
        """ Creates a decoder layer. It should only be used in the Decoder class

            Parameter
            ---------
            hidden_size : int
            The hidden size
            num_heads : int
            The number of heads for the self-attention layers
            pf_dim : int
            The dimension for the PositionwiseFeedforward layer
            dropout_value : float
            The dropout value
            device : torch.Device
            The device to run the model
        """
        super().__init__()

        self.norm_layer = nn.LayerNorm(hidden_size)

        self.attention_layer_1 = MultiHeadedSelfAttention(
            hidden_size, device, num_heads=num_heads, dropout_value=dropout_value
        )
        self.attention_layer_2 = MultiHeadedSelfAttention(
            hidden_size, device, num_heads=num_heads, dropout_value=dropout_value
        )

        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hidden_size, pf_dim, dropout_value
        )

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, trg, encoder_src, trg_mask, src_mask):
        """ Performs a forward prop on the Decoder layer

            Paremeters
            ----------
            trg : torch.LongTensor(N, S', H)
            A batch of expected target sequences with embeddings
            encoder_src : torch.LongTensor(N, S, H)
            A batch of source sequence outputted from the Encoder
            trg_mask : torch.LongTensor(N, S')
            The masks for each target sequence in the current batch
            src_mask : torch.LongTensor(N, S)
            The masks for each source sequence in the current batch

            Returns
            -------
            decoded_trg : torch.tensor(N, S', H)
            A batch of decoded target sequences with embeddings
        """
        assert trg.size()[0] == encoder_src.size()[0]
        assert trg.size()[2] == encoder_src.size()[2]

        # Apply attention
        attended_trg = self.attention_layer_1(trg, trg, trg, trg_mask)

        # Apply dropout and layer norm
        trg = self.norm_layer(trg + self.dropout(attended_trg))

        # Apply attention
        attended_trg = self.attention_layer_2(encoder_src, encoder_src, trg, src_mask)

        # Apply dropout and layer norm
        trg = self.norm_layer(trg + self.dropout(attended_trg))

        # Apply positionwise feedforward
        poswise_trg = self.positionwise_feedforward(trg)

        # Apply dropout and layer norm
        decoded_trg = self.norm_layer(trg + self.dropout(poswise_trg))

        return decoded_trg


class Decoder(nn.Module):
    def __init__(
        self,
        target_vocab_size: int,
        word_embedding_size: int,
        num_layers: int,
        num_heads: int,
        pf_dim: int,
        dropout_value: float,
        device: torch.device,
        max_length: int = 100,
    ):
        """ Constructs the Decoder

            Parameters
            ----------
            target_vocab_size : int
            The vocab size in the target language
            word_embedding_size : int
            The word embedding size
            num_layers : int
            The number of DecoderLayer layers
            num_heads : int
            The number of heads for each attention layer
            pf_dim : int
            The dimension for the Positional layer
            dropout_value : float
            The dropout value for the outputs
            device : torch.device
            The device to run the model on
            max_length : int (optional)
            The max. length of each target sequence
        """
        super().__init__()
        self.device = device

        # Create the word and length embedding
        self.word_embedding = nn.Embedding(target_vocab_size, word_embedding_size)
        self.pos_embedding = nn.Embedding(max_length, word_embedding_size)

        # Create our list of decoder layers
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    word_embedding_size, num_heads, pf_dim, dropout_value, device
                )
                for _ in range(num_layers)
            ]
        )

        self.scale = torch.sqrt(torch.FloatTensor([word_embedding_size])).to(device)

        # The last linear layer for the decoder
        self.fc_out = nn.Linear(word_embedding_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """ Performs a forward prop. on the Decoder

            Parameters
            ----------
            trg : torch.LongTensor(N, S')
            A batch of target sequences
            enc_src : torch.LongTensor(N, S, H)
            A batch of source sequences that were encoded from the Encoder
            trg_mask : torch.LongTensor(N, S')
            The masks of each target sequence in the current batch
            src_mask : torch.LongTensor(N, S)
            The masks of each source sequence in the current batch

            Returns
            -------
            logits : torch.FloatTensor(N, target_vocab_size)
            The output probabilities that are not softmaxed
        """
        batch_size, trg_seq_len = trg.size()
        _, _, hidden_dim = enc_src.size()

        pos = (
            torch.arange(0, trg_seq_len)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .to(self.device)
        )
        assert pos.size() == (batch_size, trg_seq_len)

        trg = self.dropout(
            (self.word_embedding(trg) / self.scale) + self.pos_embedding(pos)
        )
        assert trg.size() == (batch_size, trg_seq_len, hidden_dim)

        for layer in self.decoder_layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        logits = self.fc_out(trg)

        return logits
