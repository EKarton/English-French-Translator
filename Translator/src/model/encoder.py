import torch
from torch import nn
import torch.nn.functional as F

from .attention import MultiHeadedSelfAttention
from .positionwise import PositionwiseFeedforwardLayer


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        pf_dim: int,
        dropout_value: float,
        device: torch.device,
    ):
        """ Constructs the EncoderLayer

            Parameters
            ----------
            hidden_size : int
            The hidden size
            num_heads : int
            The number of heads in the self-attention layer
            pf_dim : int
            The dimension for the PositionwiseFeedforwardLayer
            dropout_value : float
            The dropout value
            device : torch.device
            The device to run the model on
        """
        super().__init__()

        self.attention_layer = MultiHeadedSelfAttention(
            hidden_size, device, num_heads=num_heads, dropout_value=dropout_value
        )

        self.norm_layer = nn.LayerNorm(hidden_size)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hidden_size, pf_dim, dropout_value
        )
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, src, src_mask):
        """ Performs a forward propagation of the Encoder layer

            Parameters
            ----------
            src : torch.Longtensor(N, S, H)
            A batch of source sequences
            src_mask : torch.LongTensor(N, 1, 1, S)
            The masks of each source sequence in the current batch

            Returns
            -------
            encoded_src : torch.Tensor(N, S, H)
            A batch of source sequences from this EncoderLayer
        """

        # Apply attention
        attended_src = self.attention_layer(src, src, src, src_mask)

        # Apply dropout and normalization
        new_src = self.norm_layer(src + self.dropout(attended_src))

        # Apply positionalwise feedforward layer
        pos_src = self.positionwise_feedforward(new_src)

        # Apply dropout and layer normalization
        encoded_src = self.norm_layer(new_src + self.dropout(pos_src))

        assert encoded_src.size() == src.size()

        return encoded_src


class Encoder(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        word_embedding_size: int,
        num_layers: int,
        num_heads: int,
        pf_dim: int,
        dropout_value: float,
        device: torch.device,
        max_length: int = 100,
    ):
        """ Constructs the Encoder

            Parameters
            ----------
            source_vocab_size : int
            The vocab size in the source language
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

        # Create our word and position embeddings
        self.word_embedding = nn.Embedding(source_vocab_size, word_embedding_size)
        self.pos_embedding = nn.Embedding(max_length, word_embedding_size)

        # Create our layers of Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    word_embedding_size, num_heads, pf_dim, dropout_value, device
                )
                for _ in range(num_layers)
            ]
        )

        # Create our dropout and scaler
        self.dropout = nn.Dropout(dropout_value)
        self.scale = torch.sqrt(torch.FloatTensor([word_embedding_size])).to(device)

    def forward(self, src, src_mask):
        """ Performs a forward propagation on the Encoder

            Parameters
            ----------
            src : torch.LongTensor(N, S)
            A batch of source sequences
            src_mask : torch.LongTensor(N, 1, 1, S)
            The masks of each source sequence in the current batch

            Returns
            -------
            encoded_src : torch.FloatTensor(N, S)
            A batch of encoded source sequences
        """
        batch_size, seq_len = src.size()

        # Get the position embeddings for each src seq.
        src_pos = (
            torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )
        src_pos = self.pos_embedding(src_pos)

        # Get word embeddings for each src seq
        src = self.word_embedding(src)

        # Combine the word and position embeddings
        src = src * self.scale + src_pos

        # Apply dropout
        src = self.dropout(src)
        # Obtain the encoded src seq
        encoded_src = src
        for layer in self.encoder_layers:
            encoded_src = layer(encoded_src, src_mask)

        return encoded_src
