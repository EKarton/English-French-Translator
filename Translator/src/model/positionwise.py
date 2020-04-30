import torch
from torch import nn
import torch.nn.functional as F


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_size: int, pf_dim: int, dropout_value: float):
        """ Constructs the PositionwiseFeedforwardLayer

            Parameters
            ----------
            hidden_size : int
            The hidden size
            pf_dim : int
            The dimension for the hidden layer
            dropout_value : float
            The dropout value for the output layer
        """
        super().__init__()

        self.fc_1 = nn.Linear(hidden_size, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hidden_size)
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        """ Performs forward prop of the PositionwiseFeedforwardLayer

            Parameters
            ----------
            x : torch.FloatTensor(N, S, H)
            A batch of sequences with word embeddings

            Returns
            -------
            x : torch.FloatTensor(N, S, H)
            A batch of transformed sequences with word embeddings
        """
        new_x = self.fc_1(x)
        new_x = torch.relu(new_x)
        new_x = self.dropout(new_x)
        new_x = self.fc_2(new_x)

        assert new_x.size() == x.size()

        return x
