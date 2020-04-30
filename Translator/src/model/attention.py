import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, hidden_size, device, num_heads=10, dropout_value=0.1):
        """ Constructs the MultiHeadedSelfAttention layer

            Parameters
            ----------
            hidden_size : int
            The hidden size
            device : torch.device
            The device running the model
            num_heads : int (optional)
            The number of heads to self-attend to
            dropout_value : float (optional)
            The dropout rate for the output layer
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0

        # Our weight matrixes
        self.to_keys = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.to_queries = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.to_values = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # The scale
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

        # The final linear transformation
        self.unify_heads = nn.Linear(self.hidden_size, self.hidden_size)

        # The dropout
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, value, key, query, mask=None):
        """ Performs a forward prop of the multi-headed attention layer

            Parameters
            ----------
            value : torch.Tensor(b, v_t, d)
            key : torch.Tensor(b, k_t, d)
            query : torch.Tensor(b, q_t, d)
            mask : torch.Tensor(b, 1, 1, d)
        """
        b, q_t, d = query.size()
        _, k_t, _ = key.size()
        _, v_t, _ = value.size()
        h = self.num_heads

        # Apply the weight matrix to x (Size: (b, t, h * k))
        queries = self.to_queries(query)
        keys = self.to_keys(key)
        values = self.to_values(value)

        assert queries.size() == query.size()
        assert keys.size() == key.size()
        assert values.size() == value.size()

        # Transform the matrix from (b, t, k) to (b, t, h, k // h)
        queries = queries.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = values.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # assert queries.size() == (b, h, t, k // h)
        # assert keys.size() == (b, h, t, k // h)
        # assert values.size() == (b, h, t, k // h)

        # Perform the dot product (Size: (b, h, t, t))
        w_prime = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / self.scale
        # assert w_prime.size() == (b, h, t, t)

        # Perform a mask (if needed)
        if mask is not None:
            w_prime = w_prime.masked_fill(mask == 0, -1e10)

        # assert w_prime.size() == (b, h, t, t)

        # Perform the softmax
        w = torch.nn.functional.softmax(w_prime, dim=-1)
        # assert w.size() == (b, h, t, t)

        # Apply dropout to the attention
        dropped_w = self.dropout(w)

        # Perform the self-attention
        y = torch.matmul(dropped_w, values)
        # assert y.size() == (b, h, t, k // h)

        # Perform the last linear transformation from (b, t, h, k) to (b, t, k)
        y = y.permute(0, 2, 1, 3).contiguous().view(b, -1, self.hidden_size)
        unified_y = self.unify_heads(y)

        # assert unified_y.size() == (b, t, k)

        return unified_y
