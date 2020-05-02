import torch
from torch import nn
import torch.nn.functional as F

class Seq2VecNN(nn.Module):
  def __init__(self, vocab_size, num_classes, num_neurons_per_layer=[1000, 1000, 1000]):

    super().__init__()

    self.vocab_size = vocab_size
    self.num_classes = num_classes

    layers = []

    prev_layer_count = vocab_size
    for num_neurons in num_neurons_per_layer:
      layers.append(nn.Linear(prev_layer_count, num_neurons))
      layers.append(nn.ReLU())
      prev_layer_count = num_neurons

    self.feedforward_layer = nn.Sequential(*layers)

    self.output_layer = nn.Linear(prev_layer_count, self.num_classes)
      
  def forward(self, F):
    ''' Given a batch of sequences, and its sequence lengths, output a softmax of
        its class

        Parameters
        ----------
        F : torch.LongTensor (N, self.vocab_size)
            It is a batch of bag-of-words

        Returns
        -------
        logits_t : torch.FloatTensor (N, self.vocab_size)
            It is a un-normalized distribution over the classes for the n-th sequence:
            Pr_b(i) = softmax(logits_t[i]) for i in self.num_classes
    '''
    x = self.feedforward_layer(F)
    return self.output_layer(x)
        