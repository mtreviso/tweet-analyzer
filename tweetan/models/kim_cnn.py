import torch
from torch import nn

class KimCNN(nn.Module):

    def __init__(self, vocab_size, emb_size, hidden_size, nb_labels):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.nb_labels = nb_labels

        self.emb_layer = nn.Embedding(vocab_size, emb_size)
        # TODO: cnn layers here
        self.output_layer = nn.Linear(hidden_size, nb_labels)
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        # TODO
        pass
        return x

