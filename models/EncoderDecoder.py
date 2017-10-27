import constants as c
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Coder(nn.Module):
    def __init__(self):
        super(Coder, self).__init__()

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.directions * self.n_layers, batch_size, self.hidden_size))
        if c.use_cuda:
            return hidden.cuda()
        else:
            return hidden


class Encoder(Coder):
    def __init__(self, input_size, hidden_size, emb_weights=None, directions=1, n_layers=1, dropout_p=0.1):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.hidden_size = hidden_size
        self.directions = directions

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        bidirectional = self.directions == 2
        self.gru = nn.GRU(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=self.dropout_p)

        if emb_weights is not None:
            self.embedding.weight = nn.Parameter(emb_weights)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        output = embedded.unsqueeze(1)
        output, hidden = self.gru(output, hidden)
        return output, hidden


class Decoder(Coder):
    def __init__(self,
                 hidden_size,
                 output_size,
                 emb_weights=None,
                 directions=1,
                 n_layers=1,
                 dropout_p=0.1,
                 max_length=30):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.directions = directions
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * (2 + directions - 1), self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        bidirectional = self.directions == 2
        self.gru = nn.GRU(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=self.dropout_p)
        self.out = nn.Linear(self.hidden_size * directions, self.output_size)

        if emb_weights is not None:
            self.embedding.weight = nn.Parameter(emb_weights)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded.squeeze(1), hidden.squeeze(0).sum(0)), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        output = torch.cat((embedded.squeeze(1), attn_applied), 1)
        output = self.attn_combine(output).unsqueeze(1)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output).squeeze(1))
        return output, hidden, attn_weights
