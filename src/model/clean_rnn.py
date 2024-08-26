import torch
import torch.nn as nn

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.8, batch_norm=True):
        super(LinearBlock, self).__init__()
        half_dim = input_dim // 2
        self.linear = nn.Linear(input_dim, half_dim)
        self.linear2 = nn.Linear(half_dim, output_dim)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(half_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        if self.batch_norm:
            x = x.permute(0, 2, 1)
            x = self.batch_norm(x)
            x = x.permute(0, 2, 1)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class LSTMSeq(nn.Module):
    def __init__(self, input_dim, input_ts, output_ts, hidden_dim=32, dropout=0.8):
        super(LSTMSeq, self).__init__()
        self.input_dim = input_dim
        self.input_ts = input_ts
        self.output_ts = output_ts
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.encoder = nn.LSTM(input_size=self.input_dim,
                               hidden_size=hidden_dim,
                               num_layers=2,
                               dropout=dropout,
                               batch_first=True)