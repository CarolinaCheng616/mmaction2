import numpy as np
import torch
import torch.nn as nn


class FFLSTM(nn.Module):
    def __init__(self, num_layers, seq_len, feature_size, hidden_size, with_init_hc=False):
        super(FFLSTM, self).__init__()
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.with_init_hc = with_init_hc

        self.cells = []
        self.cells.append(nn.LSTMCell(feature_size, hidden_size))  # feature_size, hidden_size
        for i in range(1, num_layers):
            self.cells.append(nn.LSTMCell(hidden_size, hidden_size))  # hidden_size, hidden_size
        self.output_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # x.shape: [seq_len, batch, feature_size]
        _, batch, _ = x.shape
        device = x.device
        if self.with_init_hc:
            ht = torch.randn(self.num_layers, batch, self.hidden_size)
            ct = torch.randn(self.num_layers, batch, self.hidden_size)
        else:
            ht = torch.zeros(self.num_layers, batch, self.hidden_size)
            ct = torch.zeros(self.num_layers, batch, self.hidden_size)
        output = []  # seq_len, batch, hidden_size
        for i in range(self.seq_len):
            ht[0], ct[0] = self.cells[0](x[i], (ht[0], ct[0]))
            ft = self.output_layer(ht[0])  # batch, hidden_size
            for j in range(1, self.num_layers):
                ht[j], ct[j] = self.cells[j](ft, (ht[j], ct[j]))
                ft = self.output_layer(ht[j] + ft)
            output.append(ft)
        return torch.tensor(output)
