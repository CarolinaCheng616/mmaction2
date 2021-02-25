import numpy as np
import torch
import torch.nn as nn


class FFLSTM(nn.Module):
    def __init__(self, num_layers, seq_len, feature_size, hidden_size, with_init_hc=False, bias=False):
        super(FFLSTM, self).__init__()
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.with_init_hc = with_init_hc

        self.cells = []
        self.cells.append(nn.LSTMCell(feature_size, hidden_size, bias=bias))  # feature_size, hidden_size
        for i in range(1, num_layers):
            self.cells.append(nn.LSTMCell(hidden_size, hidden_size))  # hidden_size, hidden_size
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        torch.autograd.set_detect_anomaly(True)

    def forward(self, x):
        # x.shape: [seq_len, batch, feature_size]
        _, batch, _ = x.shape
        device = x.device
        for i in range(self.num_layers):
            self.cells[i] = self.cells[i].cuda()
        ht, ct, ft = [], [], []
        if self.with_init_hc:
            h0 = torch.randn(self.num_layers, batch, self.hidden_size)
            c0 = torch.randn(self.num_layers, batch, self.hidden_size)
        else:
            h0 = torch.zeros(self.num_layers, batch, self.hidden_size)
            c0 = torch.zeros(self.num_layers, batch, self.hidden_size)
        h0 = h0.to(device, dtype=torch.float32)
        c0 = c0.to(device, dtype=torch.float32)
        ht.append(h0)
        ct.append(c0)
        output = []  # seq_len, batch, hidden_size
        import pdb
        pdb.set_trace()
        for i in range(self.seq_len):
            hti, cti, fti = [], [], []
            hti0, cti0 = self.cells[0](x[i], (ht[-1][0], ct[-1][0]))
            fti0 = self.output_layer(hti0)  # batch, hidden_size
            hti.append(hti0)
            cti.append(cti0)
            fti.append(fti0)
            for j in range(1, self.num_layers):
                # ht[j], ct[j] = self.cells[j](ft, (ht[j], ct[j]))
                htij, ctij = self.cells[j](fti[-1], (ht[-1][j], ct[-1][j]))
                ftij = self.output_layer(htij + fti[-1])
                hti.append(htij)
                cti.append(ctij)
                fti.append(ftij)
            ht.append(hti)
            ct.append(cti)
            ft.append(fti)  # [[] * 5] * 32
            output.append(fti[-1].unsqueeze(0))  # batch, hidden_size
        return torch.cat(output)
