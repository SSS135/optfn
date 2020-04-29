import torch
import torch.nn as nn
import torch.nn.functional as F


class MultilayerRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, activation=nn.ELU(), dropout=0.5):
        super(MultilayerRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout

        self.linears = nn.ModuleList([nn.Linear(
            input_size + 2 * hidden_size if i == 0 else hidden_size,
            hidden_size if i + 1 != hidden_layers else 3 * hidden_size) for i in range(hidden_layers)])
        self.cx_ln_in, self.cx_ln_out = nn.GroupNorm(1, hidden_size), nn.GroupNorm(1, hidden_size)

    def forward(self, input, hidden):
        hx, cx = hidden
        cx_norm = self.cx_ln_in(cx)
        x = torch.cat([input, hx, cx_norm], dim=1)
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i + 1 != len(self.linears):
                x = self.activation(x)
                x = F.dropout(x, self.dropout, self.training)
        forget, cell_out, hx = torch.chunk(x, 3, 1)
        cell_out = self.cx_ln_out(cell_out)
        forget, cell_out, hx = F.sigmoid(forget), self.activation(cell_out), self.activation(hx)
        cx = (1 - forget) * cx + forget * cell_out
        return hx, cx