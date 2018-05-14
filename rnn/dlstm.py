import torch.nn as nn
from torch.autograd import Variable
import torch
from optfn.sigmoid_pow import sigmoid_pow
from optfn.swish import Swish
from optfn.drelu import drelu


class DLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, layer_norm=False, activation=nn.ReLU):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation

        layers = []
        for layer_idx in range(num_layers):
            last_layer = layer_idx == num_layers - 1
            in_features = hidden_size + input_size if layer_idx == 0 else hidden_size
            out_features = hidden_size * 4 if last_layer else hidden_size
            layers.append(nn.Linear(in_features, out_features, bias=not layer_norm))
            if layer_norm:
                layers.append(nn.LayerNorm(out_features))
            if not last_layer:
                layers.append(activation())
        self.net = nn.Sequential(*layers)
        self.hx_activation = self.activation()

    def forward(self, input, cell_state=None, reset_flags=None):
        if cell_state is None:
            cell_state = input.data.new(input.shape[1], self.hidden_size).zero_()
        else:
            cell_state = cell_state.squeeze(0)
        if reset_flags is not None:
            keep_flags = 1 - reset_flags
        outputs = []
        for step, x in enumerate(input):
            x = torch.cat([x, cell_state], 1)
            new_cx, tail = self.net(x).chunk(2, 1)
            new_hx, forget = tail.chunk(2, 1)
            forget = forget.sigmoid()
            retain = 1 - forget
            if reset_flags is not None:
                retain *= keep_flags[step].unsqueeze(-1)
            new_hx = self.hx_activation(new_hx)
            new_cx = drelu(new_cx)
            cell_state = (1 - retain) * new_cx + retain * cell_state
            outputs.append(new_hx)
        return torch.stack(outputs, 0), cell_state.unsqueeze(0)