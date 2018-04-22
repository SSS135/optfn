import torch.nn as nn
from torch.autograd import Variable
import torch
from optfn.layer_norm import LayerNorm1d
from optfn.sigmoid_pow import sigmoid_pow


class DLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, layer_norm=True, activation=nn.ReLU):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation

        layers = []
        for layer_idx in range(num_layers):
            in_features = hidden_size + input_size if layer_idx == 0 else hidden_size
            out_features = hidden_size * 3 if layer_idx == num_layers - 1 else hidden_size
            layers.append(nn.Linear(in_features, out_features, bias=not layer_norm))
            if layer_norm and layer_idx != num_layers - 1:
                layers.append(LayerNorm1d(out_features, affine=True))
                layers.append(activation())
        self.net = nn.Sequential(*layers)
        self.hx_activation = self.activation()

    def forward(self, input, cell_state=None, reset_flags=None):
        if cell_state is None:
            cell_state = Variable(input.data.new(input.shape[1], self.hidden_size).zero_())
        else:
            cell_state = cell_state.squeeze(0)
        outputs = []
        for step, x in enumerate(input):
            x = torch.cat([x, cell_state], 1)
            new_hx, new_cx, forget = self.net(x).chunk(3, 1)
            forget = sigmoid_pow(forget, 2)
            if reset_flags is not None:
                forget = 1 - (1 - forget) * (1 - reset_flags[step].unsqueeze(-1)) # 1 - retain * keep_flags
            cell_state = forget * new_cx + (1 - forget) * cell_state
            new_hx = self.hx_activation(new_hx)
            outputs.append(new_hx)
        return torch.stack(outputs, 0), cell_state.unsqueeze(0)