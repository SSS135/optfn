import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
import torch.nn.init
from torch.autograd import Variable


class RNNCellBase(nn.Module):
    def create_state(self, batch_size):
        cuda = next(self.parameters()).is_cuda
        h = torch.zeros((batch_size, self.hidden_size))
        return Variable(h.cuda() if cuda else h)


class nn_GRUCell(nn.GRUCell, RNNCellBase):
    pass


class GRUCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, activation=F.tanh, recurrent_activation=F.sigmoid):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.lin_r = nn.Linear(input_size + hidden_size, hidden_size, bias)
        self.lin_z = nn.Linear(input_size + hidden_size, hidden_size, bias)
        self.lin_hadd = nn.Linear(input_size + hidden_size, hidden_size, bias)

    def forward(self, x, h):
        cat = torch.cat((x, h), 1)
        z = self.recurrent_activation(self.lin_z(cat))
        r = self.recurrent_activation(self.lin_r(cat))
        rcat = torch.cat((x, r*h), 1)
        n = self.activation(self.lin_hadd(rcat))
        return z*h + (1 - z)*n


class DRUCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True,
                 out_activation=F.tanh, learn_activation=F.tanh, forget_activation=F.sigmoid):
        super(DRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.out_activation = out_activation
        self.learn_activation = learn_activation
        self.forget_activation = forget_activation
        self.lin_in = nn.Linear(input_size + hidden_size, hidden_size * 2, bias)
        self.lin_out = nn.Linear(input_size + hidden_size, hidden_size, bias)

    def forward(self, input, h):
        lin_in = self.lin_in(torch.cat((input, h), 1)).chunk(2, 1)
        learn = self.learn_activation(lin_in[0])
        forget_mult = self.forget_activation(lin_in[1])
        h = forget_mult*h + (1 - forget_mult)*learn
        lin_out = self.lin_out(torch.cat((input, h), 1))
        out = self.out_activation(lin_out)
        return out, h


class SRUCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True,
                 learn_activation=F.tanh, forget_activation=F.sigmoid):
        super(SRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.learn_activation = learn_activation
        self.forget_activation = forget_activation
        self.lin = nn.Linear(input_size + hidden_size, hidden_size*2, bias)

    def forward(self, x, h):
        lin = self.lin(torch.cat((x, h), 1)).chunk(2, 1)
        learn = self.learn_activation(lin[0])
        forget_mult = self.forget_activation(lin[1])
        h = forget_mult*h + (1 - forget_mult)*learn
        return h


class RNNBase(nn.Module):
    def create_state(self, batch_size):
        cuda = next(self.parameters()).is_cuda
        h = torch.zeros((self.num_layers, batch_size, self.hidden_size))
        return Variable(h.cuda() if cuda else h)


class nn_GRU(nn.GRU, RNNBase):
    pass


class RNN(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers,
                 cell_type=SRUCell, **kwargs):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lin_out = nn.Linear(input_size + hidden_size * (num_layers + 1), hidden_size,
                                 kwargs['bias'] if 'bias' in kwargs else True)

        self.cells = []
        for i in range(num_layers):
            cell = cell_type(input_size if i == 0 else hidden_size, hidden_size,  **kwargs)
            self.cells.append(cell)
        self.cells = nn.ModuleList(self.cells)

    def forward(self, input, h):
        h = [v.squeeze(0) for v in torch.chunk(h, h.size(0), 0)]
        outputs = []
        for seq_index in range(input.size(0)):
            x = input[seq_index]
            for cell_idx in range(self.num_layers):
                cell = self.cells[cell_idx]
                x = cell(x, h[cell_idx])
                x, h[cell_idx] = x if type(x) is tuple else (x, x)
            outputs.append(x)

        outputs = torch.cat([o.unsqueeze(0) for o in outputs], 0)
        h = torch.cat(h, 0)
        return outputs, h


class RNNWrapNet(RNNBase):
    def __init__(self, model, num_module_outputs, num_outputs):
        super(RNNWrapNet, self).__init__()
        self.model = model
        self.lin = nn.Linear(num_module_outputs, num_outputs)

    def forward(self, input, h):
        x, h = self.model(input, h)
        x = [self.lin(v.squeeze(0)) for v in torch.chunk(x, x.size(0), 0)]
        x = torch.cat([v.unsqueeze(0) for v in x], 0)
        return x, h

    def create_state(self, batch_size):
        return self.model.create_state(batch_size)