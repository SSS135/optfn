import time
import numpy as np
import torch
from torch.autograd import Variable
from .snes import SNES


class ESGrad:
    def __init__(self,
                 parameters,
                 pop_size,
                 noise_scale=1, ):
        self.parameters = list(parameters)
        self.pop_size = pop_size
        self.noise_scale = noise_scale
        self.tried_noises = []
        self.fitnesses = []

    def get_parameters(self):
        assert len(self.fitnesses) == len(self.tried_noises)
        new_noise = []
        new_params = []
        for cur_p in self.parameters:
            new_n = cur_p.data.new(cur_p.shape).normal_(0, self.noise_scale * cur_p.data.std())
            new_p = cur_p.data + new_n
            new_noise.append(new_n)
            new_params.append(new_p)
        self.tried_noises.append(new_noise)
        return new_params

    def rate_parameters(self, fitness):
        assert len(self.fitnesses) + 1 == len(self.tried_noises)
        self.fitnesses.append(fitness.data.mean())
        if len(self.fitnesses) == self.pop_size:
            self.compute_grad()
            self.fitnesses.clear()
            self.tried_noises.clear()
            return True
        else:
            return False

    def compute_grad(self):
        fitness = self.parameters[0].data.new(self.fitnesses)
        fitness = (fitness - fitness.mean()) / (fitness.std() + 1e-6)
        for param, noises in zip(self.parameters, zip(*self.tried_noises)):
            noises = torch.stack(noises, -1)
            grad = (noises * -fitness).mean(-1) / self.noise_scale
            param.grad = Variable(grad)


class SNESGrad:
    def __init__(self,
                 parameters,
                 pop_size,
                 lr=0.1,
                 noise_scale=0.1,
                 noise_step=0.1,
                 momentum=0.5):
        self.parameters = list(parameters)
        self.last_solution = np.concatenate([p.data.cpu().numpy().reshape(-1) for p in self.parameters])
        self.snes = SNES(self.last_solution.copy(), init_std=noise_scale, pop_size=pop_size,
                         std_step=noise_step, lr=lr, momentum=momentum)

    def get_parameters(self):
        solution = self.snes.get_single_sample()
        cur_idx = 0

        new_params = []
        for cur_p in self.parameters:
            new_p = solution[cur_idx: cur_idx + cur_p.numel()].reshape(cur_p.shape)
            new_p = cur_p.data.new(new_p)
            cur_idx += cur_p.numel()
            new_params.append(new_p)
        return new_params

    def rate_parameters(self, fitness):
        if isinstance(fitness, Variable):
            fitness = fitness.data
        if fitness.__class__.__name__.find('Tensor') != -1:
            fitness = fitness.mean()

        self.snes.rate_single_sample(fitness)

        if self.snes._rcv_sample_index == 0:
            self.compute_grad()
            return True
        else:
            return False

    def compute_grad(self):
        cur_idx = 0
        for cur_p in self.parameters:
            data = self.snes.cur_solution[cur_idx: cur_idx + cur_p.numel()].reshape(cur_p.shape)
            data = cur_p.data.new(data)
            cur_p.data.copy_(data)
            cur_idx += cur_p.numel()