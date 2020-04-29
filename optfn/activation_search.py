import numpy as np
import torch
import visdom
from .namedtuple_with_defaults import namedtuple_with_defaults
from .. import optfn


LUConf = namedtuple_with_defaults(
    'LUConf',
    'size, xmin, xmax, l2_reg, smooth_reg, src_activation',
    dict(size=7, xmin=-3, xmax=3, l2_reg=3e-5, smooth_reg=5e-4, src_activation=lambda x: x))


class ActivationSearcher:
    def __init__(self, fitness_fn, lu_conf=LUConf(),
                 xnes_iterations=1500, sigma=0.3, lr=0.1,
                 xnes_log_interval=1, log_each_fitness=True, cuda=True,
                 smooth_iters=10, smooth_delta=0.05):
        self.fitness_fn = fitness_fn
        self.lu_conf = lu_conf
        self.xnes_iterations = xnes_iterations
        self.sigma = sigma
        self.lr = lr
        self.xnes_log_interval = xnes_log_interval
        self.log_each_fitness = log_each_fitness
        self.cuda = cuda
        self.smooth_iters = smooth_iters
        self.smooth_delta = smooth_delta
        self.cur_xnes = -1
        self.cur_xnes_fitness_iter = 0
        self.total_fitness_iters = 0
        self.xnes = None
        self.best_solution = None
        self.max_r = None
        self.viz = visdom.Visdom()

        if callable(self.lu_conf.src_activation):
            src_activation = optfn.LU.replicate(
                self.lu_conf.src_activation, self.lu_conf.xmin, self.lu_conf.xmax,
                self.lu_conf.size, self.cuda).y_data.cpu().numpy()
            self.lu_conf = self.lu_conf._replace(src_activation=src_activation)
        elif len(self.lu_conf.src_activation) != self.lu_conf.size:
            print('ActivationSearcher: mismatch in activation size')
            self.lu_conf = self.lu_conf._replace(size=self.lu_conf.src_activation.size)

    def search_for_activation(self):
        self.cur_xnes_fitness_iter = 0
        self.cur_xnes += 1
        self.xnes = optfn.XNES(self._do_iteration, init_solution=self.lu_conf.src_activation,
                               sigma=self.sigma, lr=self.lr, adaptive_lr=False)
        sol, r = self.xnes.learn(self.xnes_iterations, log_interval=self.xnes_log_interval)

        if self.max_r is None or self.max_r < r:
            self.max_r = r
            self.best_solution = sol
            print(self.cur_xnes, 'new best sol', r, sol)
        else:
            print(self.cur_xnes, 'cur sol', r, sol)

        return sol, r

    def search_for_activation_with_restarts(self, restarts):
        steps = []
        for _ in range(restarts):
            sol, r = self.search_for_activation()
            steps.append((sol, r))
            self.lu_conf = self.lu_conf._replace(src_activation=self.best_solution)
        return self.best_solution, self.max_r, steps

    def _do_iteration(self, w):
        w_diff = self._get_diff(w)
        l2_loss = (w ** 2).mean() * self.lu_conf.l2_reg
        smooth_loss = np.abs(w_diff).mean() * self.lu_conf.smooth_reg
        reg_loss = l2_loss + smooth_loss

        w = torch.Tensor(w)
        if self.cuda:
            w = w.cuda()
        activation = optfn.LU(w[:self.lu_conf.size], self.lu_conf.xmin, self.lu_conf.xmax).to_learned()

        fitness = self.fitness_fn(activation)

        lploty = np.stack((self.xnes.cur_solution[:self.lu_conf.size], activation.y.data.cpu().numpy()), axis=1)
        self.viz.line(lploty, win='learn_activation', opts=dict(legend=['mean', 'w']))
        self._update_trace(self.cur_xnes_fitness_iter, fitness, 'loss_total', self.cur_xnes_fitness_iter == 0)
        self._update_trace(self.total_fitness_iters, fitness, 'loss_cur', self.total_fitness_iters == 0)
        self.cur_xnes_fitness_iter += 1
        self.total_fitness_iters += 1
        if self.log_each_fitness:
            print('{:.6f} {:.6f}'.format(fitness, reg_loss))

        return fitness - reg_loss

    def _update_trace(self, x, y, win, create_new):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if create_new:
            self.viz.line(X=x, Y=y, win=win)
        else:
            self.viz.updateTrace(X=x, Y=y, win=win)

    def _get_diff(self, w):
        w_diff = w[:-1] - w[1:]
        w_diff = w_diff[1:] - w_diff[:-1]
        return w_diff

