import numpy as np
import numpy.random as rng


def get_ranks(n):
    indexes = np.arange(1, n + 1)
    ranks = np.maximum(0, np.log(n / 2 + 1) - np.log(indexes))
    return ranks / ranks.sum() - 1 / n


class SNES:
    def __init__(self, fitness_fn, dim=None, adaptive_lr=True, as_factor=0.3,
                 init_solution=None, pop_size=None, sigma=0.5, lr=0.1, sigma_step=None):
        self.dim = dim if dim is not None else init_solution.size
        self.fitness_fn = fitness_fn
        self.adaptive_lr = adaptive_lr
        self.sigma = sigma
        self.lr = lr
        self.mean_step_lim = (0.001, 1)
        self.as_factor = as_factor
        self.sigma_step = (3 + np.log(self.dim))/(20*np.sqrt(self.dim)) if sigma_step is None else sigma_step
        self.pop_size = int(round(4 + 3 * np.log(self.dim))) if pop_size is None else pop_size
        self.pop_size += self.pop_size % 2
        self.R_rank = get_ranks(self.pop_size)
        self.cur_solution = rng.randn(self.dim) if init_solution is None else init_solution
        self.grad_mean = np.zeros_like(self.cur_solution)
        self.max_R = None
        self.best_solution = self.cur_solution

    def learn(self, fitness_evals, log_interval=None):
        iters = int(max(1, np.ceil(fitness_evals / self.pop_size)))
        for iter in range(iters):
            noise = self.get_noise()
            samples, delta_mean_weights = self.get_samples(noise, self.grad_mean)
            R, sort_idx = self.evaluate(samples)
            do_log = log_interval is not None and iter % log_interval == 0
            if self.max_R is None or R[sort_idx[0]] > self.max_R:
                self.max_R = R[sort_idx[0]]
                self.best_solution = samples[sort_idx[0]]
                if do_log:
                    print((iter + 1) * self.pop_size, 'new best', self.max_R)
            elif do_log:
                print((iter + 1) * self.pop_size, 'cur iter best', R.max())
            if self.adaptive_lr:
                self.lr = self.get_updated_mean_step(delta_mean_weights, sort_idx)
            self.grad_mean, grad_sigma = self.get_grads(noise, sort_idx)
            self.cur_solution += self.lr * self.grad_mean
            self.sigma = self.get_updated_sigma(grad_sigma)
        return self.best_solution, self.max_R

    def get_updated_mean_step(self, delta_mean_weights, sort_idx):
        rank = np.linspace(1, -1, self.pop_size)
        new_step = self.lr * (1 + (rank * delta_mean_weights[sort_idx]).mean())
        new_step = np.clip(new_step, self.mean_step_lim[0], self.mean_step_lim[1])
        return new_step

    def get_updated_sigma(self, grad_sigma):
        sigma = self.sigma * np.exp(self.sigma_step / 2 * grad_sigma)
        return sigma

    def get_grads(self, noise, sort_idx):
        noise = noise[sort_idx]
        grad_mean = np.dot(self.R_rank, noise)
        grad_sigma = np.dot(self.R_rank, noise ** 2 - 1)
        return grad_mean, grad_sigma

    def get_noise(self):
        noise = rng.randn(self.pop_size, self.dim)
        noise[self.pop_size // 2:] = -noise[:self.pop_size // 2]
        return noise

    def get_samples(self, noise, grad_mean):
        delta_mean_weights = rng.randn(self.pop_size) * self.as_factor
        pos_mask = delta_mean_weights >= 0
        neg_mask = delta_mean_weights < 0
        delta_mean_weights[pos_mask] = 1 + delta_mean_weights[pos_mask]
        delta_mean_weights[neg_mask] = 1/(1 - delta_mean_weights[neg_mask])

        samples = np.empty_like(noise)
        for i in range(self.pop_size):
            samples[i] = self.cur_solution + self.lr * grad_mean * delta_mean_weights[i] + self.sigma * noise[i]
        return samples, delta_mean_weights

    def evaluate(self, samples):
        R = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            R[i] = self.fitness_fn(samples[i])
        sort_idx = (-R).argsort()
        return R, sort_idx


class XNES:
    def __init__(self, fitness_fn, dim=None, adaptive_lr=True, as_factor=0.3,
                 init_solution=None, pop_size=None, sigma=0.5, lr=0.1, sigma_step=None):
        self.dim = dim if dim is not None else init_solution.size
        self.fitness_fn = fitness_fn
        self.adaptive_lr = adaptive_lr
        self.sigma = sigma
        self.B = np.eye(self.dim)/self.sigma
        self.lr = lr
        self.lr_lim = (0.001, 1)
        self.as_factor = as_factor
        self.sigma_step = (9 + 3 * np.log(self.dim)) / (20 * np.power(self.dim, 1.5)) if sigma_step is None else sigma_step
        self.pop_size = int(round(4 + 3 * np.log(self.dim))) if pop_size is None else pop_size
        self.pop_size = max(2, self.pop_size + self.pop_size % 2)
        self.R_rank = np.linspace(1, -1, self.pop_size)
        self.R_rank = (self.R_rank - self.R_rank.mean()) / self.R_rank.std()
        self.cur_solution = rng.randn(self.dim) if init_solution is None else init_solution
        self.delta_mean = np.zeros_like(self.cur_solution)
        self.max_R = None
        self.best_solution = self.cur_solution

    def learn(self, fitness_evals, log_interval=None):
        iters = int(max(1, np.ceil(fitness_evals / self.pop_size)))
        for iter in range(iters):
            noise = self.get_noise()
            samples, delta_mean_weights = self.get_samples(noise, self.delta_mean)
            R, sort_idx = self.evaluate(samples)
            do_log = log_interval is not None and iter % log_interval == 0
            if self.max_R is None or R[sort_idx[0]] > self.max_R:
                self.max_R = R[sort_idx[0]]
                self.best_solution = samples[sort_idx[0]]
                if do_log:
                    print((iter + 1) * self.pop_size, '!!!!!!!! new best', self.max_R)
            elif do_log:
                print((iter + 1) * self.pop_size, 'cur iter best', R.max())
            if self.adaptive_lr:
                self.lr = self.get_updated_mean_step(delta_mean_weights, sort_idx)
            grad_err, _, grad_sigma, grad_B = self.get_grads(noise, sort_idx)
            self.cur_solution += self.lr * self.delta_mean
            self.delta_mean, self.sigma, self.B = self.get_updated_params(grad_err, grad_sigma, grad_B)
        return self.best_solution, self.max_R

    def get_updated_mean_step(self, delta_mean_weights, sort_idx):
        rank = np.linspace(1, -1, self.pop_size)
        new_step = self.lr * (1 + (rank * delta_mean_weights[sort_idx]).mean())
        new_step = np.clip(new_step, self.lr_lim[0], self.lr_lim[1])
        return new_step

    def get_updated_params(self, grad_err, grad_sigma, grad_B):
        delta_mean = self.sigma * np.dot(self.B, grad_err)
        sigma = self.sigma * np.exp(self.sigma_step / 2 * grad_sigma)
        B = self.B*np.exp(self.sigma_step / 2 * grad_B)
        return delta_mean, sigma, B

    def get_grads(self, noise, sort_idx):
        noise = noise[sort_idx]
        grad_err = np.dot(self.R_rank, noise)
        grad_M = np.zeros((self.dim, self.dim))
        for k in range(self.pop_size):
            grad_M += self.R_rank[k]*(np.dot(noise[k], noise[k].T) - np.eye(self.dim))
        grad_sigma = np.trace(grad_M) / self.dim
        grad_B = grad_M - grad_sigma*np.eye(self.dim)
        return grad_err, grad_M, grad_sigma, grad_B

    def get_noise(self):
        noise = rng.randn(self.pop_size, self.dim)
        noise[self.pop_size // 2:] = -noise[:self.pop_size // 2]
        return noise

    def get_samples(self, noise, delta_mean):
        delta_mean_weights = rng.randn(self.pop_size) * self.as_factor
        pos_mask = delta_mean_weights >= 0
        neg_mask = delta_mean_weights < 0
        delta_mean_weights[pos_mask] = 1 + delta_mean_weights[pos_mask]
        delta_mean_weights[neg_mask] = 1/(1 - delta_mean_weights[neg_mask])

        samples = np.empty_like(noise)
        for i in range(self.pop_size):
            samples[i] = self.cur_solution + self.lr * delta_mean_weights[i] * delta_mean + \
                         self.sigma * np.dot(self.B.T, noise[i])
        return samples, delta_mean_weights

    def evaluate(self, samples):
        R = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            R[i] = self.fitness_fn(samples[i])
        sort_idx = (-R).argsort()
        return R, sort_idx
