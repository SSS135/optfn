import numpy as np
import numpy.random as rng


class CEM:
    def __init__(self, fitness_fn, init_solution, pop_size, elite_part=0.1, lr=1, init_std=1):
        assert init_solution.ndim == 1
        self.pop_size = pop_size
        self.elite_count = int(np.round(pop_size * elite_part))
        self.lr = lr
        self.cur_solution = init_solution
        self.fitness_fn = fitness_fn
        self.std = np.ones_like(self.cur_solution) * init_std if np.isscalar(init_std) else init_std
        self.best_solution = self.cur_solution
        self.best_fitness = None

    def learn(self, fitness_evals):
        epochs = int(np.ceil(fitness_evals/self.pop_size))
        for epoch in range(epochs):
            solutions = self.cur_solution + self.std * rng.randn(self.pop_size, self.cur_solution.size)
            fitness = np.empty(self.pop_size)
            for i in range(self.pop_size):
                fitness[i] = self.fitness_fn(solutions[i])
                if self.best_fitness is None or fitness[i] > self.best_fitness:
                    self.best_fitness = fitness[i]
                    self.best_solution = solutions[i]
            f_argsort = (-fitness).argsort()
            elite_solutions = solutions[f_argsort][0:self.elite_count]
            self.cur_solution = (1 - self.lr) * self.cur_solution + self.lr * elite_solutions.mean(axis=0)
            self.std = (1 - self.lr) * self.std + self.lr * elite_solutions.std(axis=0, ddof=1)
        return self.best_solution, self.best_fitness
