import random
from baselines.incomplete_algo import IncompleteAlgo
import numpy as np


class SA(IncompleteAlgo):
    def __init__(self, pth, scale_factor=10, p=0.8, max_cycle=1000):
        super().__init__(pth, scale_factor, max_cycle)
        self.p = p

    def run(self):
        sol = {var: np.random.choice([i for i in range(self.dom_size[var])]) for var in self.dom_size.keys()}
        best_cost = 10 ** 9
        current_cycle = 0
        while current_cycle < self.max_cycle:
            current_cycle += 1
            best_cost = min(best_cost, self.total_cost(sol))
            new_sol = dict(sol)
            for var in self.dom_size.keys():
                if random.random() < self.p:
                    new_sol[var] = self.best_response(sol, var)
            sol = new_sol
        return best_cost
