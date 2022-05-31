from  core.parser import parse
from core.utility import transpose
import numpy as np


class IncompleteAlgo:
    def __init__(self, pth, scale_factor, max_cycle):
        self._parse_problem(pth, scale_factor)
        self.max_cycle = max_cycle
        self.scale_factor = scale_factor

    def _parse_problem(self, pth, scale_factor):
        self.all_vars, self.all_functions = parse(pth, scale=scale_factor)
        self.dom_size = dict()
        for name, dom in self.all_vars:
            self.dom_size[name] = dom
        self.adj_list = dict()
        self.function_table = dict()
        for data, v1, v2 in self.all_functions:
            if v1 not in self.adj_list:
                self.adj_list[v1] = list()
                self.function_table[v1] = dict()
            if v2 not in self.adj_list:
                self.adj_list[v2] = list()
                self.function_table[v2] = dict()
            self.function_table[v1][v2] = data
            self.function_table[v2][v1] = transpose(data)
            self.adj_list[v1].append(v2)
            self.adj_list[v2].append(v1)


    def local_cost(self, sol, var, val):
        return sum([self.function_table[var][n][val][sol[n]] for n in self.adj_list[var]])

    def total_cost(self, sol):
        tc = 0
        for var in self.dom_size.keys():
            tc += self.local_cost(sol, var, sol[var])
        return tc * self.scale_factor / 2

    def best_response(self, sol, var):
        costs = [self.local_cost(sol, var, i) for i in range(self.dom_size[var])]
        best_val = np.argmin(costs)
        if best_val is list:
            best_val = best_val[0]
        return best_val

    @classmethod
    def normalize(cls, data):
        max_data = max(data) + 1
        data = [max_data - val for val in data]
        normal_data = [val/sum(data) for val in data]
        return normal_data

    @classmethod
    def top_k(cls, data, k):
        sorted_data = sorted(enumerate(data), key=lambda x: x[1])
        idx = [i[0] for i in sorted_data]
        return idx[:k]

    @classmethod
    def rnd_select_k(cls, data, k):
        idx = np.random.choice([i for i in range(len(data))], size=k, replace=False, p=cls.normalize(data))
        best_id = np.argmin(data)
        if best_id is list:
            best_id = best_id[0]
        if best_id not in idx:
            idx[0] = best_id
        return idx