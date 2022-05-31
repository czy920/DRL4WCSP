from baselines.incomplete_algo import IncompleteAlgo
from core.utility import transpose
import os
import numpy as np
from time import perf_counter


class GDBA(IncompleteAlgo):
    def __init__(self, pth, scale_factor=10, max_cycle=1000):
        super().__init__(pth, scale_factor, max_cycle)
        self.min_table = dict()
        self.modifier_table = dict()

        for var in self.dom_size.keys():
            self.min_table[var] = dict()
            self.modifier_table[var] = dict()

        for data, v1, v2 in self.all_functions:
            table = []
            for val in range(self.dom_size[v1]):
                a = [0] * self.dom_size[v2]
                table.append(a)
            self.modifier_table[v1][v2] = table
            self.modifier_table[v2][v1] = transpose(table)
            self.min_table[v1][v2] = self._table_min_val(data)
            self.min_table[v2][v1] = self._table_min_val(data)

    def _table_min_val(self, data):
        min_val = 10**5
        for a in data:
            min_val = min(min_val, np.min(a))
        return min_val

    def _eff_cost(self, var, n, val, n_val):
        return self.function_table[var][n][val][n_val] * (self.modifier_table[var][n][val][n_val] + 1)

    def _eff_local_cost(self, sol, var, val):
        return sum([self._eff_cost(var, n, val, sol[n]) for n in self.adj_list[var]])

    def _gain_response(self, sol, var):
        costs = [self._eff_local_cost(sol, var, i) for i in range(self.dom_size[var])]
        best_val = np.argmin(costs)
        if best_val is list:
            best_val = best_val[0]
        gain = costs[sol[var]] - costs[best_val]
        return gain, best_val

    def _is_max_gain(self, all_gain, var):
        is_QLM = False
        is_max_g = False
        max_gain = max([all_gain[n] for n in self.adj_list[var]])
        if all_gain[var] > max_gain:
            is_max_g = True
        elif max_gain == 0:
            is_QLM = True
        return is_QLM, is_max_g

    def run(self):
        sol = {var:np.random.choice([i for i in range(self.dom_size[var])]) for var in self.dom_size.keys()}
        best_cost = 10**9
        current_cycle = 0

        cur_t = perf_counter()
        while current_cycle < self.max_cycle:
            current_cycle += 1
            best_cost = min(best_cost, self.total_cost(sol))
            all_gain = dict()
            all_best_val = dict()
            for var in self.dom_size.keys():
                all_gain[var], all_best_val[var] = self._gain_response(sol, var)

            for var in self.dom_size.keys():
                is_QLM, is_max_g = self._is_max_gain(all_gain, var)
                if is_QLM:
                    for n in self.adj_list[var]:
                        if self.function_table[var][n][sol[var]][sol[n]] > self.min_table[var][n]:
                            m = self.modifier_table[var][n]
                            for a in m:
                                for i in range(len(a)):
                                    a[i] += 1
                elif is_max_g:
                    sol[var] = all_best_val[var]
        return best_cost
