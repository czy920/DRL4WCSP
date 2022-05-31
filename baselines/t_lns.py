import random
from collections import namedtuple

import numpy as np
from numpy import argmin
from core.parser import parse
from core.utility import transpose
from core.bucket import Bucket
Node = namedtuple('Node', ['name', 'parent', 'all_parents', 'children', 'level', 'sep'])


class LNSEnv:
    def __init__(self, all_vars, all_functions):
        self.dom_size = dict()
        for name, dom in all_vars:
            self.dom_size[name] = dom
        self.dfs_tree = []
        adj_list = dict()
        self.function_table = dict()
        for data, v1, v2 in all_functions:
            if v1 not in adj_list:
                adj_list[v1] = list()
                self.function_table[v1] = dict()
            if v2 not in adj_list:
                adj_list[v2] = list()
                self.function_table[v2] = dict()
            self.function_table[v1][v2] = data
            self.function_table[v2][v1] = transpose(data)
            adj_list[v1].append(v2)
            adj_list[v2].append(v1)
        self.root = []
        self.assignment = dict()
        self.destroyed_variables = list()
        self.adj_list = adj_list

    def set_assignments(self, assignment):
        self.assignment = dict(assignment)
        self.destroyed_variables = [x for x in self.dom_size.keys() if x not in assignment]
        destroyed_variables = list(self.destroyed_variables)
        self.root.clear()
        self.dfs_tree.clear()
        while len(destroyed_variables) > 0:
            self._dfs(destroyed_variables)

    def _dfs(self, destroyed_variables, level=0, cur_node=None):
        if cur_node is None:
            cur_node = random.choice(destroyed_variables)
            self.root.append(cur_node)
            self.dfs_tree.append(dict())
            parent = None
            all_parents = set()
            sep = set()
        else:
            all_parents = set([x for x in self.adj_list[cur_node] if x in self.dfs_tree[-1]])
            sep = set(all_parents)
            parent = [x for x in all_parents if self.dfs_tree[-1][x].level == level - 1]
            assert len(parent) == 1
            parent = parent[0]
        self.dfs_tree[-1][cur_node] = Node(cur_node, parent, all_parents, set(), level, sep)
        for n in self.adj_list[cur_node]:
            if n not in self.dfs_tree[-1] and n in destroyed_variables:
                self.dfs_tree[-1][cur_node].children.add(n)
                self._dfs(destroyed_variables, level + 1, n)

        for n in self.dfs_tree[-1][cur_node].children:
            self.dfs_tree[-1][cur_node].sep.update(self.dfs_tree[-1][n].sep)
        self.dfs_tree[-1][cur_node].sep.discard(cur_node)
        destroyed_variables.remove(cur_node)


class T_LNS:
    def __init__(self, pth, p=.5, scale=10):
        self.env = LNSEnv(*parse(pth, scale=scale))
        self.p = p
        self.assignment = {x: random.randrange(self.env.dom_size[x]) for x in self.env.dom_size.keys()}
        self.scale = scale

    def _bucket_elim(self, i):
        level_var = dict()
        for var in self.env.dfs_tree[i].keys():
            level = self.env.dfs_tree[i][var].level
            if level not in level_var:
                level_var[level] = []
            level_var[level].append(var)
        buckets = dict()
        elim_buckets = dict()

        for level in range(max(level_var.keys()), -1, -1):
            for var in level_var[level]:
                joint_buckets = [elim_buckets[child] for child in self.env.dfs_tree[i][var].children]

                parent = self.env.dfs_tree[i][var].parent
                if parent is not None:
                    joint_buckets.append(Bucket.from_matrix(self.env.function_table[var][parent], var, parent))
                for n in self.env.adj_list[var]:
                    if n not in self.env.dfs_tree[i]:
                        assign = {n : self.assignment[n]}
                        b = Bucket.from_matrix(self.env.function_table[var][n], var, n)
                        b = b.reduce(assign)
                        joint_buckets.append(b)
                buckets[var] = Bucket.join(joint_buckets)
                elim_buckets[var] = buckets[var].proj(var)
        return buckets


    def step(self):
        assignment = random.sample(list(self.env.dom_size.keys()), int((1 - self.p) * len(self.env.dom_size)))
        assignment = {x: self.assignment[x] for x in assignment}
        self.env.set_assignments(assignment)
        new_assignment = dict(self.assignment)
        for i in range(len(self.env.dfs_tree)):
            root = self.env.root[i]
            if len(self.env.dfs_tree[i]) == 1:
                vec = [sum([self.env.function_table[root][n][v][self.assignment[n]] for n in self.env.adj_list[root]]) for v in range(self.env.dom_size[root])]
                new_assignment[root] = argmin(vec)
                continue
            pa = dict()
            buckets = self._bucket_elim(i)
            self._dfs_decision_making(i, root, pa, buckets)
            new_assignment.update(pa)
        return new_assignment

    def total_cost(self, assignment):
        cost = 0
        for v1 in self.env.function_table:
            for v2, func in self.env.function_table[v1].items():
                cost += func[assignment[v1]][assignment[v2]]
        return int(cost * self.scale / 2)

    def _dfs_decision_making(self, i, target_var, partial_assignment, buckets):
        if len(self.env.dfs_tree[i][target_var].children) == 0:
            vec = [sum([self.env.function_table[target_var][n][v][self.assignment[n]] for n in self.env.adj_list[target_var]]) for v
                   in range(self.env.dom_size[target_var])]
            partial_assignment[target_var] = argmin(vec)
            return
        action = self._make_decision(target_var, partial_assignment, buckets)
        partial_assignment[target_var] = action
        for c in self.env.dfs_tree[i][target_var].children:
            self._dfs_decision_making(i, c, partial_assignment, buckets)

    def _make_decision(self, target_var, partial_assignment, buckets):
        bucket = buckets[target_var]
        costs = [0] * self.env.dom_size[target_var]
        for val in range(self.env.dom_size[target_var]):
            partial_assignment[target_var] = val
            costs[val] = bucket.reduce(partial_assignment, True)
        val = np.argmin(costs)
        if val is list:
            val = val[0]
        return val
