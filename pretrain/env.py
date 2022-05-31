import random
from collections import  namedtuple
import torch
from core.utility import transpose

Node = namedtuple('Node', ['name', 'parent', 'all_parents', 'children', 'level', 'sep'])
x_embed = [1, 0, 0, 0]
c_embed = [0, 1, 0]
f_embed = [0, 0, 0, 1]

class Environment:
    def __init__(self, all_vars, all_functions, is_valid, device='cpu'):
        self.dom_size = dict()
        for name, dom in all_vars:
            self.dom_size[name] = dom
        self.dfs_tree = dict()
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
        self.root = None
        self.device = device
        if is_valid:
            max_degree = 0
            max_degree_id = -1
            for var in adj_list.keys():
                if len(adj_list[var]) > max_degree:
                    max_degree = len(adj_list[var])
                    max_degree_id = var
            self._dfs(adj_list, cur_node=max_degree_id)
        else:
            self._dfs(adj_list)

    def _dfs(self, adj_list, level=0, cur_node=None):
        if cur_node is None:
            cur_node = random.choice(list(adj_list.keys()))
        if level == 0:
            self.root = cur_node
            parent = None
            all_parents = set()
            sep = set()
        else:
            all_parents = set([x for x in adj_list[cur_node] if x in self.dfs_tree])
            sep = set(all_parents)
            parent = [x for x in all_parents if self.dfs_tree[x].level == level - 1]

            assert len(parent) == 1
            parent = parent[0]

        self.dfs_tree[cur_node] = Node(cur_node, parent, all_parents, set(), level, sep)
        for n in adj_list[cur_node]:
            if n not in self.dfs_tree:
                self.dfs_tree[cur_node].children.add(n)
                self._dfs(adj_list, level + 1, n)

        for n in self.dfs_tree[cur_node].children:
            self.dfs_tree[cur_node].sep.update(self.dfs_tree[n].sep)
        self.dfs_tree[cur_node].sep.discard(cur_node)

    def act(self, partial_assignment, target_var):
        cost = sum([self.function_table[target_var][p][partial_assignment[target_var]][partial_assignment[p]] for p in self.dfs_tree[target_var].all_parents])
        return -cost

    def build_graph(self, target_var, partial_assignment):
        checksum = sum([0 if sep in partial_assignment else 1 for sep in self.dfs_tree[target_var].sep])
        assert checksum == 0
        x = []
        edge_index = [[], []]
        node_index = dict()
        all_function_node_index = []
        self._dfs_build_graph(partial_assignment, target_var, x, edge_index, node_index, all_function_node_index)
        return torch.tensor(x, dtype=torch.float32, device=self.device), \
               torch.tensor(edge_index, dtype=torch.long, device=self.device), all_function_node_index, \
               [node_index[target_var] + i for i in range(self.dom_size[target_var])]

    def _dfs_build_graph(self, partial_assignment, cur_var, x, edge_index, node_index, all_function_node_index):
        node_index[cur_var] = len(x)
        src, dest = edge_index
        for val in range(self.dom_size[cur_var]):
            x.append(x_embed)
        for p in self.dfs_tree[cur_var].all_parents:
            f_idx = len(x)
            all_function_node_index.append(f_idx)
            x.append(f_embed)
            if p in partial_assignment:
                for val in range(self.dom_size[cur_var]):
                    idx = len(x)
                    x.append(c_embed + [self.function_table[cur_var][p][val][partial_assignment[p]]])
                    src.append(idx)
                    dest.append(f_idx)

                    src.append(idx)
                    dest.append(node_index[cur_var] + val)
            else:
                for val_p in range(self.dom_size[p]):
                    for val in range(self.dom_size[cur_var]):
                        idx = len(x)
                        x.append(c_embed + [self.function_table[cur_var][p][val][val_p]])
                        src.append(idx)
                        dest.append(f_idx)

                        src.append(idx)
                        dest.append(node_index[p] + val_p)

                        src.append(node_index[cur_var] + val)
                        dest.append(idx)
        for c in self.dfs_tree[cur_var].children:
            self._dfs_build_graph(partial_assignment, c, x, edge_index, node_index, all_function_node_index)