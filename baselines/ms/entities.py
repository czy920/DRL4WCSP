from utilities import elementwise_add, argmin
from core.parser import parse

class Node:
    def __init__(self, name):
        self.name = name
        self.incoming_msg = dict()

    def compute_msgs(self):
        pass

    def __repr__(self):
        return self.name


class VariableNode(Node):
    damp_factor = 0
    op = argmin

    def __init__(self, name, dom_size):
        super().__init__(name)
        self.dom_size = dom_size
        self.prev_sent = dict()
        self.val_idx = -1
        self.neighbors = dict()  # neighbor id: neighbor node

    def register_neighbor(self, neighbor):
        self.neighbors[neighbor.name] = neighbor

    def compute_msgs(self):
        for nei in self.neighbors:
            msg = [0] * self.dom_size
            for other_nei in self.neighbors:
                if other_nei == nei:
                    continue
                if other_nei not in self.incoming_msg:
                    continue
                msg = elementwise_add(msg, self.incoming_msg[other_nei])
            norm = min(msg)
            msg = [x - norm for x in msg]
            if nei in self.prev_sent and 0 < VariableNode.damp_factor < 1:
                prev = self.prev_sent[nei]
                msg = [(1 - VariableNode.damp_factor) * x + VariableNode.damp_factor * y for x, y in zip(msg, prev)]
                norm = min(msg)
                msg = [x - norm for x in msg]
            self.prev_sent[nei] = list(msg)
            self.neighbors[nei].incoming_msg[self.name] = msg

    def make_decision(self):
        belief = [0] * self.dom_size
        for nei in self.neighbors:
            if nei in self.incoming_msg:
                belief = elementwise_add(belief, self.incoming_msg[nei])
        self.val_idx = VariableNode.op(belief)

class FunctionNode(Node):
    op = min

    def __init__(self, name, matirx, row_vn, col_vn):
        super().__init__(name)
        self.matrix = matirx
        self.row_vn = row_vn
        self.col_vn = col_vn
        self.row_vn.register_neighbor(self)
        self.col_vn.register_neighbor(self)

    def compute_msgs(self):
        for nei in [self.row_vn, self.col_vn]:
            msg = [0] * nei.dom_size
            if nei == self.row_vn:
                belief = [0] * self.col_vn.dom_size if self.col_vn.name not in self.incoming_msg else self.incoming_msg[
                    self.col_vn.name]
                for val in range(self.row_vn.dom_size):
                    utils = [x + y for x, y in zip(belief, self.matrix[val])]
                    msg[val] = FunctionNode.op(utils)
            else:
                belief = [0] * self.row_vn.dom_size if self.row_vn.name not in self.incoming_msg else self.incoming_msg[
                    self.row_vn.name]
                for val in range(self.col_vn.dom_size):
                    local_vec = [self.matrix[i][val] for i in range(self.row_vn.dom_size)]
                    utils = [x + y for x, y in zip(belief, local_vec)]
                    msg[val] = FunctionNode.op(utils)
            nei.incoming_msg[self.name] = msg


class FactorGraph:
    def __init__(self, pth, function_node_type, variable_node_type):
        self.variable_nodes = dict()
        self.function_nodes = []
        self.function_node_type = function_node_type
        self.variable_node_type = variable_node_type
        all_vars, all_matrix = parse(pth,1)
        self._construct_nodes(all_vars, all_matrix)

    def _construct_nodes(self, all_vars, all_matrix):
        for v, dom in all_vars:
            self.variable_nodes[v] = self.variable_node_type(v, dom)
        for matrix, row, col in all_matrix:
            self.function_nodes.append(self.function_node_type(f'({row},{col})', matrix, self.variable_nodes[row],
                                                    self.variable_nodes[col]))

    def step(self):
        for func in self.function_nodes:
            func.compute_msgs()
        for variable in self.variable_nodes.values():
            variable.compute_msgs()
            variable.make_decision()
        cost = 0
        for func in self.function_nodes:
            cost += func.matrix[func.row_vn.val_idx][func.col_vn.val_idx]
        return cost