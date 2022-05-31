from baselines.incomplete_algo import IncompleteAlgo
from pretrain.model import GATNet
from pretrain.env import Environment
import torch


class ModelBasedIncompleteAlgo(IncompleteAlgo):
    def __init__(self, pth, path_model, scale_factor, max_cycle, device):
        super(ModelBasedIncompleteAlgo, self).__init__(pth, scale_factor, max_cycle)
        self.model = GATNet(4, 16)
        self.device = torch.device(device)
        self.model.load_state_dict(torch.load(path_model, map_location=self.device))
        self.env = Environment(self.all_vars, self.all_functions, is_valid=True, device=self.device)

    def dfs_decision_making(self, target_var, partial_assignment):
        x, edge_index, all_function_index, action_space = self.env.build_graph(target_var=target_var, partial_assignment=partial_assignment)
        q_values = self.model.inference(x, edge_index, action_space, all_function_index)
        action = q_values.argmax().item()
        partial_assignment[target_var] = action
        r = - self.env.act(partial_assignment, target_var)
        for c in self.env.dfs_tree[target_var].children:
            r += self.dfs_decision_making(c, partial_assignment)
        return r