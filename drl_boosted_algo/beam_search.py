from collections import namedtuple
from model_based_incomplete_algo import ModelBasedIncompleteAlgo


PA = namedtuple('PA', ['assign', 'cpa_cost'])
PA_ext = namedtuple('PA', ['assign', 'cpa_cost', 'eval_cost'])

class beam_search(ModelBasedIncompleteAlgo):
    def __init__(self, pth, path_model, scale_factor=10, k_ext=2, k_bw=4, max_cycle=-1, device='cpu'):
        super().__init__(pth, path_model, scale_factor, max_cycle, device)
        self.k_ext = k_ext
        self.k_bw = k_bw

    def _decision_making(self, target_var, B):
        B_prime = dict()
        for i in range(len(B)):
            assign = B[i].assign
            sep_assign = {key : assign[key] for key in self.env.dfs_tree[target_var].sep}
            x, edge_index, all_function_index, action_space = self.env.build_graph(target_var, sep_assign)
            q_values = - self.model.inference(x.to(self.device), edge_index.to(self.device), action_space, all_function_index)
            q_values = [x for j in q_values.numpy().tolist() for x in j]
            actions = self.top_k(q_values, min(self.k_ext, len(q_values)))
            for j in range(len(actions)):
                extend_assign = dict(assign)
                extend_assign[target_var] = actions[j]
                r = - self.env.act(extend_assign, target_var)
                B_prime[i*(len(actions))+j] = PA_ext(extend_assign, r + B[i].cpa_cost,
                                                     r +B[i].cpa_cost + q_values[actions[j]])
        costs = [B_prime[i].eval_cost for i in range(len(B_prime))]
        idx = self.top_k(costs, min(len(costs), self.k_bw))
        B_copy = dict()
        for i in range(len(idx)):
            B_copy[i] = PA(dict(B_prime[idx[i]].assign), B_prime[idx[i]].cpa_cost)
        return B_copy

    def run(self):
        B = dict()
        B[0] = PA({}, 0)
        level_var = dict()
        for var in self.env.dom_size.keys():
            level = self.env.dfs_tree[var].level
            if level not in level_var:
                level_var[level] = []
            level_var[level].append(var)

        for level in range(max(level_var.keys())+1):
            for var in level_var[level]:
                B = self._decision_making(var, B)
        costs = [B[i].cpa_cost for i in range(len(B))]
        return min(costs)
