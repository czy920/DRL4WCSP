from pretrain.env import Environment
from collections import namedtuple
from baselines.incomplete_algo import IncompleteAlgo
from core.bucket import Bucket

PA = namedtuple('PA', ['assign', 'cpa_cost'])
PA_ext = namedtuple('PA', ['assign', 'cpa_cost', 'eval_cost'])


class beam_search(IncompleteAlgo):
    def __init__(self, pth, scale_factor, k_bw, k_ext, k_mb, max_cycle=-1):
        super(beam_search, self).__init__(pth, scale_factor, max_cycle)
        self.k_ext = k_ext
        self.k_bw = k_bw
        self.k_b = k_mb
        self.buckets = dict()

    def _join_mini_buckets(self, bucket_list, var):
        buckets = [b.copy() for b in bucket_list]
        ret_buckets = []

        dims = set()
        joint_buckets = []
        for b in buckets:
            dims.update(b.dims)
            if len(dims) > self.k_b:
                ret_buckets.append(Bucket.join(joint_buckets).proj(var))

                dims = set(b.dims)
                joint_buckets = [b]
            else:
                joint_buckets.append(b)
        if len(joint_buckets) > 0:
            ret_buckets.append(Bucket.join(joint_buckets).proj(var))
        return ret_buckets

    def _mini_bucket_elim(self):
        elim_buckets = dict()
        for level in range(max(self.level_var.keys()), -1, -1):
            for var in self.level_var[level]:
                joint_buckets = []
                non_joint_buckets = []
                for child in self.env.dfs_tree[var].children:
                    for bucket in elim_buckets[child]:
                        if var in bucket.dims:
                            joint_buckets.append(bucket)
                        else:
                            non_joint_buckets.append(bucket)
                self.buckets[var] = joint_buckets + non_joint_buckets

                for ap in self.env.dfs_tree[var].all_parents:
                    joint_buckets.append(Bucket.from_matrix(self.function_table[var][ap], var, ap))

                elim_buckets[var] = self._join_mini_buckets(joint_buckets, var) + non_joint_buckets

    def _sub_problem_est(self, var, assign):
        buckets = self.buckets[var]
        cost = 0
        for b in buckets:
            cost += b.reduce(assign, True)
        return cost

    def _decision_making(self, target_var, B):
        B_prime = dict()
        for i in range(len(B)):
            assign = B[i].assign
            q_values = [0]*self.dom_size[target_var]
            for val in range(self.dom_size[target_var]):
                assign[target_var] = val
                q_values[val] = - self.env.act(assign, target_var) + self._sub_problem_est(target_var, assign)
            actions = IncompleteAlgo.top_k(q_values, min(self.k_ext, len(q_values)))
            for j in range(len(actions)):
                extend_assign = dict(assign)
                extend_assign[target_var] = actions[j]
                r = - self.env.act(extend_assign, target_var)
                B_prime[i*(len(actions))+j] = PA_ext(extend_assign, r + B[i].cpa_cost, B[i].cpa_cost + q_values[actions[j]])

        data_cost = [B_prime[i].eval_cost for i in range(len(B_prime))]
        idx = IncompleteAlgo.top_k(data_cost, min(len(data_cost), self.k_bw))
        B_copy = dict()
        for i in range(len(idx)):
            B_copy[i] = PA(dict(B_prime[idx[i]].assign), B_prime[idx[i]].cpa_cost)
        return B_copy

    def run(self):
        self.env = Environment(self.all_vars, self.all_functions, is_valid=True)

        B = dict()
        B[0] = PA({}, 0)
        self.level_var = dict()
        for var in self.dom_size.keys():
            level = self.env.dfs_tree[var].level
            if level not in self.level_var:
                self.level_var[level] = []
            self.level_var[level].append(var)

        self._mini_bucket_elim()
        for level in range(max(self.level_var.keys())+1):
            for var in self.level_var[level]:
                B = self._decision_making(var, B)
        costs = [B[i].cpa_cost for i in range(len(B))]
        return min(costs)
