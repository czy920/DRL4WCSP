from model_based_incomplete_algo import ModelBasedIncompleteAlgo


class greed_search(ModelBasedIncompleteAlgo):
    def __init__(self, pth, path_model, scale_factor=10, max_cycle=-1, device='cpu'):
        super(greed_search, self).__init__(pth, path_model, scale_factor, max_cycle, device)

    def run(self):
        return self.dfs_decision_making(self.env.root, {})
