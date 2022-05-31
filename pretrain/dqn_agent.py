import os.path
import random

import torch
from torch_geometric.data import Data, Batch
import torch.nn.functional as F

from memory import Mem
from env import Environment
from core.parser import parse


class DQNAgent:
    def __init__(self, model, target_model, optimizer, device='cpu', capacity=10000000,
                 batch_size=64, model_path='../models', epsilon=.9, scale=10, gamma=.99, episode=2000, iteration=50):
        self.model = model
        self.target_model = target_model
        self.device = device
        self.target_model.load_state_dict(model.state_dict())
        self.model.to(device)
        self.target_model.to(device)
        self.epsilon = epsilon
        self.scale = scale
        self.memory = Mem(capacity)
        self.optimizer = optimizer
        self.gamma = gamma
        self.batch_size = batch_size
        self.model_path = model_path
        self.episode = episode
        self.iteration = iteration

    def train(self, train_list, validation_list, validation=50):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        cnt = 1
        for _ in range(self.episode):
            pth = random.choice(train_list)
            all_vars, all_functions = parse(pth, self.scale)
            env = Environment(all_vars, all_functions, is_valid=False)
            total_r = self._dfs_decision_making(env, env.root, dict())
            losses = []
            for _ in range(self.iteration):
                losses.append(self._learn(self.batch_size))
            cnt += 1
            print(f'Iteration {cnt}\tCost: {int(-total_r * self.scale)}\tLoss: {sum(losses) / len(losses):.3f}')
            if cnt % validation == 0:
                cost = []
                for vp in validation_list:
                    all_vars, all_functions = parse(vp, self.scale)
                    env = Environment(all_vars, all_functions, is_valid=True)
                    cost.append(self._dfs_decision_making(env, env.root, dict(), False))
                tag = int(cnt / validation)
                print(f'Validation {tag}\tCost: {-int(sum(cost) * self.scale / len(cost))}\tLoss: {sum(losses) / len(losses):.3f}')
                torch.save(self.model.state_dict(), f'{self.model_path}/{tag}.pth')

    def _dfs_decision_making(self, env, target_var, partial_assignment, training=True):
        x, edge_index, all_function_index, action_space = env.build_graph(target_var, partial_assignment)
        s = Data(x=x, edge_index=edge_index, function_idx=all_function_index)
        if training and random.random() > self.epsilon:
            action = random.choice([i for i in range(len(action_space))])
        else:
            q_values = self.model.inference(x.to(self.device), edge_index.to(self.device), action_space, all_function_index)
            action = q_values.argmax().item()
        partial_assignment[target_var] = action
        r = env.act(partial_assignment, target_var)
        if training:
            s_prime = []
            for c in env.dfs_tree[target_var].children:
                x, edge_index, all_function_index, action_space = env.build_graph(c, partial_assignment)
                s_prime.append(Data(x=x, edge_index=edge_index, function_idx=all_function_index, action_space=action_space))
            self.memory.add(s, action_space[action], r, s_prime, len(s_prime) == 0)
        for c in env.dfs_tree[target_var].children:
            r += self._dfs_decision_making(env, c, dict(partial_assignment), training)
        return r

    def _learn(self, batch_size):
        self.optimizer.zero_grad()
        s, a, r, s_prime, done = self.memory.sample(batch_size)
        batch = Batch.from_data_list(s)
        batch.x = batch.x.to(self.device)
        batch.edge_index = batch.edge_index.to(self.device)
        self.model.eval()
        pred = self.model(batch, a)
        targets = []
        for i in range(len(s_prime)):
            if done[i]:
                targets.append(0)
            else:
                sp = s_prime[i]
                max_q_values = []
                for s in sp:
                    q_values = self.target_model.inference(s.x.to(self.device), s.edge_index.to(self.device), s.action_space,
                                                    s.function_idx)
                    max_q_values.append(q_values.max().item())
                targets.append(sum(max_q_values))
        targets = torch.tensor(r, dtype=torch.float32, device=self.device) + self.gamma * torch.tensor(targets, dtype=torch.float32, device=self.device)
        targets.unsqueeze_(1)

        self.model.train()
        loss = F.mse_loss(pred, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self._soft_update()
        return loss.item()

    def _soft_update(self, tau=.0001):
        for t_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            new_param = tau * param.data + (1.0 - tau) * t_param.data
            t_param.data.copy_(new_param)