import argparse
import os
from pretrain.model import GATNet
from pretrain.dqn_agent import DQNAgent
from torch.optim import AdamW


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run pretrain')
    parser.add_argument('-tfp', '--train_files_path', type=str, required=True, help='train problem dir path')
    parser.add_argument('-vfp', '--valid_files_path', type=str, required=True, help='valid problem dir path')
    parser.add_argument('-c', '--cap', type=int, required=False, default=1000000, help='memory budget')
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=64, help='batch size')
    parser.add_argument('-e', '--epoches', type=int, required=False, default=5000, help='number of epoches')
    parser.add_argument('-i', '--iterations', type=int, required=False, default=10, help='number of training iterations')
    parser.add_argument('-mp', '--model_path', type=str, required=True, help='path to save models')
    parser.add_argument('-d', '--device', type=str, required=False, help='computing device', default='cuda')
    args = parser.parse_args()

    problems = []
    base = args.path
    for f in os.listdir(base):
        if f.endswith('.xml'):
            problems.append(os.path.join(base, f))

    model = GATNet(4, 16)
    target_model = GATNet(4, 16)
    optimizer = AdamW(model.parameters(), lr=.0001, weight_decay=5e-5)
    dqn = DQNAgent(model, target_model, optimizer, device=args.device, capacity=args.cap, model_path=args.model_path,
                   batch_size=args.batch_size, episode=args.epoches, iteration=args.iterations)
    dqn.train(args.train_files_path, args.valid_files_path)