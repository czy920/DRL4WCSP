from drl_boosted_algo.greed_search import greed_search
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run GS_DRL')
    parser.add_argument('-pth', '--path', type=str, required=True, help='problem path')
    parser.add_argument('-mp', '--model_path', type=int, required=True, help='model path')
    parser.add_argument('-d', '--device', type=str, required=False, help='computing device', default='cpu')
    args = parser.parse_args()

    scale_factor = 10
    algo = greed_search(args.path, path_model=args.model_path, scale_factor=scale_factor, device=args.device)
    best_cost = algo.run() * scale_factor
    print(f'Best cost: {best_cost}')