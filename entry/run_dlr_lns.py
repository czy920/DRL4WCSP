from drl_boosted_algo.lns import run_lns
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run GS_LNS')
    parser.add_argument('-pth', '--path', type=str, required=True, help='problem path')
    parser.add_argument('-mp', '--model_path', type=int, required=True, help='model path')
    parser.add_argument('-d', '--device', type=str, required=False, help='computing device', default='cpu')
    parser.add_argument('-p', '--prob', type=float, required=True, help='destroy probability')
    parser.add_argument('-c', '--cycle_cnt', type=int, required=True, help='cycle count')
    args = parser.parse_args()

    scale_factor = 10
    best_cost = run_lns(args.path, args.model_path, device=args.device, p=args.prob,
                        scale=scale_factor, max_cycle=args.cycle_cnt)
    print(f'Best cost: {best_cost * scale_factor}')