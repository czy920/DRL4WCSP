from baselines.gdba import GDBA
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run GDBA')
    parser.add_argument('-pth', '--path', type=str, required=True, help='problem path')
    parser.add_argument('-c', '--cycle_cnt', type=int, required=True, help='cycle count')
    args = parser.parse_args()

    scale_factor = 10
    algo = GDBA(args.path, scale_factor=scale_factor, max_cycle=args.cycle_cnt)
    best_cost = algo.run() * scale_factor
    print(f'Best cost: {best_cost}')