from baselines.ms.damped_max_sum import run_dms
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run Damped Max-sum')
    parser.add_argument('-pth', '--path', type=str, required=True, help='problem path')
    parser.add_argument('-c', '--cycle_cnt', type=int, required=True, help='cycle count')
    parser.add_argument('-df', '--damped_factor', type=int, required=True, help='damped factor')
    args = parser.parse_args()

    best_cost = run_dms(args.path, cycle=args.cycle_cnt, damped_factor=args.damped_factor)
    print(f'Best cost: {best_cost}')
