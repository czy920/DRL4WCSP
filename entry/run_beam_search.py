from baselines.beam_search import beam_search
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run BS')
    parser.add_argument('-pth', '--path', type=str, required=True, help='problem path')
    parser.add_argument('-mpth', '--model_path', type=int, required=True, help='model path')
    parser.add_argument('-k_bw', '--k_bw', type=int, required=True, help='k_bw')
    parser.add_argument('-k_ext', '--k_ext', type=int, required=True, help='k_ext')
    parser.add_argument('-k_mb', '--k_mb', type=int, required=True, help='k_mb')
    args = parser.parse_args()

    scale_factor = 10
    algo = beam_search(args.path, scale_factor=scale_factor, k_bw=args.k_bw, k_ext=args.k_ext, k_mb=args.k_mb)
    best_cost = algo.run() * scale_factor
    print(f'Best cost: {best_cost}')