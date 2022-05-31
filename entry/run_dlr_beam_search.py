from drl_boosted_algo.beam_search import beam_search
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run BS_DRL')
    parser.add_argument('-pth', '--path', type=str, required=True, help='problem path')
    parser.add_argument('-mp', '--model_path', type=int, required=True, help='model path')
    parser.add_argument('-d', '--device', type=str, required=False, help='computing device', default='cpu')
    parser.add_argument('-k_bw', '--k_bw', type=int, required=True, help='k_bw')
    parser.add_argument('-k_ext', '--k_ext', type=int, required=True, help='k_ext')
    args = parser.parse_args()

    scale_factor = 10
    algo = beam_search(args.path, path_model=args.path_model, scale_factor=scale_factor,
                       k_bw=args.k_bw, k_ext=args.k_ext, device=args.device)
    best_cost = algo.run() * scale_factor
    print(f'Best cost: {best_cost}')