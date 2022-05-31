from baselines.t_lns import T_LNS
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run T-LNS')
    parser.add_argument('-pth', '--path', type=str, required=True, help='problem path')
    parser.add_argument('-c', '--cycle_cnt', type=int, required=True, help='cycle count')
    parser.add_argument('-p', '--prob', type=float, required=True, help='destroy probability')
    args = parser.parse_args()

    scale_factor = 10
    best_cost = 10**8
    lns = T_LNS(args.path, p=args.prob, scale=scale_factor)
    for current_cycle in range(args.cycle_cnt):
        new_assignment = lns.step()
        c = lns.total_cost(new_assignment)
        lns.assignment = new_assignment
        best_cost = min(best_cost, c)
    print(f'Best cost: {best_cost * scale_factor}')