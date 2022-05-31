import datetime
import os
from entities import FactorGraph, VariableNode, FunctionNode


def run_dms(pth, cycle=1000, damped_factor=.9):
    VariableNode.damp_factor = damped_factor
    fg = FactorGraph(pth, FunctionNode, VariableNode)
    cost = []
    best_cost = []
    for it in range(cycle):
        cost.append(fg.step())

        if len(best_cost) == 0:
            best_cost.append(cost[-1])
        else:
            best_cost.append(min(best_cost[-1], cost[-1]))
    print(pth, best_cost[-1], datetime.datetime.now())

def run(problem_dir, cycle=1000, damped_factor=.9):
    cic = []
    bcic = []
    cnt = 0
    for f in os.listdir(problem_dir):
        if not f.endswith('.xml'):
            continue

        cnt += 1
        pth = os.path.join(problem_dir, f)
        VariableNode.damp_factor = damped_factor
        fg = FactorGraph(pth, FunctionNode, VariableNode)
        cost = []
        best_cost = []
        for it in range(cycle):
            cost.append(fg.step())

            if len(best_cost) == 0:
                best_cost.append(cost[-1])
            else:
                best_cost.append(min(best_cost[-1], cost[-1]))
        print(pth, best_cost[-1], datetime.datetime.now())
        if len(cic) == 0:
            cic = cost
            bcic = best_cost
        else:
            cic = [x + y for x, y in zip(cost, cic)]
            bcic = [x + y for x, y in zip(best_cost, bcic)]
    return [x / cnt for x in cic], [x / cnt for x in bcic]
