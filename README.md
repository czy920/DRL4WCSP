# Learning Heuristics for Weighted CSPs through Deep Reinforcement Learning

# Requirements
- **PyTorch 1.9.0**
- **PyTorch Geometric 1.7.1**

# Directory structure

- `baselines` contains the implementation of all compared baselines
- `drl_boosted_algo` contains the implementation of our DRL-boosted algorithms
- `core` contains the core data structures to run the simulation
- `entry` contains the entry point of each algorithm
- `pretrain` contains the implementation of pretraining stage

# How to run the code

See the command line interface of `run_*.py` in `entry`.

Example:

`python -um entry.run_dampedmaxsum -pth problem.xml -c 1000 -df 0.9`
