# LMC-RL
Researching Linear Mode Connectivity in RL

## Quick Start

### Environment Setup
```bash
# Activate environment (for example)
source ~/.venvs/lmc-rl/bin/activate

# Install dependencies
pip install torch torchvision
pip install -r requirements-mac.txt
pip install "gymnasium[mujoco]"
```

### Running Experiments (to generate data and fork weights)

**Quick test run:**
```bash
# Quick check run with multiple algorithms and seeds
python3 -m core.check_run

# With logging to file
python3 -u -m core.check_run 2>&1 | tee logs/runtime_logs.txt
```

**Full LMC experiments:**
```bash
# Basic experiment with default settings
python3 -m experiments.run_env --env InvertedDoublePendulum-v5 --algo SAC --seed 42

# Custom fork points
python3 -m experiments.run_env --fork_points "0.1,0.3,0.5,0.7,0.9" --env Hopper-v5 --algo DDPG

# Longer training with custom evaluation frequency
python3 -m experiments.run_env --total_steps 500000 --eval_freq 2000 --env HalfCheetah-v5

# Test different buffer strategies for forks
python3 -m experiments.run_env --env InvertedDoublePendulum-v5 --algo SAC --fork_buffer_strategy fresh

# Compare buffer strategies with same seed for reproducible comparison
python3 -m experiments.run_env --env Hopper-v5 --algo SAC --seed 42 --fork_buffer_strategy copy
python3 -m experiments.run_env --env Hopper-v5 --algo SAC --seed 42 --fork_buffer_strategy fresh
```

**Available options:**
- **Environments**: 
  - Easy: InvertedDoublePendulum-v5, Hopper-v5, Swimmer-v5, Reacher-v5
  - Medium: HalfCheetah-v5, Pusher-v5
  - Hard: HumanoidStandup-v5, Humanoid-v5
- **Algorithms**: SAC, DDPG
- **Fork points**: Comma-separated percentages (e.g., "0.1,0.5,0.9")
- **Buffer strategies**: copy (default), fresh, shared, split
- **Seeds**: Any integer (random if not specified)
- **Training steps**: Any positive integer
- **Evaluation frequency**: How often to evaluate policy during training

### Analyzing Experiments

**Analyze fork results and generate plots:**
```bash
# Basic analysis - specify the experiment results directory
python3 -m experiments.analyze --results_dir results/base/InvertedDoublePendulum-v5/SAC/seed_42_20241231_143022

# With custom total steps (for proper time scaling)
python3 -m experiments.analyze --results_dir results/base/Hopper-v5/DDPG/seed_123_20241231_150000 --total_steps 500000

# Find your results directory
ls results/base/  # List environments
ls results/base/InvertedDoublePendulum-v5/SAC/  # List experiment runs
```

**Generated outputs:**
- `all_interpolations.png`: All fork interpolation curves on one plot
- `interp_fork_{id}.png`: Individual interpolation curves with error bars
- `instability_over_time.png`: Instability metrics across fork points
- Files saved to: `{results_dir}/fork_analysis/`

### Default Parameters

When running experiments without specifying parameters, the following defaults are used:

```bash
# Default experiment settings
--env InvertedDoublePendulum-v5    # Environment
--algo SAC                         # Algorithm  
--seed <random>                    # Random seed (1-1000000)
--total_steps 200000               # Total training steps (5e5 // 2.5)
--eval_freq 1000                   # Evaluation frequency
--max_episode_steps 1000           # Max steps per episode
--fork_points "0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8"  # Fork percentages
--fork_buffer_strategy copy        # Buffer handling: copy, fresh, shared, split
--config experiments/experiment_default_config.yaml          # Default config

# Default config file settings (experiment_default_config.yaml)
buffer_size: 1000000               # Replay buffer capacity
experiment_name: "base"            # Results directory name
# Note: fork_buffer_strategy now controlled via --fork_buffer_strategy command-line argument
```

**Training parameters:**
- **Train frequency**: Every 2 steps (after 10k buffer warmup)
- **Evaluation episodes**: 10 episodes for periodic evals, 100 for fork analysis
- **Linear interpolation**: 101 alpha values (0.0 to 1.0)
- **Device**: Auto-detected (MPS > CUDA > CPU)

### Buffer Strategies for Fork Analysis

The `--fork_buffer_strategy` parameter controls how experience replay buffers are handled when training forks:

- **`copy`** (default): Both forks get identical copies of the original buffer
  - *Research use*: Isolates the effect of different exploration post-fork
  - *Best for*: Testing pure exploration noise effects with same learning history

- **`fresh`**: Both forks start with empty buffers at the fork point
  - *Research use*: Tests robustness of learned representations to completely new experience
  - *Best for*: Understanding how well learned weights can continue learning from scratch

- **`shared`**: Both forks continue adding to the same shared buffer
  - *Research use*: Pure exploration noise comparison with shared experience
  - *Best for*: Testing connectivity when agents see all experience but explore differently

- **`split`**: Divide existing buffer between forks (not fully implemented)
  - *Research use*: Balanced but independent experience for each fork
  - *Best for*: Testing connectivity with different but equal amounts of experience

### Configuration

Experiments can be configured via YAML files in `configs/` or command-line arguments:

```bash
# Using custom config file
python -m experiments.run_env --config my_experiment.yaml

# Command-line overrides
python -m experiments.run_env --env Swimmer-v5 --algo SAC --seed 123 --total_steps 200000
```

### Results

Results are automatically saved to timestamped directories:
- **Training data**: `results/base/{env}/{algo}/seed_{seed}_{timestamp}/`
- **Model weights**: `model_weights/base/{env}/{algo}/seed_{seed}_{timestamp}/`
- **Fork analysis**: JSON files with instability metrics and interpolation results


## 📁 Repo structure
```
lmc_rl/
├── agents/
│   ├── sac.py               # SACAgent class and SAC components
│   ├── ddpg.py              # DDPGAgent class and DDPG components
│   ├── networks.py          # Shared neural nets (ActorDDPG, ActorSAC, Critic)
│   └── utils.py             # e.g., soft_update, log_std clamp, action sampling
│
├── buffer/
│   └── replay_buffer.py     # ReplayBuffer class with optional save/load
│
├── core/
│   ├── train.py             # Core training loop with forking logic
│   ├── evaluate.py          # Instability calculations and policy evaluation
│
├── utils/
│   └── seed.py              # Global seeding utilities for reproducibility
│
├── configs/
│   ├── default_sac.yaml     # Hyperparameters for SAC (learning rates, tau, etc.)
│   └── default_ddpg.yaml    # Hyperparameters for DDPG
│
├── experiments/ #TODO - Add
│   ├── run_env.py           # Script to run full training + fork + eval (main)
│   └── analyze_results.py   # Plotting / analysis of interpolation results
│
├── logs/                    # TensorBoard or CSV logs (auto-created)
│
├── model_weights/           # Saved agent checkpoints at fork/final
│
├── results/                 # Saved interpolation metrics (e.g., JSON/CSV/plots)
│
├── requirements_metal.txt   # All Python dependencies (PyTorch for metal)
│
└── README.md                # Project overview and usage instructions
```


## Self-notes
* Activate you env (eg. for example `source ~/.venvs/lmc-rl/bin/activate`)
* To run check run (just an endToEnd training of a single env without forking on the way): `python3 -m core.check_run`
* Run test (with logging both in console & flushing to .txt): `python -u -m core.check_run 2>&1 | tee logs/check_up_runtime_logs.txt`


## Self-notes - MuJoco Env Hardness
* Easy envs = ["InvertedDoublePendulum-v5", "Hopper-v5", "Swimmer-v5", "Reacher-v5"] (eg. small action space of few continous actions, with somewhat bigget state space of ~10-20)
* Medium hard envs = ["HalfCheetah-v5", "Pusher-v5"]
* Hard envs = ["HumanoidStandup-v5", "Humanoid-v5",]


## TODOs
* Clean requirements for M4 (it includes my reqs for ml for robotics class project, could be cleaner)
* Remove the experiment default config perhaps. I don't see why there should be two places of "ground truth" setting (eg. either using yaml or pass everything from the console!)
* Change plotting to plot multiple subfigures next to each other, rather than single figure !