# LMC-RL
Researching Linear Mode Connectivity in RL

# Repo structure
lmc_rl/
├── agents/
│   ├── base.py              # Common interfaces / utilities (optional)
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
│   ├── evaluate.py          # Weight interpolation and policy evaluation
│   └── seed.py              # Global seeding utilities for reproducibility
│
├── configs/
│   ├── default_sac.yaml     # Hyperparameters for SAC (learning rates, tau, etc.)
│   └── default_ddpg.yaml    # Hyperparameters for DDPG
│
├── experiments/
│   ├── run_halfcheetah.py   # Script to run full training + fork + eval (main)
│   └── analyze_results.py   # Plotting / analysis of interpolation results
│
├── logs/                    # TensorBoard or CSV logs (auto-created)
│
├── checkpoints/             # Saved agent checkpoints at fork/final
│
├── results/                 # Saved interpolation metrics (e.g., JSON/CSV/plots)
│
├── requirements_metal.txt         # All Python dependencies (PyTorch for metal)
└── README.md                # Project overview and usage instructions


# Self-notes
* `conda activate /Users/szczekulskij/miniforge3/envs/ml_fo_robotics` to activate my env


# TODOs
* Add "normal" requirements
* Clean requirements for metal (it includes my rqs for ml for robotics class project, could be cleaner)