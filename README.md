# LMC-RL
Researching Linear Mode Connectivity in RL

## Requirements (newly added on new laptop self-note):
* 

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
* `conda activate /Users/szczekulskij/miniforge3/envs/ml_fo_robotics` to activate my env
* To run check: `python -m core.check_run`
* Handy command to log in the console while also logging into a log txt file! `python -u -m core.check_run 2>&1 | tee logs/check_up_runtime_logs.txt`

## Self-notes - MuJoco Env Hardness
* Easy envs = ["InvertedDoublePendulum-v5", "Hopper-v5", "Swimmer-v5", "Reacher-v5"] (eg. small action space of few continous actions, with somewhat bigget state space of ~10-20)
* Medium hard envs = ["HalfCheetah-v5", "Pusher-v5"]
* Hard envs = ["HumanoidStandup-v5", "Humanoid-v5",]


## TODOs
* Add "normal" requirements
* Clean requirements for metal (it includes my reqs for ml for robotics class project, could be cleaner)
* Remove the experiment default config perhaps. I don't see why there should be two places of "ground truth" setting (eg. either using yaml or pass everything from the console!)
* Change plotting to plot multiple subfigures next to each other, rather than single figure !