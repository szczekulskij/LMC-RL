import os, random, numpy as np, torch, gymnasium as gym

# Reproducibility: set global seeds
def set_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)