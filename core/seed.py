import os, random, numpy as np, torch, gymnasium as gym

# Reproducibility: set global seeds
def set_seed(env, seed: int = 0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    env.reset(seed=seed)                     # Gymnasium env seeding
    env.action_space.seed(seed)              # Seed action sampling if any