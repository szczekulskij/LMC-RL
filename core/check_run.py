# One-off script to run 

import torch
import numpy as np
import random
import gymnasium as gym
# Import our previously defined classes (SACAgent, DDPGAgent, ReplayBuffer, evaluate_policy, etc.)
from agents.ddpg import DDPGAgent
from agents.sac import SACAgent
from agents.networks import ActorSAC, Critic
from buffer.replay_buffer import ReplayBuffer
from core.evaluate import evaluate_policy  
from utils.seed import set_seed  

# Device configuration: use MPS if available (on Apple silicon Macs)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")  # e.g., prints "Using device: mps" on an M1/M2 Mac


# Configuration
NUM_SEEDS = 3  # number of random seed trials (can adjust for more/less)
ENV_NAMES = ["HalfCheetah-v5", "Hopper-v5"]
ALGOS = {
    "SAC": SACAgent,    # agent classes defined elsewhere
    "DDPG": DDPGAgent
}
NUM_EPISODES = 500             # training episodes per run (configurable)
MAX_STEPS_PER_EPISODE = 1000   # max steps per episode (typical for MuJoCo envs)
BATCH_SIZE = 256               # batch size for agent updates
REPLAY_CAPACITY = 1000000      # capacity of replay buffer


def train_agent_on_env(agent_class, env_name, seed):
    """Train a given agent (SAC or DDPG) on a specific environment for NUM_EPISODES. 
       Returns the list of episode rewards collected during training."""
    # Create environment and set seed
    env = gym.make(env_name)
    set_seed(seed, env)
    
    # Initialize agent (with environment info and device)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = agent_class(obs_dim, act_dim, device=device)  # assuming agent constructor takes obs/act dimensions
    # Alternatively, agent could internally handle env spaces. Adjust as per actual implementation.
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=REPLAY_CAPACITY)
    
    episode_rewards = []  # to log rewards per episode
    for episode in range(1, NUM_EPISODES+1):
        state = env.reset()
        episode_reward = 0.0
        done = False
        step = 0
        
        while not done and step < MAX_STEPS_PER_EPISODE:
            # Choose action from agent (with exploration). Assume agent has a method select_action().
            action = agent.select_action(state)  # returns a NumPy array or list compatible with env action space
            # (Ensure the agent.select_action uses exploration noise internally for DDPG, stochastic sampling for SAC)
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Store transition in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)
            
            # Update agent's networks (if enough samples are available in buffer)
            if len(replay_buffer) >= BATCH_SIZE:
                agent.update(replay_buffer, batch_size=BATCH_SIZE)
                # The agent.update method should sample from the replay buffer and do a gradient update.
                # We assume this method exists in the agent class from the project structure.
            
            state = next_state
            step += 1
        
        episode_rewards.append(episode_reward)
        
        # (Optional) Print or log the reward for this episode
        print(f"[{agent_class.__name__} | {env_name} | Seed {seed}] Episode {episode}: Reward = {episode_reward:.2f}")
        
        # (Optional) Evaluate periodically using evaluation logic if available
        if episode % 50 == 0:
            eval_reward = evaluate_policy(agent, env_name)  # hypothetical evaluation function
            print(f"Evaluation reward (episode {episode}): {eval_reward:.2f}")
    
    # Save the final trained model to disk
    model_filename = f"{agent_class.__name__}_{env_name}_seed{seed}.pt"
    # If the agent class has internal networks, save their state dicts. For simplicity:
    torch.save(agent.get_state_dict(), model_filename)
    # Alternatively, if agent is an nn.Module or has a .state_dict:
    # torch.save(agent.state_dict(), model_filename)
    print(f"Saved model to {model_filename}")
    
    env.close()
    return episode_rewards


import matplotlib.pyplot as plt

# Main loop to run experiments
for algo_name, agent_class in ALGOS.items():
    for env_name in ENV_NAMES:
        for seed in range(NUM_SEEDS):
            print(f"\n=== Training {algo_name} on {env_name} (seed={seed}) ===")
            # Train the agent and get episode rewards log
            rewards = train_agent_on_env(agent_class, env_name, seed)
            
            # Plot the reward curve for this run
            plt.figure()
            plt.plot(rewards, label=f'Seed {seed}')
            plt.title(f"{algo_name} on {env_name} (Seed {seed})")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.legend()
            # Save plot to disk
            plot_filename = f"{algo_name}_{env_name}_seed{seed}_reward_curve.png"
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved reward plot to {plot_filename}")
