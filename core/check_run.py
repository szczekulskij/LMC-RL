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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Device configuration: use MPS if available (on Apple silicon Macs)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")  # e.g., prints "Using device: mps" on an M1/M2 Mac



# Configuration
NUM_SEEDS = 3  # number of random seed trials (can adjust for more/less)
ENV_NAMES = ["InvertedDoublePendulum-v5"]
ALGOS = {
    "SAC": SACAgent,    # agent classes defined elsewhere
    "DDPG": DDPGAgent
}
# NUM_EPISODES = 500             # training episodes per run (configurable)
# NUM_EVAL_EPISODES = 50          # evaluation episodes (for logging)
NUM_EPISODES = 10             # training episodes per run (configurable)
NUM_EVAL_EPISODES = 3          # evaluation episodes (for logging)



MAX_STEPS_PER_EPISODE = 1000   # max steps per episode (typical for MuJoCo envs)
BATCH_SIZE = 256               # batch size for agent updates
REPLAY_CAPACITY = 1000000      # capacity of replay buffer


def train_agent_on_env(agent_class, env_name, seed):
    """Train a given agent (SAC or DDPG) on a specific environment for NUM_EPISODES. 
       Returns the list of episode rewards and cumulative elapsed times."""
    # Create environment and set seed
    env = gym.make(env_name)
    set_seed(seed, env)
    
    # Initialize agent (with environment info and device)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = agent_class(obs_dim, act_dim, device=device)  # assuming agent constructor takes obs/act dimensions
    # Alternatively, agent could internally handle env spaces. Adjust as per actual implementation.
    
    # Initialize replay buffer with state and action dimensions
    replay_buffer = ReplayBuffer(state_dim=obs_dim, action_dim=act_dim, capacity=REPLAY_CAPACITY)
    
    episode_rewards = []  # to log rewards per episode
    cumulative_times = []  # to log cumulative elapsed time per episode
    start_time = time.time()  # Start timing

    for episode in range(1, NUM_EPISODES+1):
        state, _ = env.reset()  # Updated to unpack reset output
        episode_reward = 0.0
        done = False
        step = 0
        
        while not done and step < MAX_STEPS_PER_EPISODE:
            # Use get_action instead of select_action
            action = agent.get_action(state)  # returns a NumPy array or list compatible with env action space
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Store transition in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)
            
            # Update agent's networks (if enough samples are available in buffer)
            if len(replay_buffer) >= BATCH_SIZE:
                agent.train_step(replay_buffer, batch_size=BATCH_SIZE)  # Ensure train_step is used
            
            state = next_state
            step += 1
        
        episode_rewards.append(episode_reward)
        cumulative_times.append(time.time() - start_time)  # Record cumulative elapsed time
        
        # (Optional) Print or log the reward for this episode
        print(f"[{agent_class.__name__} | {env_name} | Seed {seed}] Episode {episode}: Reward = {episode_reward:.2f}")
        
        # (Optional) Evaluate periodically using evaluation logic if available
        if episode % NUM_EVAL_EPISODES == 0:
            eval_reward = evaluate_policy(agent, env)  # Pass the environment object instead of its name
            print(f"Evaluation reward (episode {episode}): {eval_reward:.2f}")
    
    # Save the final trained model to disk
    model_filename = f"{agent_class.__name__}_{env_name}_seed{seed}.pt"
    filepath = f"model_weights/check_run/{model_filename}"
    agent.save(filepath)  
    print(f"Saved model to {model_filename}")
    
    env.close()
    return episode_rewards, cumulative_times


import time  # Import time module for wall clock measurement
import matplotlib.pyplot as plt

# Main loop to run experiments
for env_name in ENV_NAMES:
    fig, axes = plt.subplots(2, NUM_SEEDS, figsize=(15, 10))  # 2 rows (one per algo), NUM_SEEDS columns
    fig.suptitle(f"Training Results for {env_name} (Rewards)", fontsize=16)

    time_fig, time_axes = plt.subplots(2, NUM_SEEDS, figsize=(15, 10))  # Separate figure for wall clock time
    time_fig.suptitle(f"Training Results for {env_name} (Wall Clock Time)", fontsize=16)

    # Use ThreadPoolExecutor for parallelism
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {}
        for row, (algo_name, agent_class) in enumerate(ALGOS.items()):
            for col, seed in enumerate(range(NUM_SEEDS)):
                print(f"\n=== Scheduling {algo_name} on {env_name} (seed={seed}) ===")
                # Submit the train_agent_on_env function to the executor
                future = executor.submit(train_agent_on_env, agent_class, env_name, seed)
                futures[future] = (row, col, algo_name, seed)

        # Process results as they complete
        for future in as_completed(futures):
            row, col, algo_name, seed = futures[future]
            try:
                rewards, cumulative_times = future.result()  # Get the result of the training
                # Plot the reward curve in the corresponding subplot
                ax = axes[row, col]
                ax.plot(rewards, label=f'Seed {seed}')
                ax.set_title(f"{algo_name} (Seed {seed})")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Total Reward")
                ax.legend()

                # Plot the cumulative elapsed time in the corresponding subplot
                time_ax = time_axes[row, col]
                time_ax.plot(range(len(cumulative_times)), cumulative_times, label=f'Seed {seed}')
                time_ax.set_title(f"{algo_name} (Seed {seed})")
                time_ax.set_xlabel("Episode")
                time_ax.set_ylabel("Wall Clock Time (s)")
                time_ax.legend()

            except Exception as e:
                print(f"Error occurred while training {algo_name} on {env_name} (seed={seed}): {e}")

    # Adjust layout and save the combined reward plot
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
    reward_plot_filename = f"results/check_run/{env_name}_combined_reward_plot.png"
    fig.savefig(reward_plot_filename)  # Use the correct figure for rewards
    plt.close(fig)  # Close the reward figure
    print(f"Saved combined reward plot to {reward_plot_filename}")

    # Adjust layout and save the combined time plot
    time_fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
    time_plot_filename = f"results/check_run/{env_name}_combined_time_plot.png"
    time_fig.savefig(time_plot_filename)  # Use the correct figure for wall clock time
    plt.close(time_fig)  # Close the time figure
    print(f"Saved combined time plot to {time_plot_filename}")