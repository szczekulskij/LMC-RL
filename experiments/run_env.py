import os
import json
import time
import yaml
import argparse
import torch
import numpy as np
import gymnasium as gym
from copy import deepcopy
from datetime import datetime

from agents.ddpg import DDPGAgent
from agents.sac import SACAgent
from buffer.replay_buffer import ReplayBuffer
from core.evaluate import evaluate_policy, linear_interpolation_policy
from utils.seed import set_seed
import random

DEFAULT_TOTAL_STEPS = 5e5 //10 # based on my few runs, it trains too on invertedPentulum fast, so I set it to 1/10th of the original
DEFAULT_BUFFER_SIZE = 1e6
DEFAULT_EVAL_FREQ = 1000
DEFAULT_MAX_EPISODE_STEPS = 1000

# Hyperparameters
max_episode_steps = 1000
num_eval_episodes = 10
alphas = np.linspace(0, 1, 101) # alphas for linear interpolation

def parse_args():
    parser = argparse.ArgumentParser(description='Run LMC-RL experiments')
    parser.add_argument('--env', type=str, default='InvertedDoublePendulum-v5', 
                        help='Gymnasium environment name')
    parser.add_argument('--algo', type=str, default='SAC', choices=['SAC', 'DDPG'],
                        help='RL algorithm to use')
    parser.add_argument('--seed', type=int, default=random.randint(1, 1000000), help='Random seed')
    parser.add_argument('--total_steps', type=int, default=DEFAULT_TOTAL_STEPS, 
                        help='Total training steps')
    parser.add_argument('--eval_freq', type=int, default=DEFAULT_EVAL_FREQ, 
                        help='Evaluation frequency')
    
    
    default_fork_points = ','.join([str(i/100) for i in range(0, 101, 10)])
    # default_fork_points = '0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8'
    parser.add_argument('--fork_points', type=str, default=default_fork_points,
                        help='Comma-separated list of percentages of training to fork at')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--max_episode_steps', type=int, default=DEFAULT_MAX_EPISODE_STEPS, 
                        help='Maximum number of steps per episode')
    return parser.parse_args()

def load_config(config_path=None):
    """Load configuration from file or use default."""
    if config_path is not None:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    default_config_path = f'experiments/experiment_default_config.yaml'
    with open(default_config_path, 'r') as f:
        return yaml.safe_load(f)

def create_experiment_dir(env_name, algo, seed, experiment_name=None):
    """Create directories for saving experiment results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"results/{experiment_name}/{env_name}/{algo}/seed_{seed}_{timestamp}"
    weights_dir = f"model_weights/{experiment_name}/{env_name}/{algo}/seed_{seed}_{timestamp}"
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/evaluations", exist_ok=True)
    os.makedirs(f"{exp_dir}/forks", exist_ok=True)
    
    return exp_dir, weights_dir

def create_agent(env, algo, device):
    """Create agent based on algorithm name."""
    #TODO: Do we want to be able to pass in the config here?
    # Config could include for example hidden_dims, learning rates, etc.
    # By default these are ported from configs/default_sac.yaml and configs/default_ddpg.yaml
    if algo == 'SAC':
        return SACAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device,
        )
    elif algo == 'DDPG':
        return DDPGAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0], 
            device=device,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

def fork_training(env_name, agent, algo_name, fork_step, fork_id, weights_dir, device, buffer, total_steps):
    """Fork agent training and train with different noise that comes from different seeds."""
    print(f"Forking at step {fork_step} (ID: {fork_id})")
    
    # Create two copies of the agent with the same weights
    fork1_agent = create_agent(gym.make(env_name), algo_name, device=device)
    fork2_agent = create_agent(gym.make(env_name), algo_name, device=device)
    
    # Load weights from the parent agent
    fork1_agent.load_from_another_agent(agent)
    fork2_agent.load_from_another_agent(agent)
    
    # Continue training both forks with different random seeds
    fork1_seed = fork_id + 10000
    fork2_seed = fork_id + 20000
    
    # Train fork 1
    set_seed(fork1_seed)
    fork1_env = gym.make(env_name)
    fork1_buffer = deepcopy(buffer)  # Copy the buffer for fork 1
    fork1_buffer.set_seed(fork1_seed)  # Set seed for the buffer if needed
    
    total_steps = int(total_steps) #TODO: Fit upstream
    print(f"Training fork 1 (seed: {fork1_seed}) from step {fork_step} to {total_steps}")
    train_steps = int(total_steps - fork_step)
    state, _ = fork1_env.reset(seed=fork1_seed)
    for step in range(train_steps):
        action = fork1_agent.get_action(state)
        next_state, reward, terminated, truncated, _ = fork1_env.step(action)
        done = terminated or truncated
        
        fork1_buffer.add(state, action, reward, next_state, done)
        state = next_state if not done else fork1_env.reset()[0]
        
        if len(fork1_buffer) > 10000: # SpinningUp suggests 10000 to "prevent learning from super sparse experience"
            fork1_agent.train_step(fork1_buffer)
        
        if done:
            state, _ = fork1_env.reset()
        
        # Print progress for fork 1
        if step % (train_steps // 10) == 0:  # Print every 10% of progress
            print(f"Fork 1 (ID: {fork_id}): {step / train_steps:.1%} completed")
            evaluation_rewards = evaluate_policy(fork1_agent, fork1_env, episodes=3)
            print(f"Fork 1 (ID: {fork_id}) evaluation reward: {np.mean(evaluation_rewards):.3f}")
    
    # Train fork 2 (similar to fork 1 but with different seed)
    set_seed(fork2_seed)
    fork2_env = gym.make(env_name)

    #TODO: Experiment with what happens if we create a new buffer. De-couple this option "copy_buffer=bool" to experiment yaml config file
    fork2_buffer = deepcopy(buffer)  # Copy the buffer for fork 1
    fork2_buffer.set_seed(fork2_seed)  # Set seed for the buffer if needed
    
    print(f"Training fork 2 (seed: {fork2_seed}) from step {fork_step} to {total_steps}")
    state, _ = fork2_env.reset(seed=fork2_seed)
    for step in range(train_steps):
        action = fork2_agent.get_action(state)
        next_state, reward, terminated, truncated, _ = fork2_env.step(action)
        done = terminated or truncated
        
        fork2_buffer.add(state, action, reward, next_state, done)
        state = next_state if not done else fork2_env.reset()[0]
        
        
        if len(fork2_buffer) > 10000: # SpinningUp suggests 10000 to "prevent learning from super sparse experience"
            fork2_agent.train_step(fork2_buffer)
        
        if done:
            state, _ = fork2_env.reset()
        
        # Print progress for fork 2
        if step % (train_steps // 10) == 0:  # Print every 10% of progress
            print(f"Fork 2 (ID: {fork_id}): {step / train_steps:.1%} completed")
            rewards_eval = evaluate_policy(fork2_agent, fork2_env, episodes=3)
            print(f"Fork 2 (ID: {fork_id}) evaluation reward: {np.mean(rewards_eval):.3f}")
    
    # Save the final weights
    fork1_path = f"{weights_dir}/fork1_{fork_id}.pt"
    fork2_path = f"{weights_dir}/fork2_{fork_id}.pt"
    fork1_agent.save(fork1_path)
    fork2_agent.save(fork2_path)
    print(f"Fork 1 saved to {fork1_path}")
    print(f"Fork 2 saved to {fork2_path}")
    
    return fork1_agent, fork2_agent

def analyze_instability(env_name, fork1_agent, fork2_agent, exp_dir, fork_id, num_eval_episodes=100):
    """Analyze instability between two forked agents using linear interpolation."""
    print(f"Analyzing instability for fork {fork_id}")
    
    # Create evaluation environment
    eval_env = gym.make(env_name)
    
    # Evaluate individual agents
    fork1_rewards = evaluate_policy(fork1_agent, eval_env, episodes=num_eval_episodes)
    fork2_rewards = evaluate_policy(fork2_agent, eval_env, episodes=num_eval_episodes)
    
    # Linear interpolation between the two agents
    interpolation_results = []
    
    for alpha in alphas:
        interp_rewards = linear_interpolation_policy(
            fork1_agent, fork2_agent, alpha, eval_env, num_episodes=num_eval_episodes
        )
        interpolation_results.append({
            'alpha': float(alpha),
            'mean_reward': float(np.mean(interp_rewards)),
            'std_reward': float(np.std(interp_rewards))
        })
    
    # Calculate instability metrics
    rewards = [res['mean_reward'] for res in interpolation_results]
    endpoint_avg = (interpolation_results[0]['mean_reward'] + interpolation_results[-1]['mean_reward']) / 2
    min_reward = min(rewards)
    instability = endpoint_avg - min_reward
    
    results = {
        'fork_id': fork_id,
        'fork1_reward': float(np.mean(fork1_rewards)),
        'fork2_reward': float(np.mean(fork2_rewards)),
        'interpolation': interpolation_results,
        'instability': float(instability),
        'is_stable': instability < 0.05 * endpoint_avg  # Considering stable if instability < 5% of average
    }
    
    # Save results
    with open(f"{exp_dir}/forks/instability_{fork_id}.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create experiment directories
    exp_dir, weights_dir = create_experiment_dir(args.env, args.algo, args.seed, config.get('experiment_name', 'NoExperimentNameGiven'))
    
    # Save experiment config
    with open(f"{exp_dir}/config.yaml", 'w') as f:
        yaml.dump({**config, **vars(args), "max_episode_steps": max_episode_steps, "num_eval_episodes": num_eval_episodes, "alphas": str(alphas.tolist())}, f)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create environment and set max episode steps
    env = gym.make(args.env)
    if args.max_episode_steps > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_episode_steps)
    
    agent = create_agent(env, args.algo, device)
    
    # Create replay buffer
    buffer = ReplayBuffer(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        capacity=int(config.get('buffer_size', DEFAULT_BUFFER_SIZE))
    )
    
    # Parse fork points
    fork_percentages = [float(p) for p in args.fork_points.split(',')]
    fork_steps = [int(p * int(args.total_steps)) for p in fork_percentages]
    
    # Initialize variables
    state, _ = env.reset(seed=args.seed)
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    done = False
    
    # Main training loop
    print(f"Starting training for {args.total_steps} steps")
    all_fork_results = []
    
    start_time = time.time()
    for t in range(1, int(args.total_steps) + 1):
        # Fork if at a fork point
        if t in fork_steps:
            fork_id = fork_steps.index(t)

            fork1_agent, fork2_agent = fork_training(env_name = args.env, 
                                                     agent = agent, 
                                                     algo_name = args.algo, 
                                                     fork_step = t, 
                                                     fork_id = fork_id, 
                                                     weights_dir = weights_dir, 
                                                     device = device, 
                                                     buffer = buffer, 
                                                     total_steps = args.total_steps
                                                    )
            
            # Analyze instability between the forks
            fork_result = analyze_instability(args.env, fork1_agent, fork2_agent, exp_dir, fork_id)
            all_fork_results.append(fork_result)
            
            # Print stability result
            stability_status = "STABLE" if fork_result['is_stable'] else "UNSTABLE"
            print(f"Fork {fork_id} at step {t} ({t/args.total_steps:.1%}) is {stability_status} with instability: {fork_result['instability']:.4f}")
        
        # Select action with exploration noise
        action = agent.get_action(state)
        
        # Perform action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store data in replay buffer
        buffer.add(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        episode_timesteps += 1
        
        # Update agent
        if len(buffer) > 10000: # SpinningUp suggests 10000 to "prevent learning from super sparse experience"
            agent.train_step(buffer)
        
        # Reset environment if done
        if done:
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # print number of steps every 1% of total steps
        if t % (args.total_steps // 100) == 0:
            print(f"Step {t}: {t / args.total_steps:.1%} of all (non-forked) run completed")
            
        # Save evaluation results
        if t % (args.total_steps // 100) == 0:
            eval_rewards = evaluate_policy(agent, gym.make(args.env), episodes=10)
            print(f"Step {t}: Evaluation over 10 episodes: {np.mean(eval_rewards):.3f}")
            with open(f"{exp_dir}/evaluations/step_{t}.json", 'w') as f:
                json.dump({
                    'step': t,
                    'mean_reward': float(np.mean(eval_rewards)),
                    'std_reward': float(np.std(eval_rewards)),
                    'rewards': [float(r) for r in eval_rewards]
                }, f, indent=4)
    
    # Save final model
    agent.save(f"{weights_dir}/final.pt")
    
    # Save all fork results in one file
    with open(f"{exp_dir}/all_forks_results.json", 'w') as f:
        json.dump(all_fork_results, f, indent=4)
    
    end_time = time.time()
    running_time = end_time - start_time

    # Append running_time to config.yaml
    with open(f"{exp_dir}/config.yaml", 'a') as f:
        yaml.dump({"running_time": running_time}, f)
    
    print(f"Training complete. Results saved to {exp_dir}")

if __name__ == "__main__":
    main()