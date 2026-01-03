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

def parse_args():
    default_config_path = 'core/experiment_default_config.yaml'
    with open(default_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser(description='Run LMC-RL experiments')
    # All experiment_default_config.yaml parameters can be overridden from command line
    parser.add_argument('--env', type=str, default=config['defaults']['env'], 
                        help='Gymnasium environment name')
    parser.add_argument('--algo', type=str, default=config['defaults']['algo'], choices=['SAC', 'DDPG'],
                        help='RL algorithm to use')
    parser.add_argument('--seed', type=int, default=random.randint(1, 1000000), help='Random seed')
    parser.add_argument('--total_steps', type=int, default=int(config['total_steps']), 
                        help='Total training steps')
    parser.add_argument('--eval_freq', type=int, default=config['eval_freq'], 
                        help='Evaluation frequency')
    parser.add_argument('--max_episode_steps', type=int, default=config['max_episode_steps'], 
                        help='Maximum number of steps per episode')
    parser.add_argument('--fork_points', type=str, default=config['default_fork_points'],
                        help='Comma-separated list of percentages of training to fork at')
    parser.add_argument('--fork_buffer_strategy', type=str, default=config['defaults']['fork_buffer_strategy'], 
                        choices=['copy', 'fresh', 'shared', 'split'],
                        help='Buffer handling strategy for forks')
    
    # Config parameters that can be overridden
    parser.add_argument('--buffer_size', type=int, default=config['buffer_size'],
                        help='Replay buffer size')
    parser.add_argument('--train_freq', type=int, default=config['train_freq'],
                        help='Training frequency (every N steps)')
    parser.add_argument('--num_eval_episodes', type=int, default=config['num_eval_episodes'],
                        help='Number of episodes for evaluation')
    parser.add_argument('--alpha_points', type=int, default=config['alpha_points'],
                        help='Number of alpha points for linear interpolation')
    parser.add_argument('--experiment_name', type=str, default=config['experiment_name'],
                        help='Experiment name for organizing results')
    
    # Config file override
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    
    return parser.parse_args()

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

def fork_training(env_name, agent, algo_name, fork_step, fork_id, weights_dir, device, buffer, total_steps, fork_buffer_strategy, args):
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
    
    # Handle buffer strategy for forks
    if fork_buffer_strategy == "copy":
        # Both forks get identical copies of the original buffer
        fork1_buffer = deepcopy(buffer)
        fork2_buffer = deepcopy(buffer)
    elif fork_buffer_strategy == "fresh":
        # Both forks start with empty buffers
        fork1_buffer = ReplayBuffer(
            state_dim=buffer.state_dim,
            action_dim=buffer.action_dim,
            capacity=buffer.capacity
        )
        fork2_buffer = ReplayBuffer(
            state_dim=buffer.state_dim,
            action_dim=buffer.action_dim,
            capacity=buffer.capacity
        )
    elif fork_buffer_strategy == "shared":
        # Both forks use the same buffer (shared experience)
        fork1_buffer = buffer
        fork2_buffer = buffer
    elif fork_buffer_strategy == "split":
        # Split the original buffer between the two forks
        fork1_buffer = deepcopy(buffer)
        fork2_buffer = deepcopy(buffer)
        # TODO: Implement actual buffer splitting logic
        # This would require modifying ReplayBuffer to support splitting
        print("WARNING: 'split' buffer strategy not fully implemented yet, using 'copy' instead")
    else:
        raise ValueError(f"Unknown fork_buffer_strategy: {fork_buffer_strategy}")
    
    # Set seeds for buffers if they support it (only for non-shared buffers)
    if fork_buffer_strategy != "shared":
        if hasattr(fork1_buffer, 'set_seed'):
            fork1_buffer.set_seed(fork1_seed)
        if hasattr(fork2_buffer, 'set_seed'):
            fork2_buffer.set_seed(fork2_seed)
    # Train fork 1
    set_seed(fork1_seed)
    fork1_env = gym.make(env_name)
    print(f"Training fork 1 (seed: {fork1_seed}) from step {fork_step} to {total_steps}")
    train_steps = int(total_steps - fork_step)
    state, _ = fork1_env.reset(seed=fork1_seed)
    for step in range(train_steps):
        action = fork1_agent.get_action(state)
        next_state, reward, terminated, truncated, _ = fork1_env.step(action)
        done = terminated or truncated
        
        fork1_buffer.add(state, action, reward, next_state, done)
        state = next_state if not done else fork1_env.reset()[0]
        
        if len(fork1_buffer) > 10000 and step % args.train_freq == 0: # SpinningUp suggests 10000 to "prevent learning from super sparse experience"
            fork1_agent.train_step(fork1_buffer)
        
        if done:
            state, _ = fork1_env.reset()
        
        # Print progress for fork 1
        if step % (train_steps // 10) == 0:  # Print every 10% of progress
            print(f"Fork 1 (ID: {fork_id}): {step / train_steps:.1%} completed")
            evaluation_rewards = evaluate_policy(fork1_agent, fork1_env, episodes=3)
            print(f"Fork 1 (ID: {fork_id}) evaluation reward: {np.mean(evaluation_rewards):.3f}")
    
    
    total_steps = int(total_steps) #TODO: Fix to int upstream
    # Train fork 2 (similar to fork 1 but with different seed)
    set_seed(fork2_seed)
    fork2_env = gym.make(env_name)
    
    print(f"Training fork 2 (seed: {fork2_seed}) from step {fork_step} to {total_steps}")
    state, _ = fork2_env.reset(seed=fork2_seed)
    for step in range(train_steps):
        action = fork2_agent.get_action(state)
        next_state, reward, terminated, truncated, _ = fork2_env.step(action)
        done = terminated or truncated
        
        fork2_buffer.add(state, action, reward, next_state, done)
        state = next_state if not done else fork2_env.reset()[0]
        
        if len(fork2_buffer) > 10000 and step % args.train_freq == 0: # SpinningUp suggests 10000 to "prevent learning from super sparse experience"
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

def analyze_instability(env_name, fork1_agent, fork2_agent, exp_dir, fork_id, fork_point, args):
    """Analyze instability between two forked agents using linear interpolation."""
    print(f"Analyzing instability for fork {fork_id}")
    
    # Create evaluation environment
    eval_env = gym.make(env_name)
    
    # Get evaluation parameters from args
    num_eval_episodes = args.num_eval_episodes
    alphas = np.linspace(0, 1, args.alpha_points)
    
    # Evaluate individual agents
    fork1_rewards = evaluate_policy(fork1_agent, eval_env, episodes=num_eval_episodes)
    fork2_rewards = evaluate_policy(fork2_agent, eval_env, episodes=num_eval_episodes)
    
    # Linear interpolation between the two agents
    interpolation_results = []
    
    for alpha in alphas:
        print(f"Analyzing instability for fork {fork_id} | alpha: {alpha}")
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
        'fork_point': float(fork_point),
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
    if torch.backends.mps.is_available():
        device = torch.device("mps") 
    elif torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu") 
    print(f"Using device: {device}")
    
    # Create experiment directories
    exp_dir, weights_dir = create_experiment_dir(args.env, args.algo, args.seed, args.experiment_name)
    print(f"Will save all models, results, and etc. within {exp_dir}")
    
    # Create config to save - only the actual values used in the experiment
    config_to_save = {
        # Core experiment parameters (actual values used)
        "env": args.env,
        "algo": args.algo,
        "seed": args.seed,
        "total_steps": args.total_steps,
        "eval_freq": args.eval_freq,
        "max_episode_steps": args.max_episode_steps,
        "fork_points": args.fork_points,
        "fork_buffer_strategy": args.fork_buffer_strategy,
        
        # Training parameters
        "buffer_size": args.buffer_size,
        "train_freq": args.train_freq,
        "num_eval_episodes": args.num_eval_episodes,
        "alpha_points": args.alpha_points,
        "alphas": [round(x, 2) for x in np.linspace(0, 1, args.alpha_points).tolist()],
        
        # Experiment metadata
        "experiment_name": args.experiment_name
    }
    
    # Save config as JSON with alphas inline
    with open(f"{exp_dir}/config.json", 'w') as f:
        # First save normally to get the structure
        json_str = json.dumps(config_to_save, indent=2)
        
        # Find the alphas array and make it inline
        import re
        # Replace the multi-line alphas array with inline version
        alphas_pattern = r'"alphas": \[\s*([^]]+)\s*\]'
        alphas_match = re.search(alphas_pattern, json_str, re.DOTALL)
        if alphas_match:
            # Extract the numbers and format them inline
            alphas_content = alphas_match.group(1)
            numbers = re.findall(r'[\d.]+', alphas_content)
            inline_alphas = '[' + ','.join(numbers) + ']'
            json_str = re.sub(alphas_pattern, f'"alphas": {inline_alphas}', json_str, flags=re.DOTALL)
        
        f.write(json_str)
    
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
        capacity=int(args.buffer_size)
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
                                                     total_steps = args.total_steps,
                                                     fork_buffer_strategy = args.fork_buffer_strategy,
                                                     args = args
                                                    )
            
            # Analyze instability between the forks
            fork_result = analyze_instability(args.env, fork1_agent, fork2_agent, exp_dir, fork_id, fork_point=t/args.total_steps, args=args)
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
        if len(buffer) > 10000 and t % args.train_freq == 0: # SpinningUp suggests 10000 to "prevent learning from super sparse experience"
            agent.train_step(buffer)
        
        # Reset environment if done
        if done:
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # print number of steps every 1% of total steps
        if t % (args.total_steps // 1000) == 0:
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

    # Add running_time to config.json
    with open(f"{exp_dir}/config.json", 'r') as f:
        config_data = json.load(f)
    
    config_data["running_time"] = running_time
    
    # Save with alphas inline formatting
    with open(f"{exp_dir}/config.json", 'w') as f:
        json_str = json.dumps(config_data, indent=2)
        
        # Find the alphas array and make it inline
        import re
        alphas_pattern = r'"alphas": \[\s*([^]]+)\s*\]'
        alphas_match = re.search(alphas_pattern, json_str, re.DOTALL)
        if alphas_match:
            alphas_content = alphas_match.group(1)
            numbers = re.findall(r'[\d.]+', alphas_content)
            inline_alphas = '[' + ','.join(numbers) + ']'
            json_str = re.sub(alphas_pattern, f'"alphas": {inline_alphas}', json_str, flags=re.DOTALL)
        
        f.write(json_str)
    
    print(f"Training complete. Results saved to {exp_dir}")

if __name__ == "__main__":
    main()