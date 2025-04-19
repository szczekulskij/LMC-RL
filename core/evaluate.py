import copy
import torch
import torch.nn as nn
from agents.networks import ActorSAC, ActorDDPG

def evaluate_policy(agent, env, episodes=10):
    """Evaluate the policy over a number of episodes and return a list of returns."""
    returns = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        episode_return = 0
        while not done:
            action = agent.get_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
            state = next_state
        returns.append(episode_return)
    return returns

def linear_interpolation_policy(agent1, agent2, alpha, env, num_episodes=10):
    """
    Evaluate the performance of a policy obtained by linearly interpolating
    the weights of two agents (agent1 and agent2) with a factor alpha.

    Args:
        agent1: The first agent (e.g., SACAgent or DDPGAgent).
        agent2: The second agent (e.g., SACAgent or DDPGAgent).
        alpha: Interpolation factor (0.0 = agent1, 1.0 = agent2).
        env: The environment to evaluate the policy in.
        num_episodes: Number of episodes to evaluate.

    Returns:
        List of rewards obtained in each episode.
    """
    # Save the original weights of agent1
    agent1_state = copy.deepcopy(agent1.state_dict())

    # Interpolate weights with agent2
    agent1.interpolate_with_other_agent(agent2, alpha)

    # Evaluate the interpolated policy using the evaluate_policy function
    rewards = evaluate_policy(agent1, env, episodes=num_episodes)

    # Restore the original weights of agent1
    agent1.load_state_dict(agent1_state)

    return rewards