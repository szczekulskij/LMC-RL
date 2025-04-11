import copy
import gym
import torch
import torch.nn as nn
import numpy as np
from agents.sac import SACAgent
from agents.ddpg import DDPGAgent
from agents.networks import ActorSAC, ActorDDPG
from replay_buffer import ReplayBuffer
from utils import set_seed  # if a utility function is available
# Device configuration: use MPS if available (on Apple silicon Macs)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")


def interpolate_policy(actorA: nn.Module, actorB: nn.Module, alpha: float):
    """Linearly interpolate between two actor networks' weights."""
    assert type(actorA) == type(actorB), "Actor networks must be of same class/shape"
    new_actor = copy.deepcopy(actorA)
    # Interpolate each parameter
    for param, paramA, paramB in zip(new_actor.parameters(), actorA.parameters(), actorB.parameters()):
        param.data.copy_( (1 - alpha) * paramA.data + alpha * paramB.data )
    return new_actor

def evaluate_policy(actor: nn.Module, env, episodes: int = 5):
    """Run policy (actor network) for given number of episodes and return average return."""
    total_return = 0.0
    for ep in range(episodes):
        state, _ = env.reset()  # reset environment
        done = False
        ep_return = 0.0
        while not done:
            # Get action from actor. If actor is stochastic (SAC), use deterministic mode for evaluation:
            if isinstance(actor, ActorSAC) or isinstance(actor, ActorDDPG):
                # If it's one of our actor classes, we can forward and handle accordingly
                if isinstance(actor, ActorSAC):
                    mean, log_std = actor(torch.tensor(state, dtype=torch.float32))
                    action = torch.tanh(mean)  # deterministic action (mean)
                else:  # DDPG actor
                    action = actor(torch.tensor(state, dtype=torch.float32))
                action = action.detach().cpu().numpy()
            else:
                # If a full agent object is passed instead of actor network
                action = actor.get_action(state, deterministic=True) 
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            state = next_state
        total_return += ep_return
    avg_return = total_return / episodes
    return avg_return


# Example usage underneath (the code doesn't run, just for illustration):
if __name__ == "__impossibleu__":
    if algorithm == "DDPG":
        actorA = branch_A.actor
        actorB = branch_B.actor
    else:
        actorA = branch_A.actor
        actorB = branch_B.actor

    # Use a fresh environment for evaluation (to avoid training environment state)
    eval_env = gym.make(env_name)
    set_seed(eval_env, 123)  # seed eval environment for consistency
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    for alpha in alphas:
        interp_actor = interpolate_policy(actorA, actorB, alpha)
        avg_return = evaluate_policy(interp_actor, eval_env, episodes=10)
        print(f"Interpolation {alpha:.2f}: Average Return = {avg_return:.2f}")
