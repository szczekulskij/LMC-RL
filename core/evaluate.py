import copy
import torch
import torch.nn as nn
from agents.networks import ActorSAC, ActorDDPG


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


def linear_interpolation_weights(model1, model2, alpha):
    """
    Create a new model with weights interpolated between model1 and model2.
    
    Args:
        model1: The first model
        model2: The second model
        alpha: Interpolation factor (0 = model1, 1 = model2)
        
    Returns:
        A new model with interpolated weights
    """
    interpolated_model = copy.deepcopy(model1)
    
    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()
    
    with torch.no_grad():
        for key in model1_state_dict:
            if torch.is_tensor(model1_state_dict[key]):
                interpolated_model.state_dict()[key].copy_(
                    alpha * model1_state_dict[key] + (1 - alpha) * model2_state_dict[key]
                )
    
    return interpolated_model

def linear_interpolation_policy(agent1, agent2, alpha, env, num_episodes=10):
    """
    Evaluates a policy with weights interpolated between two agents.
    
    Args:
        agent1: The first agent
        agent2: The second agent
        alpha: Interpolation factor (0 = agent1, 1 = agent2)
        env: The environment to evaluate on
        num_episodes: Number of episodes to evaluate
        
    Returns:
        List of rewards for each episode
    """
    interpolated_agent = linear_interpolation_weights(agent1, agent2, alpha)
    return evaluate_policy(interpolated_agent, env, num_episodes)

# Example usage underneath (the code doesn't run, just for illustration):
# if __name__ == "__impossibleu__":
#     if algorithm == "DDPG":
#         actorA = branch_A.actor
#         actorB = branch_B.actor
#     else:
#         actorA = branch_A.actor
#         actorB = branch_B.actor

#     # Use a fresh environment for evaluation (to avoid training environment state)
#     eval_env = gym.make(env_name)
#     set_seed(eval_env, 123)  # seed eval environment for consistency
#     alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
#     for alpha in alphas:
#         interp_actor = interpolate_policy(actorA, actorB, alpha)
#         avg_return = evaluate_policy(interp_actor, eval_env, episodes=10)
#         print(f"Interpolation {alpha:.2f}: Average Return = {avg_return:.2f}")
