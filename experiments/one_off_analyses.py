# One-off analysis script for Linear Mode Connectivity (LMC) between two models.
# Input is the two models' weights, the environment name (and the model type).
# The script evaluate linear interpolation path between two given models

import os
import json
import argparse
import numpy as np
import torch
import gymnasium as gym
from core.evaluate import linear_interpolation_policy
from agents.sac import SACAgent
from agents.ddpg import DDPGAgent


def load_agent(model_name, state_dim, action_dim, device, weights_dir):
    """Load an agent (SAC or DDPG) with weights from a specified directory."""
    if model_name == "SAC":
        agent = SACAgent(state_dim, action_dim, device=device)
    elif model_name == "DDPG":
        agent = DDPGAgent(state_dim, action_dim, device=device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    weights_path = os.path.join(weights_dir, "final.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")
    else:
        print(f"Loading model weights from: {weights_path}")
    
    agent.load(weights_path)
    return agent


def lmc_analysis(env_name, alpha_range, model1_weights_dir, model2_weights_dir, model_name, num_eval_episodes=10, output_dir="lmc_results"):
    """Perform Linear Mode Connectivity (LMC) analysis between two models."""
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    print("Loading models...")
    model1 = load_agent(model_name, state_dim, action_dim, device, model1_weights_dir)
    model2 = load_agent(model_name, state_dim, action_dim, device, model2_weights_dir)

    # Perform LMC analysis
    print("Performing LMC analysis...")
    results = []
    for alpha in alpha_range:
        print(f"Evaluating interpolation at alpha={alpha:.2f}...")
        try:
            rewards = linear_interpolation_policy(model1, model2, alpha, env, num_episodes=num_eval_episodes)
            results.append({
                "alpha": alpha,
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "rewards": [float(r) for r in rewards]
            })
        except Exception as e:
            print(f"Error during evaluation at alpha={alpha:.2f}: {e}")
            continue

    # Ensure results are sorted by alpha
    results = sorted(results, key=lambda x: x["alpha"])

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "lmc_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"LMC results saved to: {results_path}")

    # Plot results
    try:
        import matplotlib.pyplot as plt
        alphas = [r["alpha"] for r in results]
        mean_rewards = [r["mean_reward"] for r in results]
        std_rewards = [r["std_reward"] for r in results]

        plt.figure(figsize=(8, 6))
        plt.plot(alphas, mean_rewards, label="Mean Reward", marker="o")
        plt.fill_between(alphas, np.array(mean_rewards) - np.array(std_rewards), 
                         np.array(mean_rewards) + np.array(std_rewards), alpha=0.2, label="Std Dev")
        plt.xlabel("Interpolation α")
        plt.ylabel("Reward")
        plt.title("Linear Mode Connectivity Analysis")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "lmc_plot.png")
        plt.savefig(plot_path)
        print(f"LMC plot saved to: {plot_path}")
    except ImportError:
        print("Matplotlib not installed. Skipping plot generation.")


def main():
    parser = argparse.ArgumentParser(description="Perform Linear Mode Connectivity (LMC) analysis.")
    parser.add_argument("--env", type=str, required=True, help="Gymnasium environment name.")
    default_alpha = np.linspace(0.0, 1.0, 101) 
    default_alpha = ",".join([str(a) for a in default_alpha])
    # default_alpha = "0.0,0.1,0.2,0.5,0.8,1.0"
    parser.add_argument("--alpha_range", type=str, default=default_alpha,
                        help="Comma-separated list of alpha values for interpolation.")
    parser.add_argument("--model1_weights_dir", type=str, required=True, help="Directory of the first model's weights.")
    parser.add_argument("--model2_weights_dir", type=str, required=True, help="Directory of the second model's weights.")
    parser.add_argument("--model_name", type=str, required=True, choices=["SAC", "DDPG"], help="Model type (SAC or DDPG).")
    parser.add_argument("--num_eval_episodes", type=int, default=10, help="Number of episodes for evaluation.")
    parser.add_argument("--output_dir", type=str, default="lmc_results", help="Directory to save LMC results.")
    args = parser.parse_args()

    alpha_range = [float(a) for a in args.alpha_range.split(",")]
    lmc_analysis(args.env, alpha_range, args.model1_weights_dir, args.model2_weights_dir, 
                 args.model_name, args.num_eval_episodes, args.output_dir)


if __name__ == "__main__":
    main()
