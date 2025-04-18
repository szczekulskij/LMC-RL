# experiments/plot_forks.py

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from glob import glob


def load_fork_files(forks_dir):
    """Load all fork result JSON files from the specified directory."""
    fork_files = sorted(glob(os.path.join(forks_dir, "instability_*.json")))
    print(f"Searching for fork files in: {forks_dir}")
    print(f"Found fork files: {fork_files}")

    all_results = []
    for file in fork_files:
        with open(file, 'r') as f:
            data = json.load(f)
            # Extract fork step from the filename (e.g., instability_1.json -> fork_step = 1)
            try:
                fork_step = int(os.path.basename(file).split('_')[-1].split('.')[0])
                data['fork_step'] = fork_step
            except ValueError:
                print(f"Warning: Could not extract fork step from filename: {file}")
                continue
            all_results.append(data)
    return all_results


def plot_all_interpolations(fork_results, out_dir):
    plt.figure()
    for fork in fork_results:
        alpha = [pt['alpha'] for pt in fork['interpolation']]
        rewards = [pt['mean_reward'] for pt in fork['interpolation']]
        plt.plot(alpha, rewards, label=f"Fork {fork['fork_id']}")

    plt.xlabel("Interpolation α")
    plt.ylabel("Mean Reward")
    plt.title("All Fork Interpolation Curves")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/all_interpolations.png")
    plt.close()


def plot_individual_interpolations(fork_results, out_dir):
    for fork in fork_results:
        alpha = [pt['alpha'] for pt in fork['interpolation']]
        rewards = [pt['mean_reward'] for pt in fork['interpolation']]
        stds = [pt['std_reward'] for pt in fork['interpolation']]

        plt.figure()
        plt.plot(alpha, rewards, label="Mean Reward")
        plt.fill_between(alpha, np.array(rewards)-np.array(stds), np.array(rewards)+np.array(stds), alpha=0.2)
        plt.xlabel("Interpolation α")
        plt.ylabel("Mean Reward")
        plt.title(f"Fork {fork['fork_id']}: Interpolation")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/interp_fork_{fork['fork_id']}.png")
        plt.close()


def plot_instability_over_time(fork_results, total_steps, out_dir):
    """Plot instability over time with proper scaling and visualization."""
    fork_steps = [f['fork_step'] for f in fork_results]
    fork_times = [step / total_steps for step in fork_steps]
    instabilities = [f['instability'] for f in fork_results]

    # Ensure data is sorted by fork time
    sorted_indices = np.argsort(fork_times)
    fork_times = np.array(fork_times)[sorted_indices]
    instabilities = np.array(instabilities)[sorted_indices]

    plt.figure(figsize=(8, 6))
    plt.plot(fork_times, instabilities, marker='o', label="Instability")
    plt.xlabel("Fork point (% of training)")
    plt.ylabel("Instability")
    plt.title("Instability Over Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/instability_over_time.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Path to experiment folder (containing /forks)")
    parser.add_argument("--total_steps", type=int, default=1_000_000, help="Total training steps")
    args = parser.parse_args()

    forks_dir = os.path.join(args.results_dir, "forks")
    out_dir = os.path.join(args.results_dir, "fork_analysis")
    os.makedirs(out_dir, exist_ok=True)

    fork_results = load_fork_files(forks_dir)

    if not fork_results:
        raise RuntimeError("No fork results found!")

    plot_all_interpolations(fork_results, out_dir)
    plot_individual_interpolations(fork_results, out_dir)
    plot_instability_over_time(fork_results, args.total_steps, out_dir)

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()