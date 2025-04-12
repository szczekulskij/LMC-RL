# experiments/plot_forks.py

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from glob import glob


def load_fork_files(forks_dir):
    fork_files = sorted(glob(os.path.join(forks_dir, "instability_*.json")))
    all_results = []
    for file in fork_files:
        with open(file, 'r') as f:
            data = json.load(f)
            data['fork_step'] = int(file.split('_')[-1].split('.')[0])  # assumes fork_id == fork_step
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
    fork_steps = [f['fork_step'] for f in fork_results]
    fork_times = [step / total_steps for step in fork_steps]
    instabilities = [f['instability'] for f in fork_results]

    plt.figure()
    plt.plot(fork_times, instabilities, marker='o')
    plt.xlabel("Fork point (% of training)")
    plt.ylabel("Instability")
    plt.title("Instability Over Time")
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