import pandas as pd
import numpy as np
from collections import namedtuple
from matplotlib import pyplot as plt
import os


EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def plot_stats(stats, smoothing_window=10, save=False, file_prefix="x", dir="images"):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")

    current_dir = os.path.dirname(__file__)
    results_dir = os.path.join(current_dir, f"{dir}")

    if not os.path.isdir(results_dir) and save:
        os.makedirs(results_dir)

    if save:
        plt.savefig(f"{results_dir}/{file_prefix}_episode_length_over_time.png")

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = (
        pd.Series(stats.episode_rewards)
        .rolling(smoothing_window, min_periods=smoothing_window)
        .mean()
    )
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title(
        f"Episode Reward over Time (Smoothed over window size {smoothing_window})"
    )

    if save:
        plt.savefig(f"{results_dir}/{file_prefix}_episode_reward_over_time.png")

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(
        np.cumsum(stats.episode_lengths),
        np.arange(len(stats.episode_lengths)),
    )
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if save:
        plt.savefig(f"{results_dir}/{file_prefix}_episode_per_time_step.png")

    return fig1, fig2, fig3
