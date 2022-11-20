import gym
from collections import defaultdict
import numpy as np
import matplotlib
import sys
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def sample_policy(observation):
    """Generates a sample policy for the given observation

    Args:
        observation (set): contains score, dealer's score and usable ace

    Returns:
        int: stick (0) if the score is 20 or above, hit (1) otherwise
    """
    score, _, _ = observation
    return 0 if score >= 20 else 1


def generate_episode(env, policy):
    """Generates an episode according to the given policy

    Args:
        env (gym.env): environment object
        policy (callable): policy function that gives the action by providing the current state

    Returns:
        list: containing set of values ie. an episode
    """
    observation = env.reset()[0]
    episode = list()
    while True:
        action = policy(observation)
        new_observation, reward, terminated, truncated, _ = env.step(action)
        episode.append((observation, action, reward))

        if terminated or truncated:
            break

        observation = new_observation

    return episode


def calculate_return(episode, state, gamma):
    """Calculates return for the given episode starting from the given state

    Args:
        episode (list): containing set of (observation, action, reward) information
        state (set): state information (score, dealer's score, usable ace)
        gamma (float): discount rate

    Returns:
        dict: state-value table
    """
    first_visit_index = next(i for i, x in enumerate(episode) if x[0] == state)
    return sum([(gamma ** i) * x[2]  for i, x in enumerate(episode[first_visit_index:])])


def monte_carlo_prediction(env, policy, max_iteration=10000, gamma=1.0):
    V = defaultdict(float)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    for i in range(max_iteration):
        if i % 1000 == 0:
            print("\rEpisode {}/{}.".format(i, max_iteration), end="")
            sys.stdout.flush()
        episode = generate_episode(env, policy)
        states = set([obs[0] for obs in episode])
        for s in states:
            G = calculate_return(episode, s, gamma)
            returns_sum[s] += G
            returns_count[s] += 1.0
            V[s] = returns_sum[s] / returns_count[s]

    return V
        

def plot_value_function(V, title="Value Function"):
    # Taken from https://github.com/dennybritz/reinforcement-learning/blob/master/lib/plotting.py
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())


    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))


if __name__ == '__main__':
    env = gym.make("Blackjack-v1", natural=False, sab=False)
    obs = env.reset()
    
    V = monte_carlo_prediction(env, sample_policy, max_iteration=10000)
    plot_value_function(V)



"""Generates an episode according to the given policy. The caller can also
provide the initial state and the first action

Args:
    env (object): environment
    policy (func): policy function that gives the action by providing the current state
    state (int, optional): initial state, defaults to None.
    action (int, optional): first action, defaults to None.

Returns:
    list: containing set of values ie. an episode
"""
