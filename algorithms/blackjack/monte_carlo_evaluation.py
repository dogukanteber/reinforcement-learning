import gym
from collections import defaultdict
import numpy as np
import sys
from dataclasses import dataclass


@dataclass(unsafe_hash=True)
class RLExperience:
    """A data-type to represent an experience in an episode

    Returns:
        RLExperience object: state, action, reward
    """

    state: tuple
    action: int
    reward: int

    def __repr__(self) -> str:
        return f"state={self.state}, action={self.action}, reward={self.reward}"


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
        policy (func): policy function that gives the action by providing the current state

    Returns:
        list: containing set of RLExperience ie. an episode
    """
    observation = env.reset()[0]
    episode = list()
    while True:
        action = policy(observation)
        new_observation, reward, terminated, truncated, _ = env.step(action)
        episode.append(RLExperience(observation, action, reward))

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
    first_visit_index = next(
        index for index, exp in enumerate(episode) if exp.state == state
    )
    return sum(
        [
            (gamma**index) * exp.reward
            for index, exp in enumerate(episode[first_visit_index:])
        ]
    )


def monte_carlo_prediction(env, policy, max_iteration=10000, gamma=1.0):
    """Runs Monte Carlo Prediction First-Visit algorithm

    Args:
        env (object): gym.env object
        policy (func): policy to evaluate
        max_iteration (int, optional): the max number of iteration. Defaults to 10000.
        gamma (float, optional): discount rate. Defaults to 1.0.

    Returns:
        defaultdict: state-value table
    """
    V = defaultdict(float)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    for i in range(max_iteration):
        if i % 1000 == 0:
            print(f"\rEpisode {i}/{max_iteration}.", end="")
            sys.stdout.flush()

        episode = generate_episode(env, policy)
        states = set([exp.state for exp in episode])
        for s in states:
            G = calculate_return(episode, s, gamma)
            returns_sum[s] += G
            returns_count[s] += 1.0
            V[s] = returns_sum[s] / returns_count[s]

    return V


if __name__ == "__main__":
    env = gym.make("Blackjack-v1", natural=False, sab=False)
    V = monte_carlo_prediction(env, sample_policy, max_iteration=10000)
