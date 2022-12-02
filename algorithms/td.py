from collections import defaultdict
import numpy as np
import sys
import itertools


from policies import EpsilonGreedyPolicy
from utils import EpisodeStats


class QLearning:
    def __init__(
        self,
        policy,
        env,
        max_episodes,
        learning_rate=0.5,
        discount_rate=1.0,
        epsilon=1.0,
    ):
        self.policy = policy
        self.env = env
        self.max_episodes = max_episodes
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon

        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.stats = EpisodeStats(
            episode_lengths=np.zeros(self.max_episodes),
            episode_rewards=np.zeros(self.max_episodes),
        )
        self._setup_model()

    def _setup_model(self):
        if not self.policy == "EpsilonGreedyPolicy":
            raise ValueError(f"Error: The policy {self.policy} is not implemented.")

        self.policy = EpsilonGreedyPolicy(self.Q, self.env.action_space.n, self.epsilon)

    def learn(self):
        for i in range(self.max_episodes):
            if (i + 1) % 100 == 0:
                print(f"\r Episode: {i}/ {self.max_episodes}", end="")
                sys.stdout.flush()

            obs, _ = self.env.reset()

            for t in itertools.count():
                action = self.policy.get_action(obs)
                next_obs, reward, truncated, terminated, _ = self.env.step(action)

                self.stats.episode_rewards[i] += reward
                self.stats.episode_lengths[i] = t

                best_next_action = np.argmax(self.Q[next_obs])
                td_target = (
                    reward + self.discount_rate * self.Q[next_obs][best_next_action]
                )
                td_delta = td_target - self.Q[obs][action]
                self.Q[obs][action] += self.learning_rate * td_delta

                if truncated or terminated:
                    break

                obs = next_obs

        print()  # for printing a new line after the episode count

    def predict(self, observation):
        return np.argmax(self.Q[observation])


class Sarsa:
    def __init__(
        self,
        policy,
        env,
        max_episodes,
        learning_rate=0.5,
        discount_rate=1.0,
        epsilon=1.0,
    ):
        self.policy = policy
        self.env = env
        self.max_episodes = max_episodes
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon

        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.stats = EpisodeStats(
            episode_lengths=np.zeros(self.max_episodes),
            episode_rewards=np.zeros(self.max_episodes),
        )
        self._setup_model()

    def _setup_model(self):
        if not self.policy == "EpsilonGreedyPolicy":
            raise ValueError(f"Error: The policy {self.policy} is not implemented.")

        self.policy = EpsilonGreedyPolicy(self.Q, self.env.action_space.n, self.epsilon)

    def learn(self):
        for i in range(self.max_episodes):
            if (i + 1) % 100 == 0:
                print(f"\rEpisode: {i + 1}/{self.max_episodes}", end="")
                sys.stdout.flush()

            obs, _ = self.env.reset()
            action = self.policy.get_action(obs)

            for t in itertools.count():
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_action = self.policy.get_action(next_obs)

                self.stats.episode_rewards[i] += reward
                self.stats.episode_lengths[i] = t

                td_target = reward + self.discount_rate * self.Q[next_obs][next_action]
                td_delta = td_target - self.Q[obs][action]
                self.Q[obs][action] += self.learning_rate * td_delta

                obs = next_obs
                action = next_action

                if terminated or truncated:
                    break

    def predict(self, observation):
        return np.argmax(self.Q[observation])
