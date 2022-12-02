import gym
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


# if __name__ == "__main__":
#     env = gym.make("CliffWalking-v0")
#     model = QLearning(
#         "EpsilonGreedyPolicy",
#         env,
#         500,
#         learning_rate=0.5,
#         discount_rate=1.0,
#         epsilon=0.1,
#     )
#     model.learn()
#     plot_stats(model.stats)
