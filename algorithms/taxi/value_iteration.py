import gym
import numpy as np
from matplotlib import pyplot as plt


def value_iteration(env, gamma=1.0):
    v = np.zeros(env.observation_space.n)
    delta = 1e-5
    count = 0

    while True:
        prev_v = np.copy(v)
        for s in range(env.observation_space.n):
            action_values = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                action_values[a] = sum([p * (r + gamma * prev_v[s_prime]) for p, s_prime, r, _ in env.P[s][a]])

            v[s] = np.max(action_values)

        if np.sum(np.fabs(prev_v - v)) <= delta:
            print(f"Value-iteration converged after {count} steps")
            break

        count += 1

    return v


def extract_policy(env, v, gamma=1.0):
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            q_sa[a] = sum([p * (r + gamma * v[s_prime]) for p, s_prime, r, _ in env.P[s][a]])

        policy[s] = np.argmax(q_sa)

    return policy


def run_episode(env, policy, gamma=1.0):
    obs = env.reset()
    obs = obs[0]
    total_reward = 0
    step = 0
    while True:
        obs, reward, terminated, truncated, _ = env.step(int(policy[obs]))
        
        total_reward += gamma ** step * reward
        step += 1
        
        if terminated or truncated:
            break

    return total_reward


if __name__ == '__main__':
    env = gym.make("Taxi-v3")
    gamma = 0.9
    v = value_iteration(env, gamma)
    policy = extract_policy(env, v, gamma)
    scores = [run_episode(env, policy, gamma) for _ in range(100)]
    print(f"Average score: {np.mean(scores)}")
