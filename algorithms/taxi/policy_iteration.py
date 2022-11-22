import gym
import gym.wrappers
import numpy as np
from matplotlib import pyplot as plt


def policy_evaluation(env, policy, gamma=1.0):
    v = np.zeros(env.observation_space.n)
    epsilon = 1e-10
    count = 0
    while True:
        prev_v = np.copy(v)
        for s in range(env.observation_space.n):
            a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_prime]) for p, s_prime, r, _ in env.P[s][a]])

        count += 1
        if np.sum(np.fabs(prev_v - v)) <= epsilon:
            break

    return v


def policy_improvement(env, v, gamma=1.0):
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            q_sa[a] = sum(p * r + gamma * v[s_prime] for p, s_prime, r, _ in env.P[s][a])
        
        policy[s] = np.argmax(q_sa)
    
    return policy


def policy_iteration(env, gamma=1.0):
    policy = np.random.choice(env.action_space.n, size=(env.observation_space.n))

    step = 0
    while True:
        v = policy_evaluation(env, policy, gamma)
        new_policy = policy_improvement(env, v, gamma)

        if np.all(policy == new_policy):
            print(f"Policy-Iteration converged at step {step}")
            print(v)
            break

        step += 1
        policy = new_policy

    return new_policy


def run_episode(env, policy, gamma=1.0, render=False):
    obs = env.reset()
    obs = obs[0]
    total_reward = 0
    step = 0
    while True:
        if render:
            env.render()

        obs, reward, terminated, truncated, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step * reward)
        step += 1

        if terminated or truncated:
            break

    return total_reward


if __name__ == '__main__':
    env = gym.make('Taxi-v3', render_mode="rgb_array")
    # env = gym.wrappers.RecordVideo(env, 'video')
    gamma = 0.9

    optimal_policy = policy_iteration(env, gamma)
    scores = [run_episode(env, optimal_policy, gamma) for _ in range(100)]
    print(f"Average score: {np.mean(scores)}")
    plt.plot(scores)
    plt.show()
