import numpy as np


class ValueIteration:
    def __init__(self, env, delta=1e-5, gamma=1.0):
        self.env = env
        self.delta = delta
        self.gamma = gamma
        self.nA = env.action_space.n
        self.nS = env.observation_space.n
        self.V = np.zeros(self.nS)
        self.policy = np.zeros(self.nS, dtype=int)

    def learn(self):
        count = 0
        while True:
            prev_V = np.copy(self.V)
            for state in range(self.nS):
                action_values = np.zeros(self.nA)
                for action in range(self.nA):
                    action_values[action] = sum(
                        [
                            prob * (reward + self.gamma * prev_V[next_state])
                            for prob, next_state, reward, _ in self.senv.P[state][
                                action
                            ]
                        ]
                    )
                self.V[state] = np.max(action_values)

            if np.sum(np.fabs(prev_V - self.V)) <= self.delta:
                print(f"Value Iteration converged after {count} steps")
                break
            count += 1

        self._extract_policy()

    def _extract_policy(self):
        for state in range(self.nS):
            q_sa = np.zeros(self.nS)
            for action in range(self.nA):
                q_sa[action] = sum(
                    [
                        prob * (reward + self.gamma * self.V[next_state])
                        for prob, next_state, reward, _ in self.env.P[state][action]
                    ]
                )
            self.policy[state] = np.argmax(q_sa)

    def predict(self, observation):
        return self.policy[observation]


class PolicyIteration:
    def __init__(self, env, delta=1e-5, gamma=1.0):
        self.env = env
        self.delta = delta
        self.gamma = gamma
        self.nA = env.action_space.n
        self.nS = env.observation_space.n
        self.V = np.zeros(self.nS)
        self.policy = np.random.choice(self.nA, size=(self.nS))

    def learn(self):
        count = 0
        while True:
            self._policy_evaluation()
            new_policy = self._policy_improvement()

            count += 1

            if np.all(self.policy == new_policy):
                print(f"Policy-Iteration converged at step {count}")
                break

            self.policy = new_policy

    def _policy_evaluation(self):
        while True:
            prev_V = np.copy(self.V)
            for state in range(self.nS):
                action = self.policy[state]
                self.V[state] = sum(
                    [
                        prob * (reward + self.gamma * prev_V[next_state])
                        for prob, next_state, reward, _ in self.env.P[state][action]
                    ]
                )

            if np.sum(np.fabs(prev_V - self.V)) <= self.delta:
                break

    def _policy_improvement(self):
        new_policy = np.zeros(self.nS, dtype=int)
        for state in range(self.nS):
            q_sa = np.zeros(self.nA)
            for action in range(self.nA):
                q_sa[action] = sum(
                    prob * (reward + self.gamma * self.V[next_state])
                    for prob, next_state, reward, _ in self.env.P[state][action]
                )
            new_policy[state] = np.argmax(q_sa)

        return new_policy

    def predict(self, observation):
        return self.policy[observation]
