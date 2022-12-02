import numpy as np


class EpsilonGreedyPolicy:
    def __init__(self, Q, nA, epsilon=1.0):
        self.Q = Q
        self.nA = nA
        self.policy = None
        self.epsilon = epsilon

    def get_action(self, observation):
        A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(self.Q[observation])
        A[best_action] += 1 - self.epsilon
        return np.random.choice(len(A), p=A)
