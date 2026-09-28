"""Tabular Q-Learning Temporal Difference Agent
100% Python Standard Library (random).
"""

import random

class TabularQLearningAgent:
    """Model-free off-policy TD control Q-learning agent."""
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}

    def get_q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_vals = [self.get_q(state, a) for a in self.actions]
        max_q = max(q_vals)
        best_actions = [a for a, q in zip(self.actions, q_vals) if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state):
        current_q = self.get_q(state, action)
        next_max_q = max(self.get_q(next_state, a) for a in self.actions)
        td_target = reward + self.gamma * next_max_q
        self.Q[(state, action)] = current_q + self.alpha * (td_target - current_q)

    def export_q_table(self):
        return {f"{s}::{a}": round(v, 4) for (s, a), v in self.Q.items()}
