from collections import defaultdict
import numpy as np
import pickle
import os


class Agent:
    name = "Generic agent"

    def __init__(self, env, discount=.996, episodes=6000, epsilon=.9, primary_reward='env', policy_idx=0):
        self.actions = range(env.action_spec().maximum + 1)

        # Exploration probabilities for other actions, given that the action k is greedy
        self.probs = [[1.0 / (len(self.actions) - 1) if i != k else 0 for i in self.actions] for k in self.actions]

        # Initialize a tabular Q-function
        self.Q = defaultdict(lambda: np.zeros(len(self.actions)))
        self.primary_reward = primary_reward  # Set the agent's reward function

        self.discount = discount  # The discount rate, in [0,1)
        self.epsilon = epsilon  # Percentage of the time that the agent takes a greedy action

        self.episodes = episodes  # The number of training episodes

        # Metadata for saving Q-function
        self.policy_idx = str(policy_idx)
        self.game_name = env.name

    def save(self):
        write_name = 'q_functions/policy_' + self.name + '_' + self.game_name + '_' + self.policy_idx + '.pkl'
        if not os.path.exists('./q_functions/'):
            os.makedirs('./q_functions/')
        with open(write_name, 'wb') as f:
            pickle.dump(dict(self.Q), f)

    def act(self, obs):
        return self.Q[str(obs['board'])].argmax()

    def behavior_action(self, board):
        """Returns the e-greedy action for the state board string."""
        greedy = self.Q[board].argmax()
        if np.random.random() < self.epsilon or len(self.actions) == 1:  # Choose greedy action
            return greedy
        else:  # choose anything else
            return np.random.choice(self.actions, p=self.probs[greedy])