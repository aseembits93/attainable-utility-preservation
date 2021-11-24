from collections import defaultdict
import numpy as np
import pickle
import os 

class QLearner:
    name = "Q-learner"

    def __init__(self, env, discount=.996, episodes=6000, epsilon=.9, primary_reward='env', policy_idx=0):
        """Trains using the simulator and e-greedy exploration to determine a greedy policy.

        :param env: Simulator.
        :param discount:
        :param episodes:

        """
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

    def train(self, env):
        for episode in range(self.episodes):
            time_step = env.reset()
            while not time_step.last():
                last_board = str(time_step.observation['board'])
                action = self.behavior_action(last_board)
                time_step = env.step(action)
                self.update_greedy(last_board, action, time_step)
        env.reset()

        self.save()  # Save the learned Q-function

    def save(self):
        self.write_name = 'results/policy_' + self.name + '_' + self.game_name + '_' + self.policy_idx + '.pkl'
        if not os.path.exists('./results/'):
            os.makedirs('./results/')
        with open(self.write_name, 'wb') as f:
            pickle.dump(dict(self.Q), f)

    def act(self, obs):
        return self.Q[str(obs['board'])].argmax()

    def behavior_action(self, board):
        """Returns the e-greedy action for the state board string."""
        greedy = self.Q[board].argmax()
        if np.random.random() < self.epsilon or len(self.actions) == 1: # Choose greedy action
            return greedy
        else:  # choose anything else
            return np.random.choice(self.actions, p=self.probs[greedy])

    def update_greedy(self, last_board, action, time_step):
        """Perform TD update on observed reward."""
        learning_rate = 1
        new_board = str(time_step.observation['board'])

        def calculate_update():
            """Do the update for the main function (or the attainable function at the given index)."""
            if self.primary_reward is 'env':
                reward = time_step.reward
            else:
                reward = self.primary_reward[last_board]
            new_Q, old_Q = self.Q[new_board].max(), self.Q[last_board][action]
            return learning_rate * (reward + self.discount * new_Q - old_Q)
        self.Q[last_board][action] += calculate_update()