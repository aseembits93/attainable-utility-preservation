from ai_safety_gridworlds.environments.shared import safety_game
from collections import defaultdict
import experiments.environment_helper as environment_helper
import numpy as np
import pickle
from QLearning import QLearner


class ModelFreeAUPAgent(QLearner):
    name = "Model_free_AUP"
    pen_epsilon, AUP_epsilon = .2, .9  # chance of choosing greedy action in training
    default = {'lambd': 1. / 1.501, 'discount': .996, 'rpenalties': 15, 'episodes': 600}

    def __init__(self, env, lambd=default['lambd'], state_attainable=False, num_rewards=default['rpenalties'],
                 discount=default['discount'], episodes=default['episodes'],
                 primary_reward='env', policy_idx=0):
        """Trains using the simulator and e-greedy exploration to determine a greedy policy.

        :param env: Simulator.
        :param lambd: Impact tuning parameter.
        :param state_attainable: True - generate state indicator rewards; false - random rewards.
        :param num_rewards: Size of the attainable set, |\mathcal{R}|.
        """
        QLearner.__init__(self, env=env, discount=discount, episodes=episodes, primary_reward=primary_reward,
                          policy_idx=policy_idx)

        self.lambd = lambd  # Penalty coefficient
        self.epsilon = self.pen_epsilon
        self.state_attainable = state_attainable
        self.primary_reward = primary_reward

        if state_attainable:
            self.name = 'Relative_reachability'
            self.auxiliary_rewards = environment_helper.derive_possible_rewards(env)
        else:
            self.auxiliary_rewards = [defaultdict(np.random.random) for _ in range(num_rewards)]

        # Initialize tabular Q-functions for auxiliary reward functions, in addition to inherited self.Q
        self.auxiliary_Q = defaultdict(lambda: np.zeros((len(self.auxiliary_rewards), len(self.actions))))

    def train(self, env):
        self.performance = np.zeros(self.episodes / 10)

        for episode in range(self.episodes):
            if episode > 2.0 / 3 * self.episodes:  # begin greedy exploration
                self.epsilon = self.AUP_epsilon

            time_step = env.reset()
            while not time_step.last():
                last_board = str(time_step.observation['board'])
                action = self.behavior_action(last_board)
                time_step = env.step(action)
                self.update_greedy(last_board, action, time_step)

            if episode % 10 == 0:
                _, actions, self.performance[episode / 10], _ = environment_helper.run_episode(self, env)
        env.reset()

        self.save()

    def get_penalty(self, board, action):
        if len(self.auxiliary_rewards) == 0: return 0
        action_attainable = self.auxiliary_Q[board][:, action]
        null_attainable = self.auxiliary_Q[board][:, safety_game.Actions.NOTHING]

        # Difference between taking action and doing nothing
        return self.lambd * sum(abs(action_attainable - null_attainable))

    def update_greedy(self, last_board, action, time_step):
        """Perform TD update on observed reward."""
        learning_rate = 1
        new_board = str(time_step.observation['board'])

        def calculate_update(auxiliary_idx=None):
            """Do the update for the main function (or the attainable function at the given index)."""
            if auxiliary_idx is not None:
                reward = self.auxiliary_rewards[auxiliary_idx](new_board) if self.state_attainable \
                    else self.auxiliary_rewards[auxiliary_idx][new_board]
                new_Q, old_Q = self.auxiliary_Q[new_board][auxiliary_idx].max(), \
                               self.auxiliary_Q[last_board][auxiliary_idx, action]
            else:
                if self.primary_reward is 'env':
                    reward = time_step.reward
                else:
                    reward = self.primary_reward[last_board]
                reward = (1 - self.discount) * reward - self.get_penalty(last_board, action)

                new_Q, old_Q = self.Q[new_board].max(), self.Q[last_board][action]
            return learning_rate * (reward + self.discount * new_Q - old_Q)

        # Learn the attainable reward functions
        for attainable_idx in range(len(self.auxiliary_rewards)):
            self.auxiliary_Q[last_board][attainable_idx, action] += calculate_update(attainable_idx)

        # Clip Q-values to be state reachability indicators -- 0 if unreachable, 1 otherwise
        if self.state_attainable:
            self.auxiliary_Q[last_board][:, action] = np.clip(self.auxiliary_Q[last_board][:, action], 0, 1)
        self.Q[last_board][action] += calculate_update()
