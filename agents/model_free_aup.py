from ai_safety_gridworlds.environments.shared import safety_game
from collections import defaultdict
import experiments.environment_helper as environment_helper
import numpy as np
from ExactSolver import ExactSolver
from QLearning import QLearner


class ModelFreeAUPAgent(QLearner):
    name = "Model_free_AUP"

    def __init__(self, env, lambd=.01, state_attainable=False, num_rewards=15,
                 discount=.996, episodes=6000, primary_reward='env', policy_idx=0):
        """Trains using the simulator and e-greedy exploration to determine a greedy policy.

        :param env: Simulator.
        :param lambd: Impact tuning parameter.
        :param state_attainable: True - generate state indicator rewards; false - random rewards.
        :param num_rewards: Size of the attainable set, |\mathcal{R}|.
        """
        QLearner.__init__(self, env=env, discount=discount, episodes=episodes, primary_reward=primary_reward,
                          policy_idx=policy_idx)
        self.lambd = lambd  # Penalty coefficient
        self.state_attainable = state_attainable
        self.primary_reward = primary_reward

        if state_attainable:
            self.name = 'Relative_reachability'
            self.auxiliary_rewards = environment_helper.derive_possible_rewards(env)
        else:
            self.auxiliary_rewards = [defaultdict(np.random.random) for _ in range(num_rewards)]

        # Initialize tabular Q-functions for auxiliary reward functions, in addition to inherited self.Q
        self.auxiliary_agents = [ExactSolver(env, primary_reward=self.auxiliary_rewards[i])
                                 for i in range(num_rewards)]
        for agent in self.auxiliary_agents: agent.solve(env)
        self.auxiliary_Q = [agent.Q for agent in self.auxiliary_agents]
        if state_attainable: self.auxiliary_Q = map(lambda q: np.clip(q, 0, 1), self.auxiliary_Q)

    def get_penalty(self, board, action):
        if len(self.auxiliary_rewards) == 0: return 0
        diffs = np.array([agent.Q[agent.str_map(str(board))][action] - agent.Q[agent.str_map(str(board))][safety_game.Actions.NOTHING]
                          for agent in self.auxiliary_agents])

        # Difference between taking action and doing nothing
        return self.lambd * sum(abs(diffs))

    def update_greedy(self, last_board, action, time_step):
        """Perform TD update on observed reward."""
        learning_rate = 1

        if self.primary_reward is 'env':
           reward = time_step.reward
        else:
            reward = self.primary_reward[last_board]
        reward = (1-self.discount) * reward - self.get_penalty(last_board, action)

        new_board = str(time_step.observation['board'])
        self.Q[last_board][action] += learning_rate * (reward + self.discount * self.Q[new_board].max()
                                                       - self.Q[last_board][action])
