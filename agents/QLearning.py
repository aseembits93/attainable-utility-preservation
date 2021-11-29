from Agent import Agent


class QLearner(Agent):
    name = "Q-learner"

    def __init__(self, env, discount=.996, episodes=6, epsilon=.9, primary_reward='env', policy_idx=0):
        """Trains using the simulator and e-greedy exploration to determine a greedy policy.

        :param env: Simulator.
        :param discount:
        :param episodes:

        """
        Agent.__init__(self, env, discount=discount, episodes=episodes, epsilon=epsilon, primary_reward=primary_reward,
                       policy_idx=policy_idx)

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