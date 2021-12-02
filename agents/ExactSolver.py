from Agent import Agent
from experiments.environment_helper import derive_MDP
import mdptoolbox
import numpy as np


class ExactSolver(Agent):
    def solve(self, env):
        P, R, self.str_map = derive_MDP(env, reward_fn=self.primary_reward)
        pi = mdptoolbox.mdp.PolicyIteration(P, R, self.discount)
        pi.run()

        def get_action_value(V, transitions, reward, state, action):
            return reward[state] + self.discount * np.dot(transitions[action][state], V)

        self.Q = np.array([[get_action_value(pi.V, P, R, s, a) for a in range(env.action_spec().maximum + 1)]
                          for s in range(len(pi.V))])

    def act(self, obs):
        return self.Q[self.str_map(str(obs['board']))].argmax()