from Agent import Agent
from experiments.environment_helper import derive_MDP
import mdptoolbox


class ExactSolver(Agent):
    def solve(self, env):
        P, R, self.str_map = derive_MDP(env, reward_fn=self.primary_reward)
        ql = mdptoolbox.mdp.QLearning(P, R, self.discount, n_iter=1e8)
        ql.run()
        self.Q = ql.Q

    def act(self, obs):
        return self.Q[self.str_map(str(obs['board']))].argmax()