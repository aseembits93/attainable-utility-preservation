from __future__ import print_function

from agents.ExactSolver import ExactSolver
from ai_safety_gridworlds.environments import *
from agents.model_free_aup import ModelFreeAUPAgent
import datetime
from agents.QLearning import QLearner
import numpy as np
from collections import defaultdict
from environment_helper import run_episode
import pickle
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipdb as pdb

def true_goal_performance(true_agent, prefix_agent, env, correction_time):
    """
    Evaluate the prefix_agent policy on the true objective in the env environment
    Args:
        true_agent: QLearner with a "true" goal and corresponding Q-function
        prefix_agent: QLearner with some "prefix" policy
        env: ai safety gridworld environment
        correction_time: rollout length for prefix policy

    Returns:
    discounted true return under the prefix policy
    """
    time_step = env.reset()
    discount = true_agent.discount
    side_effect_score = 0
    for i in range(correction_time):
        state_idx = true_agent.str_map(str(time_step.observation['board']))
        reward = true_agent.primary_reward[state_idx] if isinstance(true_agent.primary_reward, defaultdict) else true_agent.primary_reward(time_step.observation['board'])
        side_effect_score += (discount ** i) * reward
        action = prefix_agent.act(time_step.observation)
        time_step = env.step(action)
        #if isinstance(prefix_agent, ModelFreeAUPAgent): print(time_step)
    state_idx = true_agent.str_map(str(time_step.observation['board']))
    side_effect_score += (discount ** correction_time) * np.max(true_agent.Q[state_idx])
    return side_effect_score


def run_game(game, kwargs):
    start_time = datetime.datetime.now()
    residuals = run_agents(game, kwargs)
    print("Training finished; {} elapsed.\n".format(datetime.datetime.now() - start_time))
    # TODO violin plots here

def plot_residuals(residuals, env):
    snsdict = defaultdict(list)
    #for key in residuals.keys():
    for value in residuals['random']:
        snsdict['rewardtype'].append('random')
        snsdict['value'].append(value)
    snsdict = dict(snsdict)
    ax = sns.violinplot(x="rewardtype", y="value", data=pd.DataFrame.from_dict(snsdict))
    fig = ax.get_figure()
    fig.savefig('residuals_'+env.name+'.pdf')   
    plt.close()      

def run_agents(env_class, env_kwargs):
    """
    Evaluate agents on different ground-truth reward functions

    :param env_class: class object.
    :param env_kwargs: environmental intialization parameters.
    """
    # Instantiate environment and agents
    env = env_class(**env_kwargs)

    # Agents to evaluate
    aup_agent = ModelFreeAUPAgent(env, lambd=.01)
    # for file in sorted(glob.glob('q_functions/*_AUP_'+env.name+'*.pkl')):
    #     with open(file,'rb') as f:
    #         aup_agent.Q = pickle.load(f)
    aup_agent.train(env)
    standard_agent = ExactSolver(env)

    # Learn Q-functions for ground-truth reward functions
    random_reward_agents = [ExactSolver(env, primary_reward=defaultdict(np.random.uniform))
                            for i in range(1000)]
    intended_agent = ExactSolver(env, primary_reward=env._get_true_reward)  # Agent optimizing R := 2 * (goal reached?) - 1 * (side effect had?)
    anti_intended_agent = ExactSolver(env, primary_reward=lambda state: -1 * env._get_true_reward(state))  # Agent optimizing -1 * reward of intended_agent
    true_agents = random_reward_agents + [intended_agent, anti_intended_agent]

    for agent in [standard_agent] + true_agents:
        if isinstance(agent, QLearner):
            agent.train(env)
        else:
            agent.solve(env)

    correction_time = 10
    residuals = defaultdict(list)
    for true_agent in true_agents:
        AUP_perf = true_goal_performance(true_agent=true_agent, prefix_agent=aup_agent, env=env,
                                         correction_time=correction_time)
        standard_perf = true_goal_performance(true_agent=true_agent, prefix_agent=standard_agent, env=env,
                                              correction_time=correction_time)
        def get_label(agent):
            if agent == intended_agent: return 'True reward'
            elif agent == anti_intended_agent: return 'Inverted true reward'
            else: return 'random'

        residuals[get_label(true_agent)].append(AUP_perf - standard_perf)  # higher is better for AUP
    print(residuals)
    plot_residuals(residuals,env)
    print("save residuals")
    with open('residuals_'+env.name+'.pkl','wb') as f:
        pickle.dump(residuals, f)
    return residuals


games = [(box.BoxEnvironment, {'level': 0}),
         (dog.DogEnvironment, {'level': 0})]

# Get violin plot for each game
for (game, kwargs) in games:
    run_game(game, kwargs)
