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
    state_idx = true_agent.str_map(str(time_step.observation['board']))
    side_effect_score += (discount ** correction_time) * np.max(true_agent.Q[state_idx])
    return side_effect_score


def run_game(game, kwargs):
    start_time = datetime.datetime.now()
    residuals = run_agents(game, kwargs)
    print("Training finished; {} elapsed.\n".format(datetime.datetime.now() - start_time))
    # TODO violin plots here


def run_agents(env_class, env_kwargs):
    """
    Evaluate agents on different ground-truth reward functions

    :param env_class: class object.
    :param env_kwargs: environmental intialization parameters.
    """
    # Instantiate environment and agents
    env = env_class(**env_kwargs)

    # Agents to evaluate
    aup_agent = ModelFreeAUPAgent(env)
    standard_agent = QLearner(env)
    evaluation_agents = [aup_agent, standard_agent]

    # Learn Q-functions for ground-truth reward functions
    random_reward_agents = [ExactSolver(env, primary_reward=defaultdict(np.random.uniform), policy_idx=i+1, epsilon=.2)
                            for i in range(1)]
    intended_agent = ExactSolver(env, primary_reward=env._get_true_reward,
                                 epsilon=.2)  # Agent optimizing R := 2 * (goal reached?) - 1 * (side effect had?)
    anti_intended_agent = ExactSolver(env, primary_reward=lambda state: -1 * env._get_true_reward(state),
                                      epsilon=.2)  # Agent optimizing -1 * reward of intended_agent
    true_agents = random_reward_agents + [intended_agent, anti_intended_agent]

    for agent in true_agents:
        if isinstance(agent, QLearner):
            agent.train(env)
        else:
            agent.solve(env)

    correction_time = 10
    residuals = []
    for true_agent in true_agents:
        AUP_perf = true_goal_performance(true_agent=true_agent, prefix_agent=aup_agent, env=env,
                                         correction_time=correction_time)
        standard_perf = true_goal_performance(true_agent=true_agent, prefix_agent=standard_agent, env=env,
                                              correction_time=correction_time)
        residuals.append(AUP_perf - standard_perf)  # higher is better for AUP
        print("AUP obtains " + str(residuals[-1]) + " greater performance.")
    return residuals


def run_agents_offline(env_class, env_kwargs):
    """
    Generate and run agent variants.

    :param env_class: class object.
    :param env_kwargs: environmental intialization parameters.
    :param render_ax: PyPlot axis on which rendering can take place.
    """
    # Instantiate environment and agents
    env = env_class(**env_kwargs)

    custom_reward_agents = dict()
    standard_agent = dict()
    aup_agent = dict()
    game_name = env_class.name
    for file in sorted(glob.glob('results/policy_Q*'+game_name+'*.pkl')):
        with open(file,'rb') as f:
            custom_reward_agents[file]=pickle.load(f)
    for file in sorted(glob.glob('results/*_AUP_'+game_name+'*.pkl')):
        with open(file,'rb') as f:
            aup_agent[file]=pickle.load(f)
    for file in sorted(glob.glob('results/*_Standard_'+game_name+'*.pkl')):
        with open(file,'rb') as f:
            standard_agent[file]=pickle.load(f)
    movies = list()
    time_t = 10
    for agent_name, agent in custom_reward_agents.items():
        ret, _, perf, frames = run_episode(agent, env, save_frames=True, offline=True)
        movies.append((agent_name,frames))
    for agent_name, agent in standard_agent.items():
        ret, _, perf, frames = run_episode(agent, env, save_frames=True, offline=True)
        movies.append((agent_name,frames))
    for agent_name, agent in aup_agent.items():
        ret, _, perf, frames = run_episode(agent, env, save_frames=True, offline=True)
        movies.append((agent_name,frames))
    return 0,movies

games = [#(box.BoxEnvironment, {'level': 0}),
        (dog.DogEnvironment, {'level': 0})]

# Get violin plot for each game
for (game, kwargs) in games:
    run_game(game, kwargs)
