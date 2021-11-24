from __future__ import print_function

import seaborn as sns

#sns.set_theme(style="whitegrid")

from ai_safety_gridworlds.environments import *
from agents.model_free_aup import ModelFreeAUPAgent
import datetime
from agents.QLearning import QLearner
import numpy as np
from collections import defaultdict
import imageio
import cv2
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
        state = time_step.observation['board']
        side_effect_score += (discount ** i) * true_agent.primary_reward[
            str(state)]
        time_step = env.step(prefix_agent.act(time_step.observation))
    side_effect_score += (discount ** correction_time) * np.max(true_agent.Q[str(state)])
    return side_effect_score

def plot_images_to_ani(framesets):
    # """
    # Animates all agent executions and returns the animation object.

    # :param framesets: [("agent_name", frames),...]
    # """
    for file_name, frames in framesets:
        im = list()
        for frame in frames:
            im.append(cv2.resize(frame,(100,100),interpolation=cv2.INTER_AREA))
        imageio.mimsave(file_name.split('.')[0]+'.gif', im)

def run_game(game, kwargs):
    start_time = datetime.datetime.now()
    #residuals, movies = run_agents(game, kwargs)
    residuals, movies = run_agents_offline(game, kwargs)
    print("Training finished; {} elapsed.\n".format(datetime.datetime.now() - start_time))
    # TODO violin plots here
    plot_images_to_ani(movies)


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
    random_reward_agents = [QLearner(env, primary_reward=defaultdict(np.random.uniform), policy_idx=i) for i in
                            range(10)]  # Make 10 ground-truth reward agents

    # TODO code these objectives
    intended_agent = QLearner(env)  # Agent optimizing R := 2 * (goal reached?) - 1 * (side effect had?)
    anti_intended_agent = QLearner(env)  # Agent optimizing -1 * reward of intended_agent

    true_agents = random_reward_agents + [intended_agent] + [anti_intended_agent]

    movies=list()
    for agent in evaluation_agents + random_reward_agents:
        agent.train(env)
        movies.append((agent.write_name,run_episode(agent, env,save_frames=True)[3]))
    correction_time = 10
    residuals = []
    for random_agent in random_reward_agents:
        AUP_perf = true_goal_performance(true_agent=random_agent, prefix_agent=aup_agent, env=env,
                                         correction_time=correction_time)
        standard_perf = true_goal_performance(true_agent=random_agent, prefix_agent=standard_agent, env=env,
                                              correction_time=correction_time)
        residuals.append(AUP_perf - standard_perf)  # higher is better for AUP
        print("AUP obtains " + str(residuals[-1]) + " greater performance.")
    return residuals, movies

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

games = [(box.BoxEnvironment, {'level': 0}), (dog.DogEnvironment, {'level': 0})]

# Get violin plot for each game
for (game, kwargs) in games:
    run_game(game, kwargs)