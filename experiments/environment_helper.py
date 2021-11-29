from __future__ import print_function
import itertools
import matplotlib.pyplot as plt
import numpy as np
from agents.aup import AUPAgent
from ai_safety_gridworlds.environments.shared import safety_game
from collections import defaultdict


def derive_possible_rewards(env):
    """
    Derive possible reward functions for the given environment.

    :param env:
    """

    def state_lambda(original_board_str):
        return lambda obs: int(obs == original_board_str) * env.GOAL_REWARD

    def explore(env, so_far=[]):  # visit all possible states
        board_str = str(env._last_observations['board'])
        if board_str not in states:
            states.add(board_str)
            fn = state_lambda(board_str)
            fn.state = board_str
            functions.append(fn)
            if not env._game_over:
                for action in range(env.action_spec().maximum + 1):
                    env.step(action)
                    explore(env, so_far + [action])
                    AUPAgent.restart(env, so_far)

    env.reset()
    states, functions = set(), []
    explore(env)
    env.reset()
    return functions


def derive_MDP(env, reward_fn=None):
    """
    Use depth-first-search to back out transition matrix and reward function from simulator. Assumes determinism.

    :param env:
    """
    mdp = defaultdict(lambda: [set(), 0.0])  # ({reachable successors}, state reward)

    def explore(env, so_far=[]):  # visit all possible states
        board_str = str(env._last_observations['board'])
        if board_str not in mdp.keys():
            for action in range(env.action_spec().maximum + 1):
                time_step = env.step(action)
                new_str = str(env._last_observations['board'])
                mdp[board_str][0].add((action, new_str))  # Note that (board_str -> action -> new_str) transition is possible
                explore(env, so_far + [action])

                # Note: Will not record reward for initial state unless it can be reached on another timestep
                if isinstance(reward_fn, defaultdict):
                    mdp[new_str][1] = reward_fn[str(env._last_observations['board'])]
                elif reward_fn in (None, 'env'):
                    mdp[new_str][1] = time_step.reward
                else:
                    mdp[new_str][1] = reward_fn(env._last_observations['board'])
                AUPAgent.restart(env, so_far)

    env.reset()
    explore(env)
    env.reset()  # We now know state rewards and transitions

    # Go from dictionary with successors to a sparse n x n matrix
    transitions = np.zeros((env.action_spec().maximum + 1, len(mdp.keys()), len(mdp.keys()))) # |A| x |S| x |S|
    reward_vec = np.zeros(len(mdp.keys()))
    str_map = lambda state_string: mdp.keys().index(state_string)
    for key, val in mdp.items():
        reward_vec[str_map(key)] = val[1]
        for act, succ in val[0]:
            transitions[act][str_map(key)][str_map(succ)] = 1

    return transitions, reward_vec, str_map


def run_episode(agent, env, save_frames=False, render_ax=None, max_len=9, offline=False):
    """
    Run the episode, recording and saving the frames if desired.

    :param agent:
    :param env:
    :param save_frames: Whether to save frames from the final performance.
    :param render_ax: matplotlib axis on which to display images.
    :param max_len: How long the agent plans/acts over.
    """

    def handle_frame(time_step):
        if save_frames:
            frames.append(np.moveaxis(time_step.observation['RGB'], 0, -1))
        if render_ax:
            render_ax.imshow(np.moveaxis(time_step.observation['RGB'], 0, -1), animated=True)
            plt.pause(0.001)

    frames, actions = [], []

    time_step = env.reset()
    handle_frame(time_step)
    if hasattr(agent, 'get_actions'):
        actions, _ = agent.get_actions(env, steps_left=max_len)
        if env.name == 'survival':
            actions.append(safety_game.Actions.NOTHING)  # disappearing frame
        max_len = len(actions)
    for i in itertools.count():
        if time_step.last() or i >= max_len:
            break
        if not hasattr(agent, 'get_actions'):
            if offline:
                actions.append(agent[str(time_step.observation['board'])].argmax())
            else:
                actions.append(agent.act(time_step.observation))
        time_step = env.step(actions[i])
        handle_frame(time_step)
    return float(env.episode_return), actions, float(env._get_hidden_reward()), frames
