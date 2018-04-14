import numpy as np
import copy
from ai_safety_gridworlds.environments.side_effects_sokoban import SideEffectsSokobanEnvironment as sokoban_game

def simulate(env, actions=[None]):
    """Uses a perfect world model by copying the environment
    In the future we might want to have an imperfect model by making the
    transition model or pixels displayed somewhat incorrect.

    Args:
        env: The pycolab environment to use as a basis to simulate
        actions: A list of actions to perform in the simulated environment
    """
    time_steps = []
    sim_env = copy.deepcopy(env)

    for action in actions:
        time_steps.append(env.step(action))

    return time_steps
