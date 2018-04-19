import time
from IPython import display
import copy
import numpy as np
import tensorflow as tf



def get_frame(step, x, y):
  color_state = step.observation['RGB']
  return np.moveaxis(color_state, x, y)


def refresh_screen(step, x=0, y=-1):
  time.sleep(0.6)
  frame = get_frame(step, x, y)
  im.set_data(frame)
  display.clear_output(wait=True)
  display.display(plt.gcf())


def to_grayscale(state, sess) :
    if type(state) == TimeStep :
        state = state.observation['RGB']

    state = np.moveaxis(state, 0, -1)
    resize = tf.placeholder(shape=list(state.shape), dtype=tf.uint8)
    gray_frame = tf.squeeze( tf.image.rgb_to_grayscale(state) )
  
    return sess.run( gray_frame, { resize: state } )


"""
# Usage:
plt.axis("off")


GAME_ART = ['######', '# A###', '# X  #', '##   #', '### G#', '######']
env = sokoban_game(GAME_ART)
step = env.reset()
im = plt.imshow(get_frame(step, 0, -1))
step = env.step(Actions.LEFT)
refresh_screen(step)
"""