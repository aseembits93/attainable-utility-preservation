import time
from IPython import display
import copy
import numpy as np


def get_frame(step, x, y):
  color_state = step.observation['RGB']
  return np.moveaxis(color_state, x, y)


def refresh_screen(step, x=0, y=-1):
  time.sleep(0.6)
  frame = get_frame(step, x, y)
  im.set_data(frame)
  display.clear_output(wait=True)
  display.display(plt.gcf())


def to_grayscale(obs, showMe=False) :
  if type(obs) == TimeStep :
    obs = obs.observation['RGB']
  
  obs = np.moveaxis(obs, 0, -1)
  wZ, wX, wY = env.observation_spec()['RGB'].shape

  resize = tf.placeholder(shape=[wX, wY, wZ], dtype=tf.uint8)
  gray_frame = tf.squeeze( tf.image.rgb_to_grayscale(obs) )

  with tf.Session() as sess:
    sess.run( gray_frame, { resize: obs } )
    grayed = gray_frame.eval()

  if showMe :
    plt.figure()
    plt.imshow(grayed/255.0, cmap='gray')
    plt.axis('off')
    plt.show()
  
  return grayed


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