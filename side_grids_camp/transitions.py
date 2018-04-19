from ai_safety_gridworlds.environments.shared.safety_game import Actions
from ai_safety_gridworlds.environments.shared.rl.environment import TimeStep

from hashlib import sha1
import numpy
import time
from IPython import display
import copy

ACTIONS = [ a for a in Actions if a is not Actions.QUIT ]

  
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def hash_board(state, stateType='RGB') :
    if type(state) == TimeStep :
        state = state.observation[stateType]
    return sha1(state).hexdigest()


def to_grayscale(state, sess) :
    if type(state) == TimeStep :
        state = state.observation['RGB']

    state = np.moveaxis(state, 0, -1)
    resize = tf.placeholder(shape=list(state.shape), dtype=tf.uint8)
    gray_frame = tf.squeeze( tf.image.rgb_to_grayscale(state) )
    grayed = sess.run( gray_frame, { resize: state } )
  
    return grayed



  
def recurse_for_substates(envi, state, hashes) :
    subhashes = crawl_for_states(envi, state, hashes)
    hashes = merge_two_dicts(subhashes, hashes)
    
    return hashes


def build_env_from_boardmap(boardState, gridSize=6) :
    boardToArtMap = {
        '0': '#',
        '1': ' ',
        '2': 'A',
        '3': 'C',
        '4': 'X',
        '5': 'G'
    }
    return build_env_from_bitmap(boardState, boardToArtMap, gridSize)

  
def build_env_from_grayscale(grayState, gridSize=6) :
    grayscaleToArtMap = {
        '152': '#',
        '219': ' ',
        '134': 'A',
        '3'  : 'C',
        '78' : 'X',
        '129': 'G'
    }
    return build_env_from_bitmap(grayState, grayscaleToArtMap, gridSize)


# `valueMap` is a dict of object codes to game art codes
def build_env_from_bitmap(state, valueMap, gridSize=6) :
    rows = get_game_rows_from_state(state, gridSize)
    
    for k,v in valueMap.items() :
      rows = [ [x.replace(k,v) for x in l] for l in rows ]
      
    return sokoban_game(level=0, game_art=[rows])
  
  
# Returns list of list of strings from the np state passed in
def get_game_rows_from_state(state, gridSize):
    strings = ["%.0f" % x for x in state.reshape(state.size) ]
    chunkedIntoRows = range(0, len(strings), gridSize)
    
    return [ strings[i:i+gridSize] for i in chunkedIntoRows ]
  

"""
  # Crawl state space to get all occupyable states (agent-box pairs).
  Returns 
  * `transitions`, a dict from (state, action, nextState) to 1 or 0.
  * `hashMap`, a dict from state-hash to ndarray state
"""
def crawl_for_states(envir, lastState, hashMap, stateType='board') :  
  for action in ACTIONS :
      frozenEnv = copy.deepcopy(envir)
      nextState = frozenEnv.step(action)
      lastIndex = hash_board(lastState, stateType)
      index = hash_board(nextState, stateType)
      
      # Stop infinite regress by checking history:
      if index not in hashMap :
          hashMap[index] = nextState.observation[stateType]
          
          # If `action` changed the state, recurse:
          if not lastIndex == index :
              lastState = nextState
              hashMap = recurse_for_substates(frozenEnv, lastState, hashMap)
  
  return hashMap 



# Returns a set of (state, action, state) tuples for all actions at all occupyable states.
# Needs to be supplemented with zero probabilities.
def crawl_for_transitions(allStates):    
    transitions = {}
    
    for state in allStates.values() :
        envir = build_env_from_boardmap(state)
        lastState = envir.reset()
        lastIndex = hash_board(lastState)
        
        for action in ACTIONS :
            nextState = envir.step(action)
            index = hash_board(nextState)
            transitions[ (lastIndex, action, index) ] = 1
        
    return transitions


def get_transitions(envi, initialState, states={}):
    states = crawl_for_states(envi, initialState, states)
    
    return crawl_for_transitions(states)


def test_transition_crawler() :
    GAME_ART = [['######', '# A###', '# X  #', '##   #', '### G#', '######']]
    envi = sokoban_game(level=0, game_art=GAME_ART)
    initialState = envi.reset()
    #im = plt.imshow(get_frame(initialState, 0, -1), animated=True)
    hashes = {}
    hashMap = crawl_for_states(envi, initialState, hashes)
    
    NUM_STATES_LEV_0 = 60
    assert( len(hashMap) == NUM_STATES_LEV_0 )
    
    transitions = get_transitions(envi, initialState, hashMap)
    assert( len(transitions) == 240 )


#test_transition_crawler()