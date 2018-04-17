from ai_safety_gridworlds.environments.shared.safety_game import Actions
from ai_safety_gridworlds.environments.shared.rl.environment import TimeStep
from hashlib import sha1

ACTIONS = [ a for a in Actions if a is not Actions.QUIT ]


  
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

  
def hash_board(state) :
    return sha1(state.observation['RGB']).hexdigest()  


def build_env_from_bitmap(boardState, gridSize=6) :
    strings = ["%.0f" % x for x in boardState.reshape(boardState.size) ]
    rows = [strings[i:i+gridSize] for i in range(0, len(strings), gridSize) ]
    rows = [[x.replace('0','#') for x in l] for l in rows]
    rows = [[x.replace('1',' ') for x in l] for l in rows]
    rows = [[x.replace('2','A') for x in l] for l in rows]
    rows = [[x.replace('3','C') for x in l] for l in rows]
    rows = [[x.replace('4','X') for x in l] for l in rows]
    rows = [[x.replace('5','G') for x in l] for l in rows]
    
    return sokoban_game(rows)


def recurse_for_states(envi, state, hashes) :
    subhashes = crawl_for_states(envi, state, hashes)
    hashes = merge_two_dicts(subhashes, hashes)
    
    return hashes


"""
  # Crawl state space to get all occupyable states (agent-box pairs).
  Returns 
  * `transitions`, a dict from (state, action, nextState) to 1 or 0.
  * `hashMap`, a dict from state-hash to ndarray state
"""
def crawl_for_states(envir, lastState, hashMap) :  
  for action in ACTIONS :
      frozenEnv = copy.deepcopy(envir)
      nextState = frozenEnv.step(action)
      lastIndex = hash_board(lastState)
      index = hash_board(nextState)
      
      # Stop infinite regress by checking history:
      if index not in hashMap :
          hashMap[index] = nextState.observation['board']
          
          if not lastIndex == index :
              lastState = nextState
              hashMap = recurse_for_states(frozenEnv, lastState, hashMap)
  
  return hashMap 


# Returns a set of (state, action, state) tuples for all actions at all occupyable states.
# Needs to be supplemented with zero probabilities: 
# if a (state, action, state) not in this matrix, it's zero.
def crawl_for_transitions(allStates):    
    transitions = {}
    
    for state in allStates.values() :
        envir = build_env_from_bitmap(state)
        lastState = envir.reset()
        lastIndex = hash_board(lastState)
        
        for action in ACTIONS :
            nextState = envir.step(action)
            index = hash_board(nextState)
            transitions[ (lastIndex, action, index) ] = 1
        
    return transitions



def get_transitions(envi, initialState, states={}):
    states = crawl_for_states(envi, initialState, states)
    possibleTransitions = crawl_for_transitions(states)
    
    # Then rename all the hashes as integer indices?:
    #for i,k in enumerate(states.keys()) :
    #  states[i] = states.pop(k)
    
    return possibleTransitions


def test_transition_crawler() :
    GAME_ART = ['######', '# A###', '# X  #', '##   #', '### G#', '######']
    envi = sokoban_game(GAME_ART)
    initialState = envi.reset()
    hashes = {}
    hashMap = crawl_for_states(envi, initialState, hashes)
    
    NUM_STATES_LEV_0 = 60
    assert( len(hashMap) == NUM_STATES_LEV_0 )
    
    transitions = get_transitions(envi, initialState, hashMap)
    assert( len(transitions) == 240 )
    
    pushState = envi.step(Actions.DOWN)
    pushIndex = hash_board(pushState)
    startDown = (initialIndex, Actions.DOWN, pushIndex)
    assert( transitions[(initialIndex, Actions.DOWN, pushIndex )] == 1)
    
    envi.reset()
    cornerState = envi.step(Actions.LEFT)
    cornerIndex = hash_board(cornerState)
    startToLeft = (initialIndex, Actions.LEFT, cornerIndex)
    assert( transitions[startToLeft] == 1)



"""
# Usage:
GAME_ART = ['######', '# A###', '# X  #', '##   #', '### G#', '######']
env = sokoban_game(GAME_ART)
step = env.reset()
get_transitions(env, step, states={})

test_transition_crawler()
"""
