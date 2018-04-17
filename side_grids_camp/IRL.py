#Thanks to:
@misc{alger16,
  author       = {Matthew Alger},
  title        = {Inverse Reinforcement Learning},
  year         = 2016,
  doi          = {10.5281/zenodo.555999},
  url          = {https://doi.org/10.5281/zenodo.555999}
}

import numpy as np

"""
# trajectories =  array of demonstration trajectories, each trajectory is an array of state action pairs


# feature_matrix = array of feature vectors, each feature vector is associated with the state at that index


# transition_probabilities = probability of moving from one state to another given action, 1 for legal moves, 0 for illegal moves

# feature_expectations = sum feature vectors of every state visited in every demo trajectory, divide result by the number of trajectories, to get feature expectation (over all states) for an average demo trajectory?

"""


N_ACTIONS = 4
N_STATES = # TODO
LEARNING_RATE = 0.01
EPOCHS = 200
DISCOUNT = 0.01

irl = maxEntIRL(feature_matrix, trans_probs, trajectories)


# TODO methods to map from state to state index and back
# How are states represented here - location of agent and box? Hard to apply in complex envs. Feature array? Greyscale images?

def stateToInt(state):
    # TODO
    # Return int
    

def getExampleTrajectories():
    
    trajectories = []
    # TODO
    # (each state in traj is state action pair (state, action)
    # Could generate (take num, len, policy)or do by hand here   
    return trajectories
    
    

def getFeatureMatrix(states, feature_vectors):
    
    feature_matrix = []
    # TODO 
    # Each state has associated feature vector  
    # for n in range (number of possible states)
        # append feature vector
    return feature_matrix


def getTransitionProbabilities(gridworld_environment):
    
    trans_probs = []
    # for each state
        # for each possible action 
            # get state
    
    return trans_probs


def getFeatureExpectations(feature_matrix, trajectories):
    
    # feature_expectations = sum feature vectors of every state visited in every demo trajectory, divide result by the number of trajectories, to get feature expectation (over all states) for an average demo trajectory?
    
    feature_expectations = np.zeros(feature_matrix.shape[1])

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            feature_expectations += feature_matrix[state]

    feature_expectations /= trajectories.shape[0]

    return feature_expectations


def getSVF(trajectories):
    # State visitiation frequencies
    
    svf = np.zeros(n_states)

        for trajectory in trajectories:
            for state, _, _ in trajectory:
                svf[state] += 1

        svf /= trajectories.shape[0]

    return svf


def getExpectedSVF(rewards, transition_probability, trajectories):
    # state visitation frequency
    # policy = findPolicy(transition_probability, rewards)

    # 
    
    return expected_svf
    
    

def findPolicy(transition_probability, rewards):
    
    # TODO
    
    # Find optimal policy
    
    # pull this out to irl method?
    value_function = getOptimalValueFunction(transition_probabilities, reward, conv_threshold)
    
    
    
def getOptimalValueFunction(transition_probabilities, reward, conv_threshold):
    
    val_func = np.zeros(n_states)





def maxEntIRL(feature_matrix, trans_probs, trajectories):
    
    # TODO
    
    # Initialise weights = numpy.random.uniform(size=(num_of_states,))
    
    # Get feature matrix using all states and feature vectors
    
    # Get expert demo trajectories
    
    # Get feature expectations
    
    
    # for i in range EPOCHS
        # rewards = feature_matrix.dot(weights)
        
        # get expected svfs (state visitation frequencies)
        
        # rewards += learning_rate * (feature_expectations - (feature_matrix * expectedsvf))
        
    # return feature_matrix * rewards
    
        
    
    
    
    return rewards


    