#!/bin/env python3

import gym
import numpy

class BlackjackDPAgent:
    def __init__ (self, gamma = 1.0, seed = None):
        self.__env = gym.make('Blackjack-v0')
        self.__env.seed(seed)
        self.state_value_table = numpy.zeros([10, 10, 2])
    
    def evaluate(self, state: tuple)
        return self.state_value_table.__getitem__(
            tuple(state[0]-11, state[1]-1, int(state[2]))
        )

    def value_iteration (self):
        actions = (0, 1) # Possible actions are hit or stand
        current_state = self.__env._get_obs()


    def evaluate_probs (self):
        pass