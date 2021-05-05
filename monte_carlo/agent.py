#!/bin/env python3

import gym
import numpy

class MonteCarloBlackjackAgent:
    def __init__ (self, seed = None, initial_draw_prob=0.5):
        self.__env = gym.make('Blackjack-v0')
        self.__env.seed(seed)
        self.__state_value_table = numpy.zeros([10, 10, 2])
        self.__policy_table = numpy.full([10, 10, 2], initial_draw_prob)
        self.__rewards = {}
        
        self.__state_action_value_table = numpy.zeros([10, 10, 2, 2])
    
    @property
    def state_value_table(self):
        return self.__state_value_table
    
    @property
    def policy_table(self):
        return self.__policy_table
    
    @property
    def rewards(self):
        return self.__rewards
    
    @property
    def q(self):
        return self.__policy_table
    
    def generate_random_episode(self):
        '''
        Generates random episodes based on the current policy
        '''
        episode = []
        done = False
        state = self.__env.reset()
        while (not done):
            state_prime = state
            # prob = self.evaluate(state)
            probs = [0.8, 0.2] if state[0] < 20 else [0.2, 0.8]
            action = numpy.random.choice([1, 0], p=probs)
            state, reward, done, _ = self.__env.step(action)
            if (state[0] > 11):
                episode.append((state_prime, action, reward))
        return episode, state
        
    
    def evaluate(self, state: tuple):
        '''
        Evaluates a state to get the probability of the player choosing to hit
        if the player holds cards that make up to a sum of 11 or less, the
        choice is always to hit
        
        state (tuple)
        returns (float)
        '''
        if state[0] < 12: return 1
        state_index = (state[0]-12, state[1]-1, int(state[2]))
        return self.policy_table[state_index]
    
    def update_value_table(self, state: tuple, value):
        state_index = (state[0]-12, state[1]-1, int(state[2]))
        self.state_value_table[state_index] = value
    
    def update_policy_table(self, state: tuple, value):
        state_index = (state[0]-12, state[1]-1, int(state[2]))
        self.policy_table[state_index] = value
    
    def update_q(self, state: tuple, action: int, value):
        state_index = (state[0]-12, state[1]-1, int(state[2]), action)
        self.__state_action_value_table[state_index] = value
        
    @staticmethod
    def average_list(l: list):
        return sum(l)/len(l)

    def learn (self, episode, gamma = 1, epsilon = 0.01):
        G = 0
        for state, action, reward in episode[::-1]:
            G = gamma * G + reward
            if ((state, action) in list((x[0], x[1]) for x in episode)):
                if not (state, action) in self.rewards:
                    self.rewards[(state, action)] = []
                self.rewards[(state, action)].append(G)
                value = self.average_list(self.rewards[(state, action)])
                self.update_q(state, action, value)
                state_index = (state[0]-12, state[1]-1, int(state[2]))
                max_a = numpy.argmax(self.__state_action_value_table[state_index])
                # update_value = (max_a + (1 -2*max_a)*(epsilon/2))
                # self.update_policy_table(state, update_value)
                self.update_policy_table(state, max_a)