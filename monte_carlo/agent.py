#!/bin/env python3

import gym
import numpy
from collections import defaultdict

class MonteCarloBlackjackAgent:
    def __init__ (self, seed = None):
        self.__env = gym.make('Blackjack-v0')
        self.__env.seed(seed)
        
        self.__value_table = defaultdict(lambda: numpy.zeros(2))
        
        self.__reward_sum = defaultdict(float)
        self.__visits = defaultdict(int)

    
    @property
    def policy(self):
        return self.__value_table
    
    def evaluate(self, state, epsilon):
        probs = numpy.ones(2) * epsilon / 2
        best_action = numpy.argmax(self.__value_table[state])
        probs[best_action] = 1 - epsilon + (epsilon / 2)
        return probs
    
    
    def __generate_episode(self, epsilon):
        '''
        Generates random episodes based on the current policy
        '''
        episode = []
        done = False
        state = self.__env.reset()
        while (not done):
            state_prime = state
            probs = self.evaluate(state, epsilon)
            action = numpy.random.choice([0, 1], p=probs)
            state, reward, done, _ = self.__env.step(action)
            episode.append((state_prime, action, reward))
        return episode, state, reward
        

    def learn (self, gamma = 1, epsilon = 0.1):
        while (True):
            episode, f_state, f_reward = self.__generate_episode(epsilon)
            G = 0
            for i, step in enumerate(episode[::-1]):
                state, action, reward = step
                pair = (state, action)
                G = gamma * G + reward
                if (pair not in [(x[0], x[1]) for x in episode][i::-1][1:]):
                    self.__visits[pair] += 1
                    self.__reward_sum[pair] += G
                    self.__value_table[state][action] = \
                        self.__reward_sum[pair] / self.__visits[pair]
            yield f_reward