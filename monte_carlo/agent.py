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
    
    def evaluate(self, state):
        return self.__value_table[state]
    
    
    def __generate_episode(self, policy):
        '''
        Generates random episodes based on the current policy
        '''
        episode = []
        done = False
        state = self.__env.reset()
        while (not done):
            state_prime = state
            # prob = self.evaluate(state)
            probs = policy(state)
            action = numpy.random.choice([0, 1], p=probs)
            state, reward, done, _ = self.__env.step(action)
            episode.append((state_prime, action, reward))
        return episode
        

    def learn (self, policy, gamma = 1):
        while (True):
            episode = self.__generate_episode(policy)
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
            yield

def generate_epsilon_greedy_policy(policy, epsilon):
    def policy_greedy(state):
        prob = numpy.ones(2, dtype=float) * epsilon / 2
        best_action = numpy.argmax(policy[state])
        prob[best_action] += 1. - epsilon
        return prob
    return policy_greedy