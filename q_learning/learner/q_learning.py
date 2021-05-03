import sys
import random
from collections import defaultdict

import numpy as np


class QLearner:

    def __init__(self, env):
        self.env = env

    def epsilon_greedy(self, Q, state, nA, eps):
        """Selects epsilon-greedy action for supplied state.

        Params
        ======
            Q (dictionary): action-value function
            state (int): current state
            nA (int): number actions in the environment
            eps (float): epsilon
        """
        if random.random() > eps: # select greedy action with probability epsilon
            return np.argmax(Q[state])
        else:                     # otherwise, select an action randomly
            return random.choice(np.arange(self.env.action_space.n))

    def learn(self, num_episodes, alpha, gamma=1.0, epsmin=0.01):
        """Q-Learning - TD Control

        Params
        ======
            num_episodes (int): number of episodes to run the algorithm
            alpha (float): learning rate
            gamma (float): discount factor
            plot_every (int): number of episodes to use when calculating average score
        """
        nA = self.env.action_space.n                # number of actions
        Q = defaultdict(lambda: np.zeros(nA))  # initialize empty dictionary of arrays

        for i_episode in range(1, num_episodes+1):
            # monitor progress
            if i_episode % 100 == 0:
                print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
                sys.stdout.flush()
            score = 0                                              # initialize score
            state = self.env.reset()                                    # start episode
            eps = max(1.0 / i_episode ,epsmin)                                 # set value of epsilon


            while True:
                action = self.epsilon_greedy(Q, state, nA, eps)         # epsilon-greedy action selection
                next_state, reward, done, info = self.env.step(action)  # take action A, observe R, S'
                score += reward                                    # add reward to agent's score
                Q[state][action] = self.update_q_sarsamax(alpha, gamma, Q, \
                                                          state, action, reward, next_state)
                state = next_state                                 # S <- S'
                # note: no A <- A'
                if done:
                    break
        return Q

    @staticmethod
    def update_q_sarsamax(alpha, gamma, Q, state, action, reward, next_state=None):
        """Returns updated Q-value for the most recent experience."""
        current = Q[state][action]  # estimate in Q-table (for current state, action pair)
        Qsa_next = np.max(Q[next_state]) if next_state is not None else 0  # value of next state
        target = reward + (gamma * Qsa_next)               # construct TD target
        new_value = current + (alpha * (target - current)) # get updated value
        return new_value

    @staticmethod
    def get_probs(Q_s, epsilon, nA): #nA is no. of actions in the action space
        # obtains the action probabilities corresponding to epsilon-greedy policy
        policy_s = np.ones(nA) * epsilon / nA
        best_a = np.argmax(Q_s)
        policy_s[best_a] = 1 - epsilon + (epsilon / nA)
        return policy_s
