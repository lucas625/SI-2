"""
Module for evaluating the Q learning.
"""

import random

import numpy as np


class QEvaluator:
    """
    Class for evaluating the Q learning.
    """

    def __init__(self, env):
        """
        Class constructor.
        :param BlackjackEnv env: the blackjack environment.
        """
        self._env = env
        self._number_of_actions = self._env.action_space.n

    def evaluate(self, q_learning_table, number_of_tests):
        """
        Performs the evaluation of the learning table.
        :param defaultdict q_learning_table: the table with the q learning values.
        :param int number_of_tests: the total number of tests to be run.
        """
        victories = 0
        for test_index in range(1, number_of_tests + 1):
            self._monitor_evaluation_progress(test_index, number_of_tests)
            reward = self._perform_test(q_learning_table)
            if reward > 0:
                victories += 1
        print('\nAgent won {} out of {} tests.'.format(victories, number_of_tests))
        # TODO: Generate csv.

    def _perform_test(self, q_learning_table):
        # TODO: add docstring
        state = self._env.reset()
        done = False
        while not done:
            if state in q_learning_table:
                p = self._get_probs(q_learning_table[state], 0)
                action = np.random.choice(np.arange(self._number_of_actions), p=p)
            else:
                action = self._env.action_space.sample()
            next_state, reward, done, info = self._env.step(action)
            state = next_state
        return reward

    def _get_probs(self, Q_s, epsilon):
        # TODO: refactor this method and add docstring
        # obtains the action probabilities corresponding to epsilon-greedy policy
        policy_s = np.ones(self._number_of_actions) * epsilon / self._number_of_actions
        best_a = np.argmax(Q_s)
        policy_s[best_a] = 1 - epsilon + (epsilon / self._number_of_actions)
        return policy_s

    @staticmethod
    def _monitor_evaluation_progress(test_index, number_of_tests):
        """
        Shows the evaluation progress on terminal.
        :param int test_index: the 1-based index of the current evaluation.
        :param int number_of_tests: the total number of tests.
        """
        if test_index % 100 == 0:
            print('\rEvaluating test {}/{}'.format(test_index, number_of_tests), end='')
