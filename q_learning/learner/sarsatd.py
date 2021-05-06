"""
Module for learning with Q learn.
"""

import random
from collections import defaultdict

import gym
import numpy as np



class TDLearner:
    """
    Class for controlling the Q learning.
    """

    def __init__(self):
        """
        Class constructor.
        """
        self._env = gym.make('Blackjack-v0')
        self._number_of_actions = self._env.action_space.n

    def learn(self, number_of_episodes, alpha, gamma, epsilon_min):
        """
        Q-Learning - TD Control
        :param int number_of_episodes: the number of episodes.
        :param float alpha: the learning rate.
        :param float gamma: discount factor.
        :param float epsilon_min: minimum chance of taking a random action.
        :return defaultdict:
            a defaultdict where each key is a state (hand, opponent-hand, you-have-usable-ace) and the values of the
            keys are a list of 2 items, the first is the average reward of sticking with your hand and the second is the
            average reward of drawing.
        :return list[float]: The total reward of each episode.
        """
        q_learning_table = defaultdict(lambda: np.zeros(self._number_of_actions))  # initializing the table
        episodes_rewards = np.zeros(number_of_episodes)

        for episode_index in range(1, number_of_episodes + 1):
            self._monitor_learning_progress(episode_index, number_of_episodes)
            episodes_rewards[episode_index-1] = self._iterate_episode(
                q_learning_table, episode_index, alpha, gamma, epsilon_min)
            # TODO: evaluate every x episodes

        print()
        return q_learning_table, episodes_rewards

    def _iterate_episode(self, q_learning_table, episode_index, alpha, gamma, epsilon):
        """
        Perform the actions necessary for running the episode.
        :param defaultdict q_learning_table: the table with the q learning values.
        :param int episode_index: the 1-based episode index.
        :param float alpha: the learning rate.
        :param float gamma: discount factor.
        :param float epsilon_min: minimum chance of taking a random action.
        :return float: the score on the episode.
        """
        score = 0
        state = self._env.reset()  # start episode
        done = False
        action = self._choose_action_by_epsilon_greedy(q_learning_table, state, epsilon)
        while not done:
            next_state, reward, done, info = self._env.step(action)
            score += reward
            if not done:
                next_action = self._choose_action_by_epsilon_greedy(q_learning_table, state, epsilon)
                self._update_table(q_learning_table, alpha, gamma, state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
            if done:
                self._update_table(q_learning_table, alpha, gamma, state, action, reward)
        return score

    def _choose_action_by_epsilon_greedy(self, q_learning_table, state, epsilon):
        """
        Selects epsilon-greedy action for supplied state.
        :param defaultdict q_learning_table: the table with the q learning values.
        :param tuple[int, int, bool] state: the current state.
        :param float epsilon: the chance of taking a random action.
        :return int: the action to be taken.
        """
        if random.random() > epsilon:
            action = np.argmax(q_learning_table[state])
        else:
            action = random.choice(np.arange(self._number_of_actions))
        return action

    @staticmethod
    def _update_table(q_learning_table, alpha, gamma, state, action, reward, next_state=None, next_action=None):
        """
        Updated table for the most recent experience.
        :param defaultdict q_learning_table: the table with the q learning values.
        :param tuple[int, int, bool] state: the current state.
        :param float alpha: the learning rate.
        :param float gamma: discount factor.
        :param tuple[int, int, bool] state: the current state.
        :param int action: the action taken.
        :param float reward: the reward for the action:
        :param tuple[int, int, bool] next_state: the next state.
        """
        current_estimation = q_learning_table[state][action]  # estimate in Q-table (for current state, action pair)
        next_state_value = q_learning_table[next_state][next_action] if next_state is not None else 0
        target = reward + (gamma * next_state_value)  # construct TD target
        new_value = current_estimation + (alpha * (target - current_estimation))  # get updated value
        q_learning_table[state][action] = new_value

    @staticmethod
    def _monitor_learning_progress(episode_index, number_of_episodes):
        """
        Shows the learning progress on terminal.
        :param int episode_index: the 1-based index of the current episode.
        :param int number_of_episodes: the total number of episodes.
        """
        if episode_index % 100 == 0:
            print('\rLearning episode {}/{}'.format(episode_index, number_of_episodes), end='')
