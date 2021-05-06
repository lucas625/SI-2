"""
Script for Q learning.
"""

import argparse

from evaluator import Evaluator
from learner import QLearner
from learner import TDLearner


def _arguments_definition():
    """
    Method for creating the possible parameters for execution.
    :return ArgumentParser:
    """
    parser = argparse.ArgumentParser(description='Runs the Q-learning.')
    parser.add_argument(
        '--number-of-episodes',
        default=500000,
        type=int,
        help='The number of episodes (Default is 500000).')
    parser.add_argument(
        '--number-of-evaluation-intervals',
        default=10,
        type=int,
        help='The number of evaluation intervals (Default is 10).')
    parser.add_argument(
        '--alpha',
        default=0.009,
        type=float,
        help='The learning rate (Default is 0.009).')
    parser.add_argument(
        '--gamma',
        default=1.0,
        type=float,
        help='The discount factor (Default is 1.0).')
    parser.add_argument(
        '--epsilon',
        default=0.1,
        type=float,
        help='The chance of performing a random action (Default is 0.1).')

    return parser.parse_args()


if __name__ == '__main__':
    args = _arguments_definition()

    learner = TDLearner()
    _, episodes_rewards = learner.learn(args.number_of_episodes, args.alpha, args.gamma, args.epsilon_min)

    Evaluator.evaluate(episodes_rewards, args.number_of_evaluation_intervals, '')
