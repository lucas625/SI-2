import gym

from learner import QLearner
from evaluator import QEvaluator

if __name__ == '__main__':
    # TODO: move env to learner
    # TODO: use argparser
    env = gym.make('Blackjack-v0')

    learner = QLearner(env)
    q_learning_table = learner.learn(50000, 0.01, 0.25, epsilon_min=0.2)

    evaluator = QEvaluator(env)
    evaluator.evaluate(q_learning_table, 100)
