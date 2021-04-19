"""
Blackjack with Q-learning module.
"""

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed


class BlackjackQLearningRunner:
    """Class for running blackjack with Q learning"""

    def __init__(self):
        self.env = rlcard.make('blackjack', config={'seed': 0})

        # Set a global seed
        set_global_seed(0)

        # Set up agents
        agent_0 = RandomAgent(action_num=self.env.action_num)
        self.env.set_agents([agent_0])

    def run(self, number_of_episodes):
        """
        Runs blackjack with Q learning.
        :param int number_of_episodes:
        """
        for episode in range(number_of_episodes):

            trajectories, _ = self.env.run(is_training=False)

            print('\nEpisode {}'.format(episode))
            for ts in trajectories[0]:
                print('State: {}, Action: {}, Reward: {}, Next State: {}, Done: {}'.format(ts[0], ts[1], ts[2], ts[3],
                                                                                           ts[4]))


if __name__ == '__main__':
    blackjack_q_learning_runner = BlackjackQLearningRunner()
    blackjack_q_learning_runner.run(number_of_episodes=200)
