import gym
import numpy as np

from learner.q_learning import QLearner

if __name__ == '__main__':
    env = gym.make('Blackjack-v0')

    learner = QLearner(env)
    Q_sarsamax = learner.learn(1000, 0.01)

    # obtain the corresponding state-value function
    V = dict((k,np.max(v)) for k, v in Q_sarsamax.items())

    nA = env.action_space.n 

    for i_episode in range(10):
        state = env.reset()
        while True:
            print(state)
            action = np.random.choice(np.arange(nA), p=QLearner.get_probs(Q_sarsamax[state], 0, nA)) \
                                        if state in Q_sarsamax else env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
                print('Game has ended! Your Reward: ', reward)
                print('You won :)\n') if reward > 0 else print('You lost :(\n')
                break