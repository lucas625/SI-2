#!/bin/env python3

import matplotlib.pyplot as plt
from agent import MonteCarloBlackjackAgent

if __name__ == '__main__':
    agent = MonteCarloBlackjackAgent()
    
    for i in range(1000):
        episode, _ = agent.generate_random_episode()
        agent.learn(episode)
        
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = agent.q.nonzero()
    ax.plot_trisurf(X, Y, Z, vmin=-1.0, vmax=1.0)