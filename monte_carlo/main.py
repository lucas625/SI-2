#!/bin/env python3
import numpy
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from agent import MonteCarloBlackjackAgent, generate_epsilon_greedy_policy

def plot_value_function(V, title = "Value Function"):
    '''
    Plots the value function as a surface plot.
    '''
    min_x = 11 # min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())
    
    x_range = numpy.arange(min_x, max_x + 1)
    y_range = numpy.arange(min_y, max_y + 1)
    X, Y = numpy.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = numpy.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, numpy.dstack([X, Y]))
    Z_ace = numpy.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, numpy.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize = (20, 10))
        ax = fig.add_subplot(111, projection = '3d')
        surf = ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1,
                               cmap = matplotlib.cm.coolwarm, vmin = -1.0, vmax = 1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))

if __name__ == '__main__':
    agent = MonteCarloBlackjackAgent()
    
    random_policy = defaultdict(lambda: numpy.zeros(2))
    p = generate_epsilon_greedy_policy(random_policy, epsilon = 0.1)
    test = agent.learn(p)
    
    for _ in range(500000):
        next(test)
        if (_+1 in [50000, 100000, 500000]):    
            V = defaultdict(float)
            for state, actions in agent.policy.items():
                V[state] = numpy.max(actions)

            plot_value_function(V, title = 'Optimal Value Function')