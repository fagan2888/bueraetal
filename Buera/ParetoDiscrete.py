"""Function to discretise a Pareto with scale x_m and tail coefficient eta
Arguments are:
p - Probability such that q_p = F^-1(p) is the upper bound of the grid
x_m = scale parameter
eta - tail coefficient
N - number of interior points of the grid
"""
import numpy as np
from scipy.stats import pareto

def ParetoDiscrete(eta, x_m = 1, p = 0.95, N = 8):
    q_p = x_m/(1-p)**(1/eta)
    grid = np.linspace(x_m, q_p, N+2)
    probGrid = np.empty(N+2)
    for i, x in enumerate(grid):
        if i == 0:
            probGrid[i] = pareto.cdf((grid[i+1] + x)/2, b = eta, scale = x_m)
        else: 
            if i == len(grid)-1:
                probGrid[i] = 1- pareto.cdf((grid[i-1] + x)/2, b = eta, scale = x_m )
            else:
                probGrid[i] = pareto.cdf((grid[i+1] + x)/2, b = eta, scale = x_m) - \
                pareto.cdf((grid[i-1] + x)/2, b = eta, scale = x_m )
    return [grid, probGrid]