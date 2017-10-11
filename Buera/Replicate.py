import numpy as np
from Buera import Buera
from scipy.interpolate.interpolate import interpn
from datetime import datetime 

#Parametrization according to Buera (2011)
eta = 4.84
alpha = (1/eta + 0.79)*0.3
theta = 0.79 - alpha
kappa_s, kappa_m, delta, gamma, beta, sigma, epsilon, psi =  0.0, 4.68, 0.06, 0.89, 0.92, 1.5, 1.0,  0.91
rho = 1 - beta


perfect_benchmark =  Buera(alpha, theta, kappa_s, kappa_m, delta, gamma, eta, rho, sigma, epsilon, psi, phi = 1,  upperA = 500, \
                           nA = 1000, nZ = 3, pgrid = 0.95)

#Computes excess-demand for given vector of prices (r, w, ps)
start = datetime.now()
print('Excess demand given (r,w,ps) = (0.04,0.9,0.8)', perfect_benchmark.excess_demand(0.04, 0.9, 0.8))
print(datetime.now() - start)


#Computes equilibirum prices for perfect-benchmark
eq_rates = perfect_benchmark.solve_model(np.array([0.03,0.5,0.6]), rep_tot = 300)

print('Equilibrium prices are:', eq_rates)