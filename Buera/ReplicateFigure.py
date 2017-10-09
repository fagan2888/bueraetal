'''
Created on 7 de out de 2017

@author: luisfantozzialvarez
'''
import numpy as np
from Buera import Buera
from scipy.interpolate.interpolate import interpn

#Parametrization according to Buera (2011)
eta = 4.84
alpha = (1/eta + 0.79)*0.3
theta = 0.79 - alpha
kappa_s, kappa_m, delta, gamma, beta, sigma, epsilon, psi =  0.0, 4.68, 0.06, 0.89, 0.92, 1.5, 1.0,  0.91
rho = 1 - beta


perfect_benchmark =  Buera(alpha, theta, kappa_s, kappa_m, delta, gamma, eta, rho, sigma, epsilon, psi, phi = 1.0,  upperA = 1000, \
                           nA = 1000, nZ = 3, pgrid = 0.95)



excess = perfect_benchmark.excess_demand(0.04, 1, 2,step = 1000, maxiter = 2000)
print(excess)
