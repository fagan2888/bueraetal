'''
Solves Buera, Kaboski, Shin (2011) - Finance and Development: A tale of two sectors; in continuous time
Upwind scheme and computation of stationary distribution is based on Ahn and Moll's solution of the Aiyagari model 
(check their notebook http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/aiyagari_continuous_time.ipynb)
'''
from ParetoDiscrete import ParetoDiscrete
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse

class Buera(object):
    
    def __init__(self, alpha, theta, kappa_s, kappa_m, delta, gamma, eta, rho, sigma, epsilon, psi, phi, upperA, nA, nZ, pgrid):
        self.alpha, self.theta, self.kappa_s, self.kappa_m, self.delta, self.gamma, self.eta, \
        self.rho, self.sigma, self.epsilon, self.psi, self.phi = alpha, theta, kappa_s, kappa_m, delta, gamma, eta,\
        rho, sigma, epsilon, psi, phi
        self.gridz = ParetoDiscrete(self.eta, x_m = 1, p = pgrid, N = nZ)[0]
        self.probz = ParetoDiscrete(self.eta, x_m = 1, p = pgrid, N = nZ)[1]
        self.muZ = np.outer(self.probz, self.probz).flatten('C')

        #We define Z = {(zm1, zs1), (zm2, zs1) ... (zmL, zs1), (zm1, zs2) ... (zmL, zs2) ... (zmL zsML)
        #Given this, we define Zm, Zs:
        self.Zm = np.outer(self.gridz, np.ones(len(self.gridz))).flatten('F')
        self.Zs = np.outer(self.gridz, np.ones(len(self.gridz))).flatten('C')
        
        #Defines grid of A
        self.grida = np.linspace(0, upperA, nA + 2)
        
        #Defines state space S =  [SA, SZ] = [SA, SZm, SZs]
        self.SA = np.outer(self.grida, np.ones(len(self.Zm))).flatten('F')
        self.SZm = np.outer(self.Zm, np.ones(len(self.grida))).flatten('C')
        self.SZs = np.outer(self.Zs, np.ones(len(self.grida))).flatten('C')
        
        #Creates dimension of state space
        self.dimS = len(self.SA)
        
        #da - to be used in nummerical differentiation
        self.da = self.grida[1] - self.grida[0]
     
        
        #creates matrix N
        self.N = sparse.kron(np.outer(self.muZ, np.ones(len(self.muZ))), sparse.eye(len(self.grida)))

        #Identity matrix with dimension of states
        self.Is = sparse.eye(self.dimS) 
        
        #Reshaped Matrix of A (len(Zm)xlen(grida)) (to be used in upwiund scheme)
        self.SAreshape = np.outer(np.ones(len(self.Zm)), self.grida)
        
        #Transition matrix due to the poisson shock
        self.Q = sparse.csr_matrix((1-self.gamma)*(np.transpose(self.N) - self.Is))
        
        
        
    #Utility function
    def u(self, cs, cm):
        if self.epsilon != 1:
            return (1/(1-self.sigma))*(self.psi*cs**(1- 1/self.epsilon) + (1-self.psi)*cm**(1- 1/self.epsilon))**((1-self.sigma)/(1- 1/self.epsilon))
        else:
            return (self.sigma-1)*((self.psi)*np.log(cs)+(1-self.psi)*np.log(cm))
        
    #production function
    def f(self, k, l):
        return (k**self.alpha)*(l**self.theta)
    
    #solves and returns M(S), Ks(S), Km(S), L(S) for a given guess of r, w, ps
    def occup_choice(self, r, w, ps):
        #unconstrained demand for capital in each sector
        Kus = (((self.alpha**(1-self.theta))*(self.theta**self.theta)*ps*self.SZs)/((w**self.theta)*(r+self.delta)**(1-self.theta)))**(1/(1-self.alpha-self.theta))
        Kum = (((self.alpha**(1-self.theta))*(self.theta**self.theta)*self.SZm)/((w**self.theta)*(r+self.delta)**(1-self.theta)))**(1/(1-self.alpha-self.theta))

        #checks whether the economy is in perfect credit benchmark or not and defines appropriate demand
        if self.phi == 1:
            Kds = Kus
            Kdm = Kum
        else:
            Kds = np.min([self.SA/(1-self.phi),Kus], axis =0)
            Kdm = np.min([self.SA/(1-self.phi),Kum], axis =0)
        
        #Labour demand in each sector
        Lds = ((ps*self.SZs*(Kds**self.alpha)*self.theta)/w)**(1/(1-self.theta))
        Ldm = ((self.SZm*(Kdm**self.alpha)*self.theta)/w)**(1/(1-self.theta))
        
        #Computes profits in each sector
        PIs = ps*self.SZs*self.f(Kds, Lds) - w*Lds - (r+self.delta)*Kds - ps*self.kappa_s
        PIm = self.SZm*self.f(Kdm, Ldm) - w*Ldm - (r+self.delta)*Kdm - self.kappa_m
        
        #wage repeated accross dimensions
        Wrep = np.repeat(w, self.dimS)
        
        #Optimal income
        MS = np.max([Wrep, PIs, PIm], axis = 0)
        
        is_labour = (Wrep>=MS)
        is_services = (PIs>=MS)
        is_manufacturing = (PIm>=MS)
        
        #Computes labour supply  and demand for factors given occupational choices
        L_supply = 1*is_labour
        Ks_demand =  Kds*is_services
        Km_demand =  Kdm*is_manufacturing
        Ls_demand =  Lds*is_services
        Lm_demand =  Ldm*is_manufacturing
        
        
        #Checks if there are no individuals indifferent between sectors. Sets them to workers or services accordingly
        if np.sum(is_services*is_manufacturing) > 0:
            Km_demand[np.where(is_services*is_manufacturing==1)] =0
            Lm_demand[np.where(is_services*is_manufacturing==1)] =0
            
        if np.sum(is_labour*is_services) > 0:
                Ks_demand[np.where(is_labour*is_services==1)] = 0
                Ls_demand[np.where(is_labour*is_services==1)] = 0
        
        if np.sum(is_manufacturing*is_labour) > 0:
                Km_demand[np.where(is_manufacturing*is_labour)==1] = 0
                Lm_demand[np.where(is_manufacturing*is_labour)==1] = 0
    
        return[MS, L_supply, Ks_demand, Km_demand, Ls_demand, Lm_demand]
        
    #Solves model using up-wind method 
    def solveupwind(self, Ms, r, w, ps, tol = 1e-3, maxiter = 1000, step = 1000):
        Mreshape = np.reshape(Ms, (len(self.Zm), len(self.grida)), order = 'C' )
        
        #Forward and backward savings matrix
        ssf = np.zeros((len(self.Zm), len(self.grida)))
        ssb = np.zeros((len(self.Zm), len(self.grida)))
        
        if self.epsilon != 1:
            cm_static = (Mreshape + r*self.SAreshape)/(1+ ps*(ps*(1-self.psi)/self.psi)**(-self.epsilon))
        else:
            cm_static = (1-self.psi)*(Mreshape + r*self.SAreshape)
        
        cs_static = (Mreshape + r*self.SAreshape - cm_static)/ps
        
        v = self.u(cs_static, cm_static)/self.rho

        #creates empty vectors to store results
        cm, cs, S, A = [], [], [], []
        
        for i in range(maxiter):

            dV = (v[:,1:] - v[:,:-1])/self.da
            dV[np.where(dV==0)] = 1e-10
            
            if self.epsilon != 1:
                cmf = (dV)**(-1/self.sigma)*((self.psi**self.epsilon*(1-self.psi)**(1-self.epsilon)*ps**(1-self.epsilon) \
                                    + (1-self.psi))**((1-self.epsilon*self.sigma)/(self.epsilon - 1))*(1-self.psi))**(1/self.sigma)
                csf = cmf*(ps*(1-self.psi)/self.psi)**(-self.epsilon)
            else:
                csf = (((self.sigma-1)*self.psi)/ps)*(dV)**(-1)
                cmf = ((self.sigma-1)*(1- self.psi))*(dV)**(-1)
            
            csf[np.where(csf<=0)] = 1e-10
            cmf[np.where(cmf<=0)] = 1e-10
            
            ssf[:,:-1] = Mreshape[:,:-1] + r*self.SAreshape[:,:-1] - cmf - ps*csf
            ssb[:,1:] = Mreshape[:,1:] + r*self.SAreshape[:,1:] - cmf - ps*csf
            is_forward = ssf>0
            is_backward = ssb<0
            
            cm = cm_static.copy()
            cs = cs_static.copy()

        
            cm[:,0] +=  is_forward[:,0]*(cmf[:,0] - cm[:,0])
            cs[:,0] += is_forward[:,0]*(csf[:,0] - cs[:,0])
            
            cm[:,-1] += is_backward[:,-1]*(cmf[:,-1] - cm[:,-1])
            cs[:,-1] += is_backward[:,-1]*(csf[:,-1] - cs[:,-1])

            cs[:,1:-1] += is_forward[:,1:-1]*(csf[:,1:] - cs[:,1:-1]) + is_backward[:,1:-1]*(csf[:,0:-1] - cs[:,1:-1])
            cm[:,1:-1] += is_forward[:,1:-1]*(cmf[:,1:] - cm[:,1:-1]) + is_backward[:,1:-1]*(cmf[:,0:-1] - cm[:,1:-1])
   
            
            helper = (-ssf*is_forward/self.da + ssb*is_backward/self.da).reshape(self.dimS)
            A = self.Q.copy()
            A += sparse.spdiags(helper, 0, self.dimS, self.dimS)
            helper = ((ssf*is_forward)/self.da).reshape(self.dimS)
            A += sparse.spdiags(np.hstack((0,helper)),1, self.dimS, self.dimS)
            helper = (-(ssb*is_backward)/self.da).reshape(self.dimS)
            A+= sparse.spdiags(helper[1:],-1, self.dimS, self.dimS)
            
            B = sparse.eye(self.dimS)*(1/step + self.rho) - A
            c = self.u(cs, cm).reshape(self.dimS) + (1/step)*v.reshape(self.dimS)
            
            v1 = spsolve(B,c).reshape((len(self.Zm), len(self.grida)))
            
            err = np.amax(np.abs(v1-v))
            v = v1
            print(err)
            if err < tol:
                S = ssf*is_forward + ssb*is_backward
                break
            
        return  [S, cm.reshape(self.dimS), cs.reshape(self.dimS), v, A]
    
    #Computes stationary distribution for solution using given matrix A from upwind method
    def statdist_upwind(self,A):
        A_prime = A.transpose().tocsr()
        b = np.zeros((self.dimS,1))
        b[0] = 0.1
        A_prime.data[1:A_prime.indptr[1]] = 0
        A_prime.data[0] = 1.0
        A_prime.indices[0] = 0
        A_prime.eliminate_zeros()
        g = spsolve(A_prime,b)
        return g/np.sum(g)
       
    #Computes excess demand in labour, asset and service (non-tradable) markets
    def excess_demand(self, r, w, ps, tol = 1e-3, maxiter = 200, step = 1000):
        MS, L_supply, Ks_demand, Km_demand, Ls_demand, Lm_demand = self.occup_choice(r, w, ps)
        Mreshape = np.reshape(MS, (len(self.Zm), len(self.grida)), order = 'C' )
        S, cm, cs, v, A = self.solveupwind(MS, r, w, ps, tol = tol, maxiter = maxiter, step = step)
        distribution = self.statdist_upwind(A)
        excess = np.array([np.sum((Ls_demand+Lm_demand-L_supply)*distribution),\
                          np.sum((Ks_demand+Km_demand - self.SA)*distribution),\
                          np.sum((cs - self.SZs*self.f(Ks_demand, Ls_demand) - self.kappa_s*(Ks_demand > 0))*distribution)])
        return excess       

    #solves model given grids for r, ps and w. Evaluates excess demand at grid and uses spline to smooth excess demand.
    def solve_perfect(self, guess, gridr, gridps, gridw, tol = 1e-3, maxiter = 200, step = 1000):
        excess_l = np.empty((len(gridr), len(gridps), len(gridw)))
        excess_k = np.empty((len(gridr), len(gridps), len(gridw)))
        excess_s = np.empty((len(gridr), len(gridps), len(gridw)))
        for i, r in enumerate(gridr):
            for j, ps in enumerate(gridps):
                for k, w in enumerate(gridw):
                    excess_l[i,j,k], excess_k[i,j,k], excess_s[i,j,k] = self.excess_demand(r, w, ps, tol, maxiter, step)
    
    


        