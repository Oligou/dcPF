#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: ogouvert

x_l \sim ZTP(p) (zero-truncated Poisson distribution)
which implies: y \sim sumZTP(n,p)
"""

import numpy as np
import scipy.special as special
import scipy.sparse as sparse
import dcpf

class dcpf_ZTP(dcpf.dcpf):

    MAX_MATRIX = 1000
    
    def __init__(self, K, p, t=1.,
                 alphaW=1., alphaH=1., betaW=1., betaH=1.):
        """
        p (float) - p=exp(\theta) where \theta is the natural parameter of the EDM
        """
        assert p>=0
        self.p = p
        dcpf.dcpf.__init__(self,K=K, t=t,
                 alphaW = alphaW, alphaH = alphaH, betaW = betaW, betaH = betaW)
        self.classname = 'dcpf_ZTP'
        
    def c_en(self,Y,s):
        y = Y.data
        # Limit cases
        if self.p==0:  # PF on raw data
            en = Y.data
            elbo = - np.sum(special.gammaln(y+1)) + np.sum(y*np.log(s))
        elif self.p==float('inf'):  # PF on binary data
            en = y>0
            elbo = np.sum(np.log(s))
        else: # 0 < p < +inf - Trade-off
            en, logZ = self.q_N(y,s)
            # ELBO
            elbo_cst = -np.sum(special.gammaln(y+1)) + Y.sum()*np.log(self.p)
            elbo = elbo_cst + logZ.sum()
        return en, elbo
  
    def q_N(self,Y,S):
        R = S/np.expm1(self.p) 
        assert all(Y>0)
        assert Y.shape == R.shape
        En = np.ones_like(Y, dtype=float)
        logZ = np.log(R)
        unique_value = np.unique(Y)
        unique_value = unique_value[unique_value>1]
        for y in unique_value:
            idx = (Y==y)
            N = np.arange(1,y+1)[np.newaxis,:]
            r = R[idx,np.newaxis]
            logSt2 = self.calculate_logSt2(y)
            logz = special.logsumexp(N*np.log(r) + logSt2, axis=1, keepdims=True)
            q_n = np.exp(N*np.log(r) + logSt2 - logz)
            En[idx] = (N*q_n).sum(axis=1)
            logZ[idx] = logz[:,0]
        return En, logZ

    def calculate_logSt2(self,y):
        if y <= dcpf_ZTP.MAX_MATRIX:
            return self.M[y,1:y+1]
        else:
            N = np.arange(0,y+1)
            V = float(y)/N
            G = -np.real(special.lambertw(-V*np.exp(-V)))
            approx = (np.log(y-N) - np.log(y) - np.log(1.-G))/2. + \
                    - N*np.log(G) + (y-N)*(np.log(y-N) - np.log(V-G) - 1.) + \
                    + special.gammaln(y+1) - special.gammaln(N+1) - special.gammaln(y-N+1)
            approx2 = y*np.log(N) - special.gammaln(N+1)
            
            approx[approx==float('inf')] = approx2[approx==float('inf')]
            approx[np.isnan(approx)] = approx2[np.isnan(approx)]
            approx[y] = 0.
            return approx[1:]

    def opt_param_xl(self,s_en, s_y):
        """" Hyper-parameter optimization : Newton algortithm """
        ratio = float(s_en)/s_y
        p = self.p
        cost_init = s_y*np.log(p) - s_en*np.log(np.expm1(p))
        for n in range(10):
            f = (1-np.exp(-p))/p
            grad = -(1.-np.exp(-p))/(p**2) + np.exp(-p)/p
            delta = (f-ratio)/grad
            while p - delta < 0 :
                delta = delta/2
            p = p - delta
        cost = s_y*np.log(p) - s_en*np.log(np.expm1(p))
        # Is the p better?
        if cost>cost_init:
            self.p = p
    
    def stirling_matrix(self,Y):
        self.M = logSt2_fast_matrix(np.min([Y.max(),dcpf_ZTP.MAX_MATRIX]))

def logSt2_fast_matrix(Ymax):
    M = np.empty((Ymax+1,Ymax+1))
    M[:] = np.nan
    
    S0 = np.array([0.])
    Sinf = np.array([-np.float('inf')])
    L = np.concatenate((Sinf, S0))
    
    M[0,0] = 0.
    M[1,:2] = L
    
    for y in range(Ymax+1)[2:]:
        b = L[:-1]
        a = L[1:]
        K = np.arange(1,y)
        S = a + np.log(K + np.exp(b-a))
        L = np.concatenate((Sinf,S,S0))
        M[y,0:y+1] = L
    return M

#%% Synthetic example
if False:
    import matplotlib.pyplot as plt

    U = 1000
    I = 1000
    K = 5
    
    np.random.seed(93)
    W = np.random.gamma(1.,1., (U,K))
    H = np.random.gamma(1.,1., (I,K))
    L = np.dot(.01*W,H.T)
    Ya = np.random.poisson(L)
    Y = sparse.csr_matrix(Ya)
        
    #%%
    model = dcpf_ZTP(K=K,p=1.)
    model.fit(Y,verbose=True, opt_hyper=['p','beta'], save=False)
    
    #%%
    Ew = model.Ew
    Eh = model.Eh            
    Yr = np.dot(Ew,Eh.T)
    
    #%%
    plt.figure('Truth')
    plt.imshow(L,interpolation='nearest')
    plt.colorbar()
    plt.figure('Obs')
    plt.imshow(Ya,interpolation='nearest')
    plt.colorbar()
    plt.figure('Reconstruction')
    plt.imshow(Yr,interpolation='nearest')
    plt.colorbar()

    #%%
    plt.figure('elbo')
    plt.plot(model.Elbo)