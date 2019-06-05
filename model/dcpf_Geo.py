#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: ogouvert

x_l \sim Geo(a,p) (shifted geometric distribution)
which implies: y - n \sim NB(n,p)
"""

import numpy as np
import scipy.special as special
import scipy.sparse as sparse
import dcpf

class dcpf_Geo(dcpf.dcpf):
    def __init__(self, K, p, t=1.,
                 alphaW=1., alphaH=1., betaW=1., betaH=1.):
        """
        p (float) - p=exp(\theta) where \theta is the natural parameter of the EDM
        """
        assert p>=0 and p<=1
        self.p = p
        dcpf.dcpf.__init__(self,K=K, t=t,
                 alphaW = alphaW, alphaH = alphaH, betaW = betaW, betaH = betaW)
        self.classname = 'dcpf_Geo'
        
    def c_en(self,Y,s):
        y = Y.data
        # Limit cases
        if self.p==0: # PF on raw data
            en = Y.data
            elbo = - np.sum(special.gammaln(y+1)) + np.sum(y*np.log(s))
        elif self.p==1: # PF on binary data
            en = y>0
            elbo = np.sum(np.log(s))
        else: # 0 < p < 1 - Trade-off
            en, logZ = self.q_N(y,s)
            # ELBO
            elbo_cst = -np.sum(special.gammaln(y+1)) + Y.sum()*np.log(self.p)
            elbo = elbo_cst + logZ.sum()
        return en, elbo
    
    def opt_param_xl(self,s_en, s_y):
        """" Hyper-parameter optimization :closed-form update """
        self.p = 1.-float(s_en)/s_y
    
    def q_N(self,Y,S):
        R = S*(1./self.p-1.) 
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
            logSt3 = np.log(y) - np.log(N) + 2*special.gammaln(y) - 2*special.gammaln(N) - special.gammaln(y-N+1)
            logz = special.logsumexp(N*np.log(r) + logSt3, axis=1, keepdims=True)
            q_n = np.exp(N*np.log(r) + logSt3 - logz)
            En[idx] = (N*q_n).sum(axis=1)
            logZ[idx] = logz[:,0]
        return En, logZ

#%% Synthetic example
if False:
    import matplotlib.pyplot as plt

    U = 1000
    I = 1000
    K = 3
    
    np.random.seed(93)
    W = np.random.gamma(1.,1., (U,K))
    H = np.random.gamma(1.,1., (I,K))
    L = np.dot(.01*W,H.T)
    Ya = np.random.poisson(L)
    Y = sparse.csr_matrix(Ya)
        
    #%%
    model = dcpf_Geo(K=K,p=0.5)
    model.fit(Y,verbose=True, opt_hyper=['p','beta'], save =False)
    
    #%%
    Ew = model.Ew
    Eh = model.Eh
    Yr = np.dot(Ew,Eh.T)
    
    #%%
    plt.figure('Obs')
    plt.imshow(Ya,interpolation='nearest')
    plt.colorbar()
    plt.figure('Truth')
    plt.imshow(L,interpolation='nearest')
    plt.colorbar()
    plt.figure('Reconstruction')
    plt.imshow(Yr,interpolation='nearest')
    plt.colorbar()

    #%%
    plt.figure('elbo')
    plt.plot(model.Elbo)
    