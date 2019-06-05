#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: ogouvert

x_l - 1 \sim NB(a,p) (negative-binomial distribution)
which implies: y - n \sim NB(a*n,p)
"""

import numpy as np
import scipy.special as special
import scipy.sparse as sparse
import dcpf

class dcpf_sNB(dcpf.dcpf):
    
    def __init__(self, K, p, a=1, t=1.,
                 alphaW=1., alphaH=1., betaW=1., betaH=1.):
        """
        p (float) - p=exp(\theta) where \theta is the natural parameter of the EDM
        a (float, >0) - \kappa = (1,a)^T is the dispersion parameter of the EDM
        """
        assert p>=0 and p<=1
        assert a>0
        self.a = a
        self.p = p
        dcpf.dcpf.__init__(self,K=K, t=t,
                 alphaW = alphaW, alphaH = alphaH, betaW = betaW, betaH = betaW)
        self.classname = 'dcpf_sNB'
        
    def c_en(self,Y,s):
        y = Y.data
        # Limit cases
        if self.p==0: # PF on raw data
            en = Y.data
            elbo = - np.sum(special.gammaln(y+1)) + np.sum(y*np.log(s))
        elif self.p==1:  # PF on binary data
            en = y>0
            elbo = np.sum(np.log(s))
        else: # 0 < p < 1 - Trade-off
            en, em, logZ = self.q_N(y,s)
            self.s_em = em.sum()
            # ELBO
            elbo_cst = Y.sum()*np.log(self.p)
            elbo = elbo_cst + logZ.sum()            
        return en, elbo
    
    def opt_param_xl(self, s_en, s_y):
        """" 
        Hyper-parameter optimization : 
            closed-form update with the augmentation (see article) 
        """
        self.a = self.s_em/float(-s_en *np.log(1-self.p))
        self.p = float(s_y-s_en)/((self.a-1.)*s_en + s_y)
    
    def q_N(self,Y,S):
        assert all(Y>0)
        assert Y.shape == S.shape
        En = np.ones_like(Y, dtype=float)
        Em = np.zeros_like(Y, dtype=float)
        logZ = self.a*np.log(1.-self.p) + np.log(S) - np.log(self.p)
        unique_value = np.unique(Y)
        unique_value = unique_value[unique_value>1]
        for y in unique_value:
            idx = (Y==y)
            N = np.arange(1,y+1)[np.newaxis,:]
            s = S[idx,np.newaxis]
            logSt = special.gammaln((self.a-1.)*N+y) - special.gammaln(y-N+1) + \
                    - special.gammaln(N*self.a) - special.gammaln(N+1) + \
                    + N*self.a*np.log(1.-self.p) - N*np.log(self.p) + N*np.log(s)
            logz = special.logsumexp(logSt, axis=1, keepdims=True)
            q_n = np.exp(logSt - logz)
            En[idx] = (N*q_n).sum(axis=1)
            Em[idx] = (N*self.a*(special.digamma((self.a-1.)*N+y) - special.digamma(N*self.a))*q_n).sum(axis=1)
            logZ[idx] = logz[:,0]
        return En, Em, logZ
    
    def create_filename(self,prefix,suffix):
        if prefix is not None:
            prefix = prefix+'_'
        else:
            prefix = ''
        if suffix is not None:
            suffix = '_'+suffix
        else:
            suffix = ''
        return prefix + self.classname + \
                '_K%d' % (self.K) + \
                '_p%.3e' % (self.p) + \
                '_a%.1e' % (self.a) + \
                '_alpha%.2f_%.2f' %(self.alphaW, self.alphaH) + \
                '_beta%.2f_%.2f' % (self.betaW, self.betaH) + \
                '_opthyper_' + '_'.join(sorted(self.opt_hyper)) + \
                '_precision%.1e' %(self.precision) + \
                '_seed' + str(self.seed) + suffix

#%% Synthetic example
if False:
    import matplotlib.pyplot as plt

    U = 1000
    I = 1000
    K = 9
    
    np.random.seed(913)
    W = np.random.gamma(1.,1., (U,K))
    H = np.random.gamma(1.,1., (I,K))
    L = np.dot(.01*W,H.T)
    A = np.random.gamma(L,size=(U,I))
    Ya = np.random.poisson(A)
    Y = sparse.csr_matrix(Ya)
        
    #%%
    model = dcpf_sNB(K=K, p=0.5, a=1.)
    model.fit(Y,verbose=True, opt_hyper=['beta','p'], save=False)
    
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
    