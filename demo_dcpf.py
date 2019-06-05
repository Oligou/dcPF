#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ogouvert
"""

#%% READ
import numpy as np
import scipy.sparse as sparse
import cPickle as pickle

from function.train_test import divide_train_test

from model.dcpf_Log import dcpf_Log
from model.dcpf_ZTP import dcpf_ZTP
from model.dcpf_Geo import dcpf_Geo
from model.dcpf_sNB import dcpf_sNB

import matplotlib.pyplot as plt

#%% Generating data
K = 10
U = 1000
I = 2000
alpha = 1.
p = 0.8

W = np.random.gamma(alpha,.1,(U,K))
H = np.random.gamma(alpha,.1,(I,K))
L = W.dot(H.T)
Ya = np.random.negative_binomial(L/(-np.log(1-p)),1-p)
Y = sparse.csr_matrix(Ya)

#%% FIT DCPF model

model_to_fit = 'dcpf_sNB'

if model_to_fit == 'dcpf_Log':
    p_init = .5
    model = dcpf_Log(K=K, p=p_init, alphaW=alpha,alphaH=alpha)
    model.fit(Y, opt_hyper=['beta','p'], seed=10928,
             save=False, verbose=False)
    print 'Learnt parameter p='+str(p)
    
elif model_to_fit == 'dcpf_ZTP':
    p_init = 1.
    model = dcpf_ZTP(K=K, p=p_init, alphaW=alpha,alphaH=alpha)
    model.fit(Y, opt_hyper=['beta','p'], seed=10928,
             save=False, verbose=False)
    print 'Learnt parameter p='+str(p)

elif model_to_fit == 'dcpf_Geo':
    p_init = .5
    model = dcpf_Geo(K=K, p=p_init, alphaW=alpha,alphaH=alpha)
    model.fit(Y, opt_hyper=['beta','p'], seed=10928,
             save=False, verbose=False)
    print 'Learnt parameter p='+str(p)

elif model_to_fit == 'dcpf_sNB':
    p_init = .5
    model = dcpf_sNB(K=K, p=p_init, a=1., alphaW=alpha,alphaH=alpha)
    model.fit(Y, opt_hyper=['beta','p'], seed=10928,
             save=False, verbose=False)
    print 'Learnt parameter p='+str(p)

#%%
    
Ew = model.Ew
Eh = model.Eh
Yr = Ew.dot(Eh.T)

plt.figure('Observation')
plt.imshow(Ya,interpolation='nearest')
plt.colorbar()
plt.figure('Truth')
plt.imshow(L,interpolation='nearest')
plt.colorbar()
plt.figure('Reconstruction')
plt.imshow(Yr,interpolation='nearest')
plt.colorbar()