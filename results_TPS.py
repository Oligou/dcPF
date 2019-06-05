#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ogouvert
"""

import os
import glob
import pandas as pd
import cPickle as pickle
import numpy as np

from function.train_test import divide_train_test

from model.dcpf_Log import dcpf_Log
from model.dcpf_ZTP import dcpf_ZTP
from model.dcpf_Geo import dcpf_Geo
from model.dcpf_sNB import dcpf_sNB

from function import rec_eval

#%%
folder_name = 'tps' 
path = 'out/'+ folder_name

#%% DATA
seed_test = 1992
prop_test = 0.2
# Pre-processed data 
with open('data/tps/tps_12345_U1.63e+04_I1.21e+04_min_uc20_sc20', 'rb') as f:
    out = pickle.load(f)
    Y = out['Y_listen']

Y_train,Y_test = divide_train_test(Y,prop_test=prop_test,seed=seed_test)

#%% Calculate scores
for filename in glob.glob(os.path.join(path,'*')):  
    with open(filename,'rb') as f:
        model = pickle.load(f)
        W = model.Ew
        H = model.Eh
    #model.score={} - erase the score
    for s in [0,1,2,5]:
        if ~np.isin('ndcg@100s'+str(s), model.score.keys()):
            ndcg = rec_eval.normalized_dcg_at_k(Y_train>0,Y_test>s,W,H,k=100)
            model.score['ndcg@100s'+str(s)]=ndcg
    for k in [100]:
        if ~np.isin('prec_at_'+str(k), model.score.keys()):
            prec = rec_eval.prec_at_k(Y_train>0,Y_test>0,W,H,k=k)
            model.score['prec_at_'+str(k)]=prec
    for k in [100]:
        if ~np.isin('recall_at_'+str(k), model.score.keys()):
            recall = rec_eval.recall_at_k(Y_train>0,Y_test>0,W,H,k=k)
            model.score['recall_at_'+str(k)]=recall
    model.save_dir = path
    model.save_model()

#%% Read scores
appended_data = []
for filename in glob.glob(os.path.join(path,'*')):  
    with open(filename,'rb') as f:
        model = pickle.load(f)
    df_name = pd.DataFrame.from_dict([{'filename':filename, 'classname':model.classname}])
    df_init = pd.DataFrame.from_dict([model.saved_args_init])
    df_fit = pd.DataFrame.from_dict([model.saved_args_fit])
    df_score = pd.DataFrame.from_dict([model.score])
    df_loc = pd.concat([df_name,df_init,df_fit,df_score], axis=1)
    appended_data.append(df_loc)
    
if appended_data!=[]:
    df = pd.concat(appended_data, axis=0)