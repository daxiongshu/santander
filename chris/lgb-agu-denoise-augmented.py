#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 20:39:58 2018

@author: hzs
"""

import numpy as np
import pandas as pd
import gc

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve,mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.simplefilter(action='ignore', category=FutureWarning)


from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.utils import shuffle
import datetime

cache_path = '../cache/'
seed = 2019

import time
import os
import random
import multiprocessing
import sys
import pickle

def CVR_helper(train_base,data,base_feat,target,window_size):
    '''转化率计算函数'''

    i = data.shape[0]//window_size
    data = data.sort_values(base_feat) 
    data[base_feat+'_'+str(window_size)] = data[base_feat].rolling(i,min_periods=1).mean()
    
    col = 'mean_y'
    df = data.groupby(base_feat).agg({base_feat+'_'+str(window_size):'mean'})
    df.columns = [col]
    df = df.reset_index()
    
    
    result = train_base.merge(df,on=base_feat,how='left')
    return result['mean_y'].values


def get_feat_CVR_kflold(train,base_feat,target,num_folds,seed,window_size,stratified = False,prefix=''):
    train = train[[base_feat]+[target]]
    train_df = train[train[target].notnull()]
    test_df = train[train[target].isnull()]
    folds = KFold(n_splits= num_folds, shuffle=True, random_state=seed)
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=seed)

    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df[target])):
        train_x = train_df.iloc[train_idx]
        valid_x = train_df.iloc[valid_idx]
        oof_preds[valid_idx] = CVR_helper(valid_x,train_x,base_feat,target,window_size)
     
    sub_preds = CVR_helper(test_df,train_df,base_feat,target,window_size)
    train[base_feat +"_CVR" + str(window_size)] = np.hstack((oof_preds,sub_preds))
    result = train[[base_feat +"_CVR"+str(window_size)]]
    return result.values

def shuffle_(x):
    ids,x1 = x
    np.random.shuffle(ids)
    x1 = x1[ids]
    return x1

def parallel_run(func,data,silent=False,mc=-1):
    mr = multiprocessing.cpu_count()
    if mc==-1 or mc>mr:
        mc = mr
    if silent==0:
        print("using %d cpus"%mc)
    p = multiprocessing.Pool(mc)
    results = p.imap(func, data)
    num_tasks = len(data)
    while (True):
        completed = results._index
        if silent==0:
            print("\r--- parallel {} completed {:,} out of {:,}".format(func.__name__,completed, num_tasks),end="")
        sys.stdout.flush()
        time.sleep(1)
        if (completed == num_tasks):
            break
    p.close()
    p.join()
    if silent==0:
        print()
    return list(results)    

def get_groups_from_cols(cols):
    group = {}
    for c,col in enumerate(cols):
        var = '_'.join(col.split('_')[:2])
        if var not in group:
            group[var] = []
        group[var].append(c)
    return [v for k,v in group.items()]

def augment_fast_pd(x,y,cols,t=2,include_raw=True):
    xs,xn = [],[]
    groups = get_groups_from_cols(cols)
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        inputs = [[ids,x1[:,g]] for g in groups]
        outputs = parallel_run(shuffle_,inputs,mc=16)
        for c,g in enumerate(groups):
            x1[:,g] = outputs[c]
        xs.append(x1)

    for i in range(t//1):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        inputs = [[ids,x1[:,g]] for g in groups]
        outputs = parallel_run(shuffle_,inputs,mc=16)
        for c,g in enumerate(groups):
            x1[:,g] = outputs[c]
        xn.append(x1)
    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    if include_raw:
        x = np.vstack([x,xs,xn])
        y = np.concatenate([y,ys,yn])
    else:
        x = np.vstack([xs,xn])
        y = np.concatenate([ys,yn])
    return x,y      

def augment(x,y,t=16):
        xs,xn = [],[]
        for i in range(t):
            mask = y>0
            x1 = x[mask].copy()
            ids = np.arange(x1.shape[0])
            for c in range(x1.shape[1]//4):
                np.random.shuffle(ids)
                x1[:,c] = x1[ids][:,c]
                x1[:,c+200] = x1[ids][:,c+200]
                x1[:,c+400] = x1[ids][:,c+400]
                x1[:,c+600] = x1[ids][:,c+600]
            xs.append(x1)
    
        for i in range(t//4):
            mask = y==0
            x1 = x[mask].copy()
            ids = np.arange(x1.shape[0])
            for c in range(x1.shape[1]//4):
                np.random.shuffle(ids)
                x1[:,c] = x1[ids][:,c]
                x1[:,c+200] = x1[ids][:,c+200]
                x1[:,c+400] = x1[ids][:,c+400]
                x1[:,c+600] = x1[ids][:,c+600]
            xn.append(x1)
        xs = np.vstack(xs)
        xn = np.vstack(xn)
        ys = np.ones(xs.shape[0])
        yn = np.zeros(xn.shape[0])
        x = np.vstack([x,xs,xn])
        y = np.concatenate([y,ys,yn])
        return x,y

def shuffle_col_vals_fix(x1, groups):
    group_size = x1.shape[1]//groups
    xs = [x1[:, i*group_size:(i+1)*group_size] for i in range(groups)]
    rand_x = np.array([np.random.choice(x1.shape[0], size=x1.shape[0], replace=False) for i in range(group_size)]).T
    grid = np.indices(xs[0].shape)
    rand_y = grid[1]
    res = [x[(rand_x, rand_y)] for x in xs]
    return np.hstack(res)


def augment_fix_fast(x,y,groups,t1=2, t0=2, include_raw=True):
    # In order to make the sync version augment work, the df should be the form of:
    # var_1, var_2, var_3 | var_1_count, var_2_count, var_3_count | var_1_rolling, var_2_rolling, var_3_rolling
    # for the example above, 3 groups of feature, groups = 3
    xs,xn = [],[]
    for i in range(t1):
        mask = y>0
        x1 = x[mask].copy()
        x1 = shuffle_col_vals_fix(x1, groups)
        xs.append(x1)

    for i in range(t0):
        mask = (y==0)
        x1 = x[mask].copy()
        x1 = shuffle_col_vals_fix(x1, groups)
        xn.append(x1)

    xs = np.vstack(xs); xn = np.vstack(xn)
    ys = np.ones(xs.shape[0]);yn = np.zeros(xn.shape[0])
    if include_raw:
        x = np.vstack([x,xs,xn]); y = np.concatenate([y,ys,yn])
    else:
        x = np.vstack([xs,xn]); y = np.concatenate([ys,yn])
    return x,y        

# =============================================================================

def kfold_lightgbm(params,df, predictors,target,num_folds, stratified = True,
                   objective='binary', metrics='auc',debug= False,
                   feval=None, early_stopping_rounds=100, num_boost_round=1000, verbose_eval=50, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
    }

    lgb_params.update(params)
    
    train_df = df[df['target'].notnull()]
    test_df = df[df['target'].isnull()]
    
    # Divide in training/validation and test data
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=seed)
    # Create arrays and dataframes to store results
    
    # raw data
    oof_raw_preds = np.zeros(train_df.shape[0])
    sub_raw_preds = np.zeros(test_df.shape[0])

    # Denoised raw data
    oof_denoise_preds = np.zeros(train_df.shape[0])
    sub_denoise_preds = np.zeros(test_df.shape[0])

    # Raw + augmented data
    oof_aug_preds = np.zeros(train_df.shape[0])
    sub_aug_preds = np.zeros(test_df.shape[0])

    # Denoised augmented data
    oof_dn_aug_preds = np.zeros(train_df.shape[0])
    sub_dn_aug_preds = np.zeros(test_df.shape[0])

    #Raw + denoised augmented data
    oof_raw_dn_aug_preds = np.zeros(train_df.shape[0])
    sub_raw_dn_aug_preds = np.zeros(test_df.shape[0])

    # Blended 
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
 

    # feature_importance_df = pd.DataFrame()
    feats = predictors
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['target'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df[target].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df[target].iloc[valid_idx]

        # Raw data
        print("Raw data shape:", train_x.shape, valid_x.shape)
        print("Training on raw data....")

        xgtrain = lgb.Dataset(train_x.values, label=train_y,
                              feature_name=predictors,
                              categorical_feature=categorical_features,
                              )
        xgvalid = lgb.Dataset(valid_x.values, label=valid_y.values,
                              feature_name=predictors,
                              categorical_feature=categorical_features
                              )

        raw_clf = lgb.train(lgb_params, 
                         xgtrain, 
                         valid_sets=[xgtrain, xgvalid], 
                         valid_names=['train','valid'], 
                         num_boost_round=9000,
                        #  early_stopping_rounds=early_stopping_rounds,
                         verbose_eval=verbose_eval, 
                         feval=feval)

        oof_raw_preds[valid_idx] = raw_clf.predict(valid_x, num_iteration=raw_clf.best_iteration)
        sub_raw_preds += raw_clf.predict(test_df[feats], num_iteration=raw_clf.best_iteration)/ folds.n_splits

        del xgtrain, xgvalid
        gc.collect()
        # # Denoised raw data
        # print("Denoising raw data....")
        # raw_cutoff = 10
        # train_raw_prob = raw_clf.predict(train_x.values)
        # raw_threshold = np.percentile(train_raw_prob, raw_cutoff)
        # print(train_raw_prob, raw_threshold)

        # train_denoise_x = train_x.values[train_raw_prob>raw_threshold,:]
        # train_denoise_y = train_y.values[train_raw_prob>raw_threshold]
        # print("Denoised raw data shape:", train_denoise_x.shape)
        # print("Training on denoised raw data....")
        # xgtrain = lgb.Dataset(train_denoise_x, label=train_denoise_y,
        #                     feature_name=predictors,
        #                     categorical_feature=categorical_features
        #                     )

        # xgvalid = lgb.Dataset(valid_x.values, label=valid_y.values,
        #                     feature_name=predictors,
        #                     categorical_feature=categorical_features
        #                     )

        # denoise_clf = lgb.train(lgb_params, 
        #                 xgtrain, 
        #                 valid_sets=[xgtrain, xgvalid], 
        #                 valid_names=['train','valid'], 
        #                 num_boost_round=num_boost_round,
        #                 early_stopping_rounds=early_stopping_rounds,
        #                 verbose_eval=verbose_eval, 
        #                 feval=feval)

        # oof_denoise_preds[valid_idx] = denoise_clf.predict(valid_x, num_iteration=denoise_clf.best_iteration)
        # sub_denoise_preds += denoise_clf.predict(test_df[feats], num_iteration=denoise_clf.best_iteration)/ folds.n_splits
        # del xgtrain, xgvalid, denoise_clf
        # gc.collect()

        # Raw + augmented data
        print("Augmenting....")
        augments=16
        cols = [i for i in train_x.columns]
        # train_aug_x, train_aug_y = augment_fast_pd(train_x.values, train_y.values, cols, 
        #     t=augments, include_raw=False)
        train_aug_x, train_aug_y = augment_fix_fast(train_x.values, train_y.values, 
                                            groups=4, t1=augments, t0=augments//4, 
                                            include_raw=False)

        # print("Training on raw data + augmented data:")
        # print("Augmented data shape:", train_aug_x.shape)
        # xgtrain = lgb.Dataset(np.vstack((train_x.values, train_aug_x)), 
        #                       label=np.hstack((train_y, train_aug_y)),
        #                       feature_name=predictors,
        #                       categorical_feature=categorical_features
        #                       )
                              
        # xgvalid = lgb.Dataset(valid_x.values, label=valid_y.values,
        #                       feature_name=predictors,
        #                       categorical_feature=categorical_features
        #                       )

        # clf = lgb.train(lgb_params, 
        #                  xgtrain, 
        #                  valid_sets=[xgtrain, xgvalid], 
        #                  valid_names=['train','valid'], 
        #                  num_boost_round=num_boost_round,
        #                  early_stopping_rounds=early_stopping_rounds,
        #                  verbose_eval=verbose_eval, 
        #                  feval=feval)

        # oof_aug_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        # sub_aug_preds += clf.predict(test_df[feats], num_iteration=clf.best_iteration)/ folds.n_splits                         

        # del xgtrain, xgvalid, clf
        # gc.collect()

        # Denoised augmented data
        aug_cutoff=25
        train_aug_prob = raw_clf.predict(train_aug_x)
        aug_threshold = np.percentile(train_aug_prob, aug_cutoff)

        train_aug_x = train_aug_x[train_aug_prob>aug_threshold,:]
        train_aug_y = train_aug_y[train_aug_prob>aug_threshold]

        print("Training on denoised augmented data:")
        print("Denoised augmented data shape:", train_aug_x.shape)
        xgtrain = lgb.Dataset(train_aug_x, 
                              label=train_aug_y,
                              feature_name=predictors,
                              categorical_feature=categorical_features
                              )
                              
        xgvalid = lgb.Dataset(valid_x.values, label=valid_y.values,
                              feature_name=predictors,
                              categorical_feature=categorical_features
                              )

        clf = lgb.train(lgb_params, 
                         xgtrain, 
                         valid_sets=[xgtrain, xgvalid], 
                         valid_names=['train','valid'], 
                         num_boost_round=num_boost_round,
                         early_stopping_rounds=early_stopping_rounds,
                         verbose_eval=verbose_eval, 
                         feval=feval)

        oof_dn_aug_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        sub_dn_aug_preds += clf.predict(test_df[feats], num_iteration=clf.best_iteration)/ folds.n_splits                         

        # del xgtrain, xgvalid, clf
        # gc.collect()

        # # Raw + Denoised augmented data
        # print("Training on raw data + denoised augmented data:")
        # print("Denoised augmented data shape:", train_aug_x.shape)
        # xgtrain = lgb.Dataset(np.vstack((train_x.values, train_aug_x)), 
        #                       label=np.hstack((train_y, train_aug_y)),
        #                       feature_name=predictors,
        #                       categorical_feature=categorical_features
        #                       )
                              
        # xgvalid = lgb.Dataset(valid_x.values, label=valid_y.values,
        #                       feature_name=predictors,
        #                       categorical_feature=categorical_features
        #                       )

        # clf = lgb.train(lgb_params, 
        #                  xgtrain, 
        #                  valid_sets=[xgtrain, xgvalid], 
        #                  valid_names=['train','valid'], 
        #                  num_boost_round=num_boost_round,
        #                  early_stopping_rounds=early_stopping_rounds,
        #                  verbose_eval=verbose_eval, 
        #                  feval=feval)

        # oof_raw_dn_aug_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        # sub_raw_dn_aug_preds += clf.predict(test_df[feats], num_iteration=clf.best_iteration)/ folds.n_splits                         

        # del xgtrain, xgvalid, clf
        # gc.collect()        

        # oof_preds[valid_idx] = (oof_raw_preds[valid_idx] 
        #                         + oof_denoise_preds[valid_idx] 
        #                         + oof_aug_preds[valid_idx]
        #                         + oof_raw_dn_aug_preds[valid_idx]
        #                         + oof_dn_aug_preds[valid_idx])/5
        # sub_preds += (sub_raw_preds 
        #                 + sub_denoise_preds
        #                 + sub_aug_preds
        #                 + sub_raw_dn_aug_preds
        #                 + sub_dn_aug_preds
        #                 )/5


        oof_preds[valid_idx] =  oof_dn_aug_preds[valid_idx]
        sub_preds += sub_dn_aug_preds
   
        print('Fold %2d AUC : %.6f' % (n_fold + 1, 
                        roc_auc_score(valid_y, oof_preds[valid_idx])))


    
    # Write submission file and plot feature importance
    score = roc_auc_score(train_df['target'], oof_preds)
    train_df['predict'] = oof_preds
    train_df[['ID_code', 'predict']].to_csv('../res/'+'val_{}.csv'.format(score), index= False)
    test_df[target] = sub_preds
    test_df[['ID_code', target]].to_csv('../res/'+'sub_{}.csv'.format(score), index= False)


def run():
    ###################application##########################################
    # train = pd.read_csv('../input/train.csv')
    # train['real'] = 1
    # test = pd.read_csv('../input/test.csv')
    # print("Train samples: {}, test samples: {}".format(len(train), len(test)))
    # columns = [col for col in train.columns if col not in ['target','ID_code','real']]

    # for col in columns:
    #     test[col] = test[col].map(test[col].value_counts())
    # a = test[columns].min(axis=1)

    # test = pd.read_csv('../input/test.csv')
    # test['real'] = (a == 1).astype('int')

    # train = train.append(test).reset_index(drop=True)
    # del test;gc.collect()

    # columns = [col for col in train.columns if col not in ['target','ID_code']]

    # for col in train.columns:
    #     if col not in ['target','ID_code','real']:
    #         train[col+'_size'] = train[col].map(train.loc[train.real==1,col].value_counts())

    # for col in columns:
    #     if col not in ['target','ID_code','real']:
    #         train.loc[train[col+'_size']>1,col+'_nonoise'] = train.loc[train[col+'_size']>1,col]
            
    # for col in columns:
    #     if col not in ['target','ID_code','real']:
    #         train.loc[train[col+'_size']>2,col+'_nonoise2'] = train.loc[train[col+'_size']>2,col]        


    train = pickle.load(open('../input/train.pkl', 'rb'))

    params = {
    #    "objective" : "binary",
    #    "metric" : "auc",
    #    "boosting": 'gbdt',
        "max_depth" : -1,
        "num_leaves" : 13,
        "learning_rate" : 0.01,
        "bagging_freq": 5,
        "bagging_fraction" : 0.4,
        "feature_fraction" : 0.05,
        "min_data_in_leaf": 80,
        # "min_sum_hessian_in_leaf": 10,
        "tree_learner": "serial",
        "boost_from_average": "false",
        "max_bin": 20000,
        #"lambda_l1" : 5,
        #"lambda_l2" : 5,
    #    "bagging_seed" : random_state,
        "verbosity" : 1,
    #    'scale_pos_weight':2
    #    "seed": random_state
    }

    # no_use= ['var_45_size',
    #             'var_117_size',
    #             'var_74_size',
    #             'var_61_size',
    #             'var_97_size',
    #             'var_166_no_noise',
    #             'var_71_no_noise',
    #             'var_43_no_noise',
    #             'var_148_no_noise',
    #             'var_161_no_noise',
    #             'var_103_no_noise',
    #             'var_91_no_noise',
    #             'var_12_no_noise',
    #             'var_108_no_noise',
    #             'var_68_no_noise']
    no_use = []
    target = 'target'    
    no_use_col = ['target','ID_code','real']+no_use
    feats = [f for f in train.columns if f not in no_use_col]


    categorical_columns = [col for col in feats if train[col].dtype == 'object']

    # pickle.dump(train,open('../input/train.pkl', 'wb'))
    kfold_lightgbm(params,train,feats,'target',5,num_boost_round=10000000, verbose_eval=1000,
    early_stopping_rounds=1000,categorical_features=categorical_columns)
    
if __name__ == '__main__':
    run()
