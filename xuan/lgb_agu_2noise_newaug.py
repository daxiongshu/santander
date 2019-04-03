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




###################application##########################################

    

train = pd.read_csv('../input/train.csv')

train['real'] = 1

test = pd.read_csv('../input/test.csv')
print("Train samples: {}, test samples: {}".format(len(train), len(test)))
columns = [col for col in train.columns if col not in ['target','ID_code','real']]

for col in columns:
    test[col] = test[col].map(test[col].value_counts())
a = test[columns].min(axis=1)

test = pd.read_csv('../input/test.csv')
test['real'] = (a == 1).astype('int')

train = train.append(test).reset_index(drop=True)
del test;gc.collect()

columns = [col for col in train.columns if col not in ['target','ID_code']]

for col in train.columns:
    if col not in ['target','ID_code','real']:
#        train[col+'size'] = train.groupby(col)['target'].transform('size')
        train[col+'size'] = train[col].map(train.loc[train.real==1,col].value_counts())

# augment function, faster version 
def shuffle_col_vals(x1):
    rand_x = np.array([np.random.choice(x1.shape[0], size=x1.shape[0], replace=False) for i in range(x1.shape[1])]).T
    grid = np.indices(x1.shape)
    rand_y = grid[1]
    return x1[(rand_x, rand_y)]

def augment_fast1(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        x1 = shuffle_col_vals(x1)
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        x1 = shuffle_col_vals(x1)
        xn.append(x1)

    xs = np.vstack(xs); xn = np.vstack(xn)
    ys = np.ones(xs.shape[0]);yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn]); y = np.concatenate([y,ys,yn])
    return x,y

def shuffle_col_vals_fix(x1, groups):
    group_size = x1.shape[1]//groups
    xs = [x1[:, i*group_size:(i+1)*group_size] for i in range(groups)]
    rand_x = np.array([np.random.choice(x1.shape[0], size=x1.shape[0], replace=False) for i in range(group_size)]).T
    grid = np.indices(xs[0].shape)
    rand_y = grid[1]
    res = [x[(rand_x, rand_y)] for x in xs]
    return np.hstack(res)

def augment_fix_fast(x,y,groups,t1=2, t0=2):
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
    x = np.vstack([x,xs,xn]); y = np.concatenate([y,ys,yn])
    return x,y


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

# =============================================================================
# for col in columns:
#     if col not in ['target','ID_code','real']:
# #        train[col+'size'] = train.groupby(col)['target'].transform('size')
#         train[col+'encode'+str(50)] = get_feat_CVR_kflold(train,col,'target',5,2019,50,stratified = True,prefix='')
# =============================================================================

for col in columns:
    if col not in ['target','ID_code','real']:
#        train[col+'size'] = train.groupby(col)['target'].transform('size')
        train.loc[train[col+'size']>1,col+'no_noise'] = train.loc[train[col+'size']>1,col]

for col in columns:
    if col not in ['target','ID_code','real']:
#        train[col+'size'] = train.groupby(col)['target'].transform('size')
        train.loc[train[col+'size']>2,col+'no_noise2'] = train.loc[train[col+'size']>2,col]
#for col in columns:
##    print(col)
#    train[col+'last'] = train[col].apply(lambda x:str(x*10000)[-2:]).astype('int')



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
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = predictors
     
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['target'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df[target].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df[target].iloc[valid_idx]
        #train_x, train_y = augment(train_x.values, train_y.values)
        train_x, train_y = augment_fix_fast(train_x.values, train_y.values, groups=4, t1=16, t0=4)
        print(train_y.mean())
        xgtrain = lgb.Dataset(train_x, label=train_y,
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



        oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        sub_preds += clf.predict(test_df[feats], num_iteration=clf.best_iteration)/ folds.n_splits


        gain = clf.feature_importance('gain')
        fold_importance_df = pd.DataFrame({'feature':clf.feature_name(),
                                           'split':clf.feature_importance('split'),
                                           'gain':100*gain/gain.sum(),
                                           'fold':n_fold,                        
                                           }).sort_values('gain',ascending=False)
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
#        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
        
    print('Full AUC score %.6f' % roc_auc_score(train_df['target'], oof_preds))
    # Write submission file and plot feature importance
    score = roc_auc_score(train_df['target'], oof_preds)
    train_df['predict'] = oof_preds
    train_df[['ID_code', 'predict']].to_csv('../res/'+'val_{}.csv'.format(score), index= False)
    test_df[target] = sub_preds
    test_df[['ID_code', target]].to_csv('../res/'+'sub_{}.csv'.format(score), index= False)
    display_importances(feature_importance_df,score)



def display_importances(feature_importance_df_,score):
    ft = feature_importance_df_[["feature", "split","gain"]].groupby("feature").mean().sort_values(by="split", ascending=False)
    print(ft.head(60))
    ft.to_csv('../tiaotz/importance_lightgbm_{}.csv'.format(score),index=True)
    cols = ft[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="split", y="feature", data=best_features.sort_values(by="split", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
#    plt.savefig('lgbm_importances_{}.png'.format(score))

####################################计算#################################################################
#params = {'num_leaves': 31,
#         'min_data_in_leaf': 30, 
#         'max_depth': 6,
#         'learning_rate': 0.1,
#         "feature_fraction": 0.5,
#         "bagging_freq": 1,
#         "bagging_fraction": 0.9 ,
#         "bagging_seed": 11,
#         "lambda_l1": 0.1,
#         "verbosity": -1,
#         "random_state": 2333}

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
    "min_sum_heassian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
#    "bagging_seed" : random_state,
    "verbosity" : 1,
#    'scale_pos_weight':2
#    "seed": random_state
}



no_use= [
         ]

#imp = pd.read_csv('../tiaotz/importance_lightgbm_0.9020662919408432.csv')  
#no_use = no_use+list(imp[imp.ratio>4]['feature'])  

target = 'target'    
no_use_col = ['target','ID_code','real']+no_use
feats = [f for f in train.columns if f not in no_use_col]


categorical_columns = [col for col in feats if train[col].dtype == 'object']

for feature in categorical_columns:
    print(f'Transforming {feature}...')
    encoder = LabelEncoder()    
    train[feature] = encoder.fit_transform(train[feature].astype(str))  


#train.loc[train.target.notnull(),'target'] = shuffle(train.loc[train.target.notnull(),'target']).values


#train = reduce_mem_usage(train)
clf = kfold_lightgbm(params,train,feats,'target',5,
                     num_boost_round=100000,early_stopping_rounds=200,
                     categorical_features=categorical_columns, 
                     verbose_eval=1000,)


