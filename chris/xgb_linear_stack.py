import numpy as np
import pandas as pd

## Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

## Modelling
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge,ElasticNet, SGDRegressor, LogisticRegression
from sklearn.linear_model import RidgeClassifier,SGDClassifier, SGDRegressor, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from scipy import sparse
from scipy.stats import norm, skew
## Others
import os
import time
import warnings
import gc
import os
import pickle
from six.moves import urllib
import warnings
import copy
warnings.filterwarnings('ignore')

KFOLDS = 5
RANDOM_STATE = 42

def xgb_binary_stack(stack_params, train_x, train_y, test_x, kfolds, stratified=False,  random_state=42,
                     early_stopping_rounds=0, missing=None, full_vars=None, cat_vars=None, y_dummy=None, verbose=False):
    
    if stratified:
        kf = StratifiedKFold(n_splits=kfolds, shuffle=True,
                             random_state=random_state)
        kf_ids = list(kf.split(train_x, y_dummy))
    else:
        kf = KFold(n_splits=kfolds, random_state=random_state)
        kf_ids = list(kf.split(train_y))

    train_blend_x = np.zeros((train_x.shape[0], len(stack_params)))
    test_blend_x = np.zeros((test_x.shape[0], len(stack_params)))
    blend_scores = np.zeros((kfolds, len(stack_params)))

    test_dataset = xgb.DMatrix(test_x)

    if verbose:
        print("Start stacking.")
    for j, params in enumerate(stack_params):
        num_boost_round = copy.deepcopy(params.get('num_boost_round', 1000000))
        if verbose:
            print("Stacking model", j+1, params)
        test_blend_x_j = np.zeros((test_x.shape[0]))
        for i, (train_ids, val_ids) in enumerate(kf_ids):
            start = time.time()
            if verbose:
                print("Model %d fold %d" % (j+1, i+1))
            train_x_fold = train_x[train_ids]
            train_y_fold = train_y[train_ids]
            val_x_fold = train_x[val_ids]
            val_y_fold = train_y[val_ids]
            print(i, params)

            train_dataset = xgb.DMatrix(train_x_fold,
                                        train_y_fold
                                        )
            valid_dataset = xgb.DMatrix(val_x_fold,
                                        val_y_fold
                                        )
            watchlist = [(train_dataset, 'train'), (valid_dataset, 'valid')]

            if early_stopping_rounds == 0:
                model = xgb.train(params,
                                  train_dataset,
                                  num_boost_round=num_boost_round,
                                  verbose_eval=verbose
                                  )
                val_y_predict_fold = model.predict(valid_dataset)
                score = roc_auc_score(val_y_fold, val_y_predict_fold)
                print("Score for Model %d fold %d: %f " % (j+1, i+1, score))
                blend_scores[i, j] = score
                train_blend_x[val_ids, j] = val_y_predict_fold
                test_blend_x_j = test_blend_x_j + model.predict(test_dataset)
                if verbose:
                    print("Model %d fold %d finished in %d seconds." %
                          (j+1, i+1, time.time()-start))
            else:
                model = xgb.train(params,
                                  train_dataset,
                                  evals=watchlist,
                                  num_boost_round=num_boost_round,
                                  early_stopping_rounds=early_stopping_rounds,
                                  verbose_eval=verbose
                                  )
                best_iteration = model.best_iteration + 50
#                 print(model.best_score['valid']['rmse'])
                if params.get('booster','gbtree')=='gblinear':
                    val_y_predict_fold = model.predict(valid_dataset)
                else:
                    val_y_predict_fold = model.predict(
                            valid_dataset, ntree_limit=best_iteration)
                score = roc_auc_score(val_y_fold, val_y_predict_fold)
                if verbose:
                    print("Score for Model %d fold %d: %f " %
                          (j+1, i+1, score))
                blend_scores[i, j] = score
                train_blend_x[val_ids, j] = val_y_predict_fold
                if params.get('booster','gbtree')=='gblinear':
                    test_blend_x_j = test_blend_x_j + \
                        model.predict(test_dataset)
                else:
                    test_blend_x_j = test_blend_x_j + \
                        model.predict(test_dataset, ntree_limit=best_iteration)
                if verbose:
                    print("Model %d fold %d finished in %d seconds." %
                          (j+1, i+1, time.time()-start))

        test_blend_x[:, j] = test_blend_x_j/kfolds
        print("Score for model %d is %f" % (j+1, np.mean(blend_scores[:, j])))
    return train_blend_x, test_blend_x, blend_scores


def main():
    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')
    test_id = test_df['ID_code'].values
    train_y = train_df['target'].values

    cat_vars = []
    num_vars = [] 
    for v, d in zip(train_df.columns.values, train_df.dtypes.values):
        if 'float' in str(d) or 'int' in str(d):
            num_vars.append(v)
        if 'object' in str(d):
            cat_vars.append(v)            
            
    cat_vars.remove('ID_code')
    num_vars.remove('target')
            
    print ("Categorical variables:", cat_vars)
    print ("Numeric variables:", num_vars)

    ID_var = 'ID_code'
    target_var = 'target'
    test_id = test_df[ID_var].values

    full_vars = copy.copy(num_vars)
    processor = StandardScaler()
    processor.fit(train_df[full_vars])
    train_tran_df = pd.DataFrame(processor.transform(train_df[full_vars]), columns=full_vars)
    train_y = train_df[target_var].values
    train_dummy_y = train_df[target_var].values
    test_tran_df = pd.DataFrame(processor.transform(test_df[full_vars]), columns=full_vars)

    train_tran_df['target'] = train_df['target']

    train_tran_df['p1_mean'] = train_tran_df[num_vars].apply(lambda x:np.mean(x), axis=1)
    test_tran_df['p1_mean'] = test_tran_df[num_vars].apply(lambda x:np.mean(x), axis=1)


    p2_num_vars = []
    for v in num_vars:
        train_tran_df[v +'p2'] = train_tran_df[v] **2
        test_tran_df[v +'p2'] = test_tran_df[v] **2
        p2_num_vars.append(v +'p2')

    train_tran_df['p2_sqrt'] = train_tran_df[p2_num_vars].apply(lambda x:np.sqrt(np.sum(x)), axis=1)
    test_tran_df['p2_sqrt'] = test_tran_df[p2_num_vars].apply(lambda x:np.sqrt(np.sum(x)), axis=1)

    train_tran_df['p1_sum'] = train_tran_df[num_vars].apply(lambda x:sum(x), axis=1)
    test_tran_df['p1_sum'] = test_tran_df[num_vars].apply(lambda x:sum(x), axis=1)


    train_tran_df['p2_sqrt_mean'] = train_tran_df[p2_num_vars].apply(lambda x:np.sqrt(np.mean(x)), axis=1)
    test_tran_df['p2_sqrt_mean'] = test_tran_df[p2_num_vars].apply(lambda x:np.sqrt(np.mean(x)), axis=1)


    p3_num_vars = []
    for v in num_vars:
        train_tran_df[v +'p3'] = train_tran_df[v] **3
        test_tran_df[v +'p3'] = test_tran_df[v] **3
        p3_num_vars.append(v +'p3')

    train_tran_df['p3_sum'] = train_tran_df[p3_num_vars].apply(lambda x:np.sum(x), axis=1)
    test_tran_df['p3_sum'] = test_tran_df[p3_num_vars].apply(lambda x:np.sum(x), axis=1)

    train_tran_df['p3_sqrt_mean'] = train_tran_df[p3_num_vars].apply(lambda x:np.mean(x)**(1/3), axis=1)
    test_tran_df['p3_sqrt_mean'] = test_tran_df[p3_num_vars].apply(lambda x:np.mean(x)**(1/3), axis=1)           

    p4_num_vars = []
    for v in num_vars:
        train_tran_df[v +'p4'] = train_tran_df[v] **4
        test_tran_df[v +'p4'] = test_tran_df[v] **4
        p4_num_vars.append(v +'p4')

        
    train_tran_df['p4_sum'] = train_tran_df[p4_num_vars].apply(lambda x:np.sum(x), axis=1)
    test_tran_df['p4_sum'] = test_tran_df[p4_num_vars].apply(lambda x:np.sum(x), axis=1)

    train_tran_df['p4_sqrt_mean'] = train_tran_df[p4_num_vars].apply(lambda x:np.mean(x)**(1/4), axis=1)
    test_tran_df['p4_sqrt_mean'] = test_tran_df[p4_num_vars].apply(lambda x:np.mean(x)**(1/4), axis=1)  

    full_vars = num_vars + p2_num_vars + p3_num_vars + p4_num_vars +\
        ['p2_sqrt','p1_sum', 'p2_sqrt_mean', 'p1_mean', 'p3_sum', 'p3_sqrt_mean', 'p4_sum', 'p4_sqrt_mean']
    train_x =train_tran_df[full_vars].values
    train_y = train_tran_df[target_var].values
    train_dummy_y = train_tran_df[target_var].values

    test_x = test_tran_df[full_vars].values

    print(train_x.shape, test_x.shape)

    xgb_stack_params = [{'ective': 'binary:logistic', 
                     'booster':'gblinear',
                     'num_boost_round':3000,
                     'eta': 0.1, 'eval_metric':'auc', 
                     'lambda': 0, 
                     'alpha': 0, 
                     'feature_selector': 'cyclic',
                     'seed': 1234, 'nthread': -1}]

    train_blend_x_xgb, test_blend_x_xgb, blend_scores_xgb = \
            xgb_binary_stack(xgb_stack_params, train_x, train_y, test_x, 5, early_stopping_rounds=200, 
                            stratified=True, random_state=42,
                        full_vars=full_vars, cat_vars=cat_vars,y_dummy=train_dummy_y, verbose=500)


    stack_score = roc_auc_score(train_y, train_blend_x_xgb.mean(axis=1))
    print ('Score: %f ' % (stack_score))
   
    stack_score = str(int(stack_score * 10000))
    time_str= datetime.now().strftime("%Y%m%d%H%M")
    pickle.dump(train_blend_x_xgb, open('../stacking/'+stack_score + '_' + time_str+'_train_blend_x_xgb.pkl', 'wb'))
    pickle.dump(test_blend_x_xgb, open('../stacking/'+stack_score + '_' + time_str+'_test_blend_x_xgb.pkl', 'wb'))
    print(stack_score, time_str)
    sub = pd.DataFrame({'ID_code':test_id,'target':test_blend_x_xgb.mean(axis=1)})
    sub.to_csv('../output/%s_%s.csv' % (stack_score, time_str),index=False)

if __name__ == '__main__':
    main()
