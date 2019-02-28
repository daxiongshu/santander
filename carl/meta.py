from ml_robot import search_ng
import pandas as pd
from time import time
from ml_robot import write_log
import numpy as np
from santander_model import YCOL,IDCOL,METRIC,get_meta,get_meta_paths
from ml_robot.metrics import get_score
import os
import glob

def ranknorm(X):
    for c in range(X.shape[1]):
        X[:,c] = X[:,c].argsort().argsort()*1.0/X.shape[0]
    return X

def meta():
    
    start = time()
    #xx = list(range(10,14))#+[222,223,225,227,231]
    xx = [30,31]
    paths = get_meta_paths(xx)

    y = pd.read_pickle('cache/train_y.pkl')[YCOL].values

    X,s = get_meta(paths,'cv')
    #X = ranknorm(X)
    #for i in range(X.shape[1]):
    #    X[:,i] = X[:,i].argsort().argsort()*1.0/X.shape[0]
    model = search_ng(feval=METRIC,silent=False,lr=0.01,iters=20,ave=True,maximize=True)
    model.fit(X,y)
    #model.fit2(X,y)
    #model.bw = np.array([[1,1,2]])
    yp = model.predict(X)
    score = get_score(y,yp,METRIC)
    print(score)


    s[YCOL] = yp 
    s.to_csv('cv_meta.csv.gz',index=False,compression='gzip')#,float_format='%.4f')

    Xt,s = get_meta(paths,'sub')
    #Xt = ranknorm(Xt)
    yp = model.predict(Xt)

     
    print(score,'meta search ng')
    s[YCOL] = yp 
    s.to_csv('sub_meta.csv.gz',index=False,compression='gzip')#,float_format='%.4f')
    duration = time()-start
    write_log(duration,score,'meta search ng %s'%(str(xx)),mfiles=['cv_meta.csv.gz','sub_meta.csv.gz'])

if __name__ == '__main__':
    meta()
