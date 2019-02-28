from santander_model import build,METRIC,run_cv_sub,preprocess
import os
import pandas as pd
from ml_robot import timer
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
FOLDS = 4

def bw(model_name):
    bad = []
    mode = 'sub'
    leak = False
    bl = [1]
    cl = [1]*len(bl)
    model_name = 'nn'
    X,Xt,y,names,tr_id,te_id = build(mode,build_list=bl,cache=cl)
    X,Xt = preprocess(X,Xt,names)
    scores,score,yp,yps,model = run_cv_sub(X,y,FOLDS,names,tr_id,model_name=model_name,Xt=Xt,leak=leak)
    best = score
    for i in range(X.shape[1]):
        bf = None
        for col in names:
            if col in bad:# or col.startswith('emb_')==1:
                continue
            X_tmp,Xt_tmp,names_tmp = drop(X,Xt,names,bad+[col])
            scores,score,yp,yps,model = run_cv_sub(X_tmp,y,FOLDS,names_tmp,tr_id,model_name=model_name,Xt=Xt_tmp,leak=leak)
            print(bad+[col],score,best)
            bslog(bad+[col],score)
            if best<score:
                best = score
                bf = col
        if bf is None:
            break
        bad.append(bf)

def drop(X,Xt,names,bad):
    names_tmp = [i for i in names if i not in bad]
    cols = [c for c,i in enumerate(names) if i not in bad]
    assert len(names_tmp) == len(cols)
    return X[:,cols],Xt[:,cols],names_tmp

def bslog(bad,score,name='bs.log'):
    if os.path.exists(name)==0:
        fo = open(name,'w')
        tag = 'bad' if name == 'bs.log' else 'good'
        fo.write('%s,score\n'%tag)
        fo.close()
    fo = open(name,'a')
    fo.write('"%s",%.4f\n'%(str(bad),score))
    fo.close()

def readbs(name='bs.log'):
    df = pd.read_csv(name)
    df = df.sort_values(by='score',ascending=False)
    print(df[['bad','score']].head())

if __name__ == '__main__':
    bw('xgb')
