try:
    import cudf as gd
    import nvstrings
    from librmm_cffi import librmm
    from nvstring_workaround import get_unique_tokens,on_gpu,get_token_counts,is_in
    from cudf_workaround import unique,rename_col,to_pandas,merge
except:
    print('cudf not imported')

import glob
from scipy import sparse
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn.model_selection import train_test_split
import time
from ml_robot import timer,bag_xgb_model,xgb_model,lgb_model,bag_lgb_model,write_log,sk_lgb_model,bag_sk_lgb_model
from ml_robot.metrics import get_score
import os
import pandas as pd
import numpy as np
from collections import OrderedDict,Counter
import sys
import re
from sklearn.model_selection import KFold,StratifiedKFold
import pickle

FDIC = {}
FOLDS = 4
SAVE = False
WORK = '/raid/data/ml/santander'
CACHE = '%s/code/cache'%WORK
PATH = '%s/input'%WORK
RANKNORM = 1 
METRIC = 'auc'
IDCOL = 'ID_code'
YCOL = 'target'
NUM_CLASS=1
MODE = None
GPU = 7 
if len(sys.argv)==2 and sys.argv[0].endswith('.py'):
    GPU = int(sys.argv[1])
    print('Reset GPU',GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
print("GPU:",GPU)

@timer
def build(mode,build_list=[1],cache=[0]):
    assert len(build_list) == len(cache)
    mkdir(CACHE)
    y,tr_id,te_id = get_id_y(mode)
    if len(build_list)==0:
        return None,None,y,None,tr_id,te_id
    X,Xt,names = [],[],[]
    for i,s in zip(build_list,cache):
        if s and exist(i,mode):
            x,xt,cols = load(i,mode)
        else:
            x,xt,cols = eval('build%d(mode)'%i)
            if s:
                dump(i,mode,x,xt,cols)
            else:
                rm(i,mode)
        X.append(x)
        Xt.append(xt)
        names.append(cols)
    if True:
        if has_sparse(X):
           X = sparse.hstack(X,format='csr')
           Xt = sparse.hstack(Xt,format='csr')
        else:
           X = np.hstack(X)
           Xt = np.hstack(Xt)
    names = [i for cols in names for i in cols]
    return X,Xt,y,names,tr_id,te_id

def get_meta(paths,mode):
    df = []
    for i in paths:
        name = glob.glob("%s/%s*.csv.gz"%(i,mode))
        if len(name) == 0:
            name = glob.glob("%s/%s*.csv"%(i,mode))
        name = [j for j in name if 'post' not in j]
        name = name[-1:]
        print(name)
        assert len(name) == 1
        name = name[0]
        dt = pd.read_csv(name)
        df.append(dt)
    if mode == 'cv':
        X = []
        for i in df:
            if IDCOL in i.columns:
                i.drop(IDCOL,axis=1,inplace=True)
            if 'real' in i.columns:
                i.drop('real',axis=1,inplace=True)
            X.append(i.values)
    else:
        X = [i.drop(IDCOL,axis=1).values for i in df]
    return np.hstack(X),df[0]

def get_meta_paths(lines):
    paths = {}
    with open('run.log') as f:
        for c,line in enumerate(f):
            if c+1 in lines:
                ps = line.split(',')[:2]
                path = 'backup/%s_cv_%s'%(ps[0],ps[1])
                paths[c+1] = path
    paths = [paths[i] for i in lines]
    for i in paths:
        print(i)
    return paths

def rm(i,mode):
    out = '%s/fea%s_%d_tr.pkl'%(CACHE,mode,i)
    if os.path.exists(out):
        os.remove(out)
    out = out.replace('tr','te')
    if os.path.exists(out):
        os.remove(out)

def exist(i,mode):
    out = '%s/fea%s_%d_tr.pkl'%(CACHE,mode,i)
    return os.path.exists(out)

def load(i,mode):
    out = '%s/fea%s_%d_tr.pkl'%(CACHE,mode,i)
    x = pd.read_pickle(out)
    xt = pd.read_pickle(out.replace('tr','te'))
    cols = [i for i in x.columns]
    return x.values,xt.values,cols

def dump(i,mode,x,xt,cols):
    out = '%s/fea%s_%d_tr.pkl'%(CACHE,mode,i)
    pd.DataFrame(x,columns=cols).to_pickle(out)
    pd.DataFrame(xt,columns=cols).to_pickle(out.replace('tr','te'))

@timer
def build1(mode):
    tr_path,te_path = get_tr_te_paths(mode)
    gtr,_ = read_csv_hash_nvstring(tr_path)
    gte,_ = read_csv_hash_nvstring(te_path)
    badcols = []#['var_37']
    gtr = rm_cols(gtr,[IDCOL,YCOL]+badcols)
    gte = rm_cols(gte,[IDCOL,YCOL]+badcols)

    cols = ['var_12','var_81']
    #cols = ['var_20', 'var_21']
    #cols += ['var_7', 'var_8']
    #cols += ['var_10']
    #cols += ['var_70','var_74']
    #gtr,gte = gtr[cols],gte[cols]

    #gtr['var_12'] = gtr['var_12']*100
    print("build1",len(gtr),len(gte))
    return post_gdf(gtr,gte)

def build2(mode):
    name = '%s/xgb_loo_err.pkl'%(CACHE)
    if os.path.exists(name):
        x = pd.read_pickle(name)
    else:
        x = get_xgb_loo_err()
        x.to_pickle(name)

    N = x.shape[0]//2
    names = [i for i in x.columns]
    x = x.values
    s = pd.DataFrame()
    funcs = ['median']#,'std','min','max']
    for func in funcs:
        s[func] = eval('np.%s(x,axis=1)'%func)
    x = s.values
    x,xt = x[:N],x[N:]
    print(x[0])
    return x,xt,['loo_err_%s'%i for i in funcs]

def build3(mode):
    x,xt,names = build1('sub')
    N = x.shape[0]
    x = np.vstack([x,xt])
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    #x,xt = x[:N],x[N:]
    #return x,xt,names
    #x.sort(axis=1)
    #xt.sort(axis=1)
    #print(x[0,-10:])
    #M = 10
    #return x[:,-M:],xt[:,-M:],['max_%d'%i for i in range(M,0,-1)]
    for i in range(x.shape[1]):
        x[:,i] = x[:,i].argsort().argsort()
    funcs = ['std']
    s = pd.DataFrame()
    for func in funcs:
        s[func] = eval('np.%s(x,axis=1)'%func)
    x = s.values
    x,xt = x[:N],x[N:]
    print(x[0])
    return x,xt,['%s'%i for i in funcs]

def get_xgb_loo_err():
    x,xt,names = build1('sub')
    x = np.vstack([x,xt])
    feas = np.zeros_like(x).astype(np.float32)
    scaler = StandardScaler()
    x = scaler.fit_transform(x) 
    for i in range(x.shape[1]):
        cols = [j for j in range(x.shape[1]) if j!=i]
        model = get_xgb_loo_model(names)
        xm,ym = x[:,cols],x[:,i]
        model.fit(xm,ym,va=[xm,ym]) 
        yp = model.predict(xm)
        feas[:,i] = (ym-yp)**2#get_score(ym,yp,'rmse')
        print(feas[:10,i])
        del model.bst
        del model
        #break
    return pd.DataFrame(feas,columns=['loo_%s'%i for i in names])  

def get_xgb_loo_model(names):
    xgb_params = get_xgb_params(names[:-1],1)
    xgb_params.update({
        'objective': 'reg:linear',
        'num_round':100,
        'max_depth': 5,#5 if 'stack' not in MODE else 1,
        'maximize':False,
        'eval_metric':'rmse',
    })
    model = xgb_model(**xgb_params)
    return model

def build0(mode):
    lines = [14,19]
    paths = get_meta_paths(lines)        
    X,_ = get_meta(paths,'cv')
    Xt,_ = get_meta(paths,'sub')
    return X,Xt,['meta%d'%i for i in range(X.shape[1])]    

def has_sparse(X):
    for i in X:
        if isinstance(i,sparse.csr_matrix):
            return True
    return False

def post_gdf(gtr,gte):
    tr = gtr.to_pandas().values
    te = gte.to_pandas().values
    cols = [i for i in gtr.columns]
    del gtr,gte
    return tr,te,cols

@timer
def read_csv_hash_nvstring(path,cols=None):
    """read a csv file to a dictionary of nvstring objects
    Parameters
    ----------
    path : str, path of the csv file
    cols : list, a list of column names to be read

    Returns
    ----------
    str_cols : list of string column name 
    df: gd.dataframe
    """
    dic = get_dtype(path,1000)
    col_names = [i for i in dic]
    dtypes = [dic[i] for i in col_names]

    gd_cols = gd.io.csv.read_csv_strings(path,
                        names=col_names, dtype=dtypes,
                        skiprows=1)
    str_cols = []
    df = gd.DataFrame()
    for name,dtype,ds in zip(col_names,dtypes,gd_cols):
        if cols is not None and name not in cols:
            continue
        if dtype =='str':
            df[name] = on_gpu(ds,'hash')
            str_cols.append(name)
        else:
            df[name] = ds

    del gd_cols
    return df,str_cols


@timer
def read_csv_with_nvstring(path,cols=None,allstr=False):
    """read a csv file to a dictionary of nvstring objects
    Parameters
    ----------
    path : str, path of the csv file
    cols : list, a list of column names to be read

    Returns
    ----------
    str_cols : dictionary, column name => nvstring object
    df: gd.dataframe
    """
    dic = get_dtype(path,1000)
    col_names = [i for i in dic]
    dtypes = [dic[i] for i in col_names]
    if allstr:
        dtypes = ['str' for i in col_names]
    gd_cols = gd.io.csv.read_csv_strings(path,
                        names=col_names, dtype=dtypes,
                        skiprows=1)
    str_cols = {} # column name => nvstring object
    df = gd.DataFrame()
    for name,dtype,ds in zip(col_names,dtypes,gd_cols):
        if cols is not None and name not in cols:
            continue
        if dtype =='str':
            str_cols[name] = ds
        else:
            df[name] = ds
    del gd_cols
    return df,str_cols

@timer
def get_dtype(path,nrows=10):
    """get data type for cudf.read_csv
    by using pandas's read_csv with a small number of rows.

    Parameters
    ----------
    path : str, path of the csv file
    nrows : int, default: 10
        number of rows to read for pd.read_csv

    Returns
    ----------
    col2dtype : dictionary, column name => data type
    """
    if nrows is not None:
        train = pd.read_csv(path,nrows=nrows)
    else:
        train = pd.read_pickle(path.replace('.csv','.pkl'))
    col2dtype = OrderedDict()
    for col in train.columns:
        if train[col].dtype=='O':
            col2dtype[col] = 'str'
        elif train[col].dtype==np.int64:
            col2dtype[col] = 'int32'
        else:
            col2dtype[col] = 'float32'
    return col2dtype

def mkdir(path):
    if os.path.exists(path)==0:
        os.mkdir(path)

@timer
def get_id_y(mode):
    tr,te = 'train','test'
    out = '%s/%s_y.pkl'%(CACHE,tr)
    if os.path.exists(out):
        y = pd.read_pickle(out)[YCOL].values
        out = out.replace('_y','_id')
        tr_id = pd.read_pickle(out)[IDCOL].values
        te_id = pd.read_pickle(out.replace(tr,te))[IDCOL].values
        return y,tr_id,te_id

    tr_path,te_path = get_tr_te_paths(mode)
    gtr_y,gtr_id = read_csv_with_nvstring(tr_path,[IDCOL,YCOL])
    _,gte_id = read_csv_with_nvstring(te_path,[IDCOL])

    y = gtr_y[YCOL].to_array()
    pd.DataFrame({YCOL:y}).to_pickle(out)
    #pd.DataFrame({YCOL:y}).to_csv(out,index=False,compression='gzip')
    del gtr_y

    out = out.replace('_y','_id')
    tr_id = np.array(gtr_id[IDCOL].to_host())
    pd.DataFrame({IDCOL:tr_id}).to_pickle(out)
    #pd.DataFrame({IDCOL:tr_id}).to_csv(out,index=False,compression='gzip')
    del gtr_id

    te_id = np.array(gte_id[IDCOL].to_host())
    pd.DataFrame({IDCOL:te_id}).to_pickle(out.replace(tr,te))
    #pd.DataFrame({IDCOL:te_id}).to_csv(out.replace(tr,te),index=False,compression='gzip')
    del gte_id

    return y,tr_id,te_id

def get_tr_te_paths(mode):
    tr_path = '%s/train.csv'%PATH
    te_path = '%s/test.csv'%PATH
    return tr_path,te_path

def rm_cols(gdf,cols):
    gcols = [i for i in gdf.columns]
    cols = set(cols)
    for col in cols:
        if col in gcols:
            del gdf[col]
    return gdf

def one_zero_shuffle(x,y):
    x = x.copy()
    mask = y>0
    x1 = x[mask].copy()
    x1 = x1.T
    np.random.shuffle(x1)
    x[mask] = x1.T
    return x,y
    print(x[mask][0,-10:])
    ids = np.arange(x1.shape[0])
    for c in range(x1.shape[1]):
        np.random.shuffle(ids)
        x1[:,c] = x1[ids][:,c]
    x[mask] = x1#[ids]
    print(x[mask][0,-10:])
    return x,y 

def augment_fix(x,y,t=1):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for k,v in FDIC.items():
            np.random.shuffle(ids)
            x1[:,v] = x1[ids][:,v]
        xs.append(x1)

    for i in range(0):
        mask = (y==0)
        mask = mask&(np.random.rand(mask.shape[0])<0.1)
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for k,v in FDIC.items():
            np.random.shuffle(ids)
            x1[:,v] = x1[ids][:,v]
        xn.append(x1)

    xs = np.vstack(xs)
    #xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])#*0.9
    #yn = np.zeros(xn.shape[0])#*0.1
    x = np.vstack([x,xs])#,xn])
    y = np.concatenate([y,ys])#,yn])
    return x,y

def augment(x,y,t=1):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    #xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])#*0.9
    #yn = np.zeros(xn.shape[0])#*0.1
    x = np.vstack([x,xs])#,xn])
    y = np.concatenate([y,ys])#,yn])
    return x,y

@timer
def add_distance(x,y,xt,xte,funcs):
    ids = np.arange(x.shape[0])
    x1,x2,y1,y2 = train_test_split(ids,y, test_size=0.5, random_state=42,stratify=y)
    #funcs = ['mean','min']    

    xnew = np.zeros([x.shape[0],len(funcs)])    
    xtnew = np.zeros([xt.shape[0],len(funcs)])

    x2_pos = x[x2][y2==1]
    xnew[x1] = _add_distance(x[x1], x2_pos, funcs)

    x1_pos = x[x1][y1==1]
    xnew[x2] = _add_distance(x[x2], x1_pos, funcs)

    x_pos = x[y==1]
    xtnew = _add_distance(xt, x_pos, funcs)
    xtenew = _add_distance(xte, x_pos, funcs)

    return xnew,xtnew,xtenew

def _add_distance(x,base,funcs):
    B = 10000
    res = []
    base = np.expand_dims(base,0)
    for i in range(0,x.shape[0],B):
        s,e = i,min(i+B,x.shape[0])
        xm = x[s:e]
        xm = np.expand_dims(xm,1)
        dist = xm-base
        dist = np.sum(dist*dist,axis=2)
        #print(xm.shape,base.shape,dist.shape)
        s = pd.DataFrame()
        for func in funcs:
            s[func] = eval('np.%s(dist,axis=1)'%func)
        res.append(s.values)
        if i%100 == 0:
            print(i,'done')
    return np.vstack(res)
   

def run_cv_sub(X,y,folds,names,xid,rs=126,model_name='nn',Xt=None,leak=False):
    global FOLD
    ypred = np.zeros_like(y)*1.0
    scores = []
    ysub = 0
    Xt0 = Xt
    splits = []
    #kf = KFold(n_splits=folds, shuffle=True, random_state=rs)
    #kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rs)
    kf = pickle.load(open('cache/kfolds.pkl','rb'))
    #for i,(train_index, test_index) in enumerate(kf.split(X,y,xid)):
    for i,(train_index, test_index) in enumerate(kf):
        splits.append((train_index, test_index))
        FOLD = i
        y_train,y_test = y[train_index],y[test_index]
        X_train,X_test = X[train_index], X[test_index]

        #X_train,y_train = one_zero_shuffle(X_train,y_train)
        yp,ysubc = 0,0
        N = 1
        #tags = ['mean','std'] 
        for _ in range(N):
            #X_train0,y_train0 = augment(X_train,y_train)
            #X_train0,y_train0 = X_train,y_train
            X_train0,X_test0,Xt0,names0 = mtr_encodes(X_train.copy(),y_train.copy(),X_test.copy(),Xt.copy(),names)
            X_train0,y_train = augment_fix(X_train0,y_train)
            nf = X_train0.shape[1]//X.shape[1]-1
            #names0 = ['%s_%d'%(i) for c,i in enumerate(names)]+names
            print(len(names0),X_train0.shape[1])
            assert len(names0) == X_train0.shape[1]
            ypc,ysubc,model = run_one_fold(X_train0,y_train,X_test0,y_test,model_name,Xt0,leak,names0,ysubc)
            yp+=ypc
            ysub+=ysubc
        yp/=N
        ysub/=N
        loss = get_score(y_test,yp,METRIC)
        scores.append(loss)
        ypred[test_index] = yp
        print('Fold: %d %s:%.5f'%(i,METRIC,scores[-1]))
        fo = open('cv.score','a')
        fo.write('GPU: %d Fold: %d %s:%.5f\n'%(GPU,i,METRIC,scores[-1]))
        fo.close()
    if ysub is not None:
        ysub/=(i+1)
    score = get_score(y,ypred,METRIC)
    scores = ['%.4f'%i for i in scores]+['ave:%.4f'%np.mean(scores),'std:%.3f'%np.std(scores)]
    if os.path.exists('%s/splits.p'%CACHE)==0:
        pickle.dump(splits,open('%s/splits.p'%CACHE,'wb'))
    return scores,score,ypred,ysub,model 

def mtr_encodes(x0,y,xt0,xte0,names):
    global FDIC
    x,xt,xte = [x0],[xt0],[xte0]
    out = names.copy()
    fc = x0.shape[1]
    for i in range(x0.shape[1]):
        FDIC[i] = [i]
        if i in [10,14,17,29,30,38,39,41,42,46,47,61,64,65,68,69,72,73,79,84,96,98,100,103,117,120,124,126,129,136,140,152,158,160,161,176,182,183,185,189]:
            continue
        a,b,c = mtr_encode(x0[:,i:i+1],y,xt0[:,i:i+1],xte0[:,i:i+1])
        #x[:,i],xt[:,i],xte[:,i] = a[:,0],b[:,0],c[:,0]
        if a is None:
            continue
        FDIC[fc] = [fc+j for j in range(a.shape[1])]
        fc += a.shape[1]
        out.extend(['mtr_%s_%d'%(names[i],j) for j in range(a.shape[1])])
        x.append(a)
        xt.append(b)
        xte.append(c)
    x = np.hstack(x)
    xt = np.hstack(xt)
    xte = np.hstack(xte)
    return x,xt,xte,out

def mtr_encode(x,y,xt,xte):
    xnew = None
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=128)
    for i,(x1,x2) in enumerate(kf.split(x,y)):
        _,b,_ = mtr_gd(x[x1],y[x1],x[x2],None)
        if b is None:
            return None,None,None
        if xnew is None:
            xnew = np.zeros([x.shape[0],b.shape[1]])
        xnew[x2] = b

    a,b,c = mtr_gd(x,y,xt,xte)
    #xnew = (xnew+a)*0.5
    if b is None:
        return None,None,None
    xt,xte = b,c
    return xnew,xt,xte

def mtr_gd(x,y,xt,xte):
    col = 'sum_y'
    tr = gd.DataFrame()
    tr['y'] = y.astype(np.float32)
    tr['x'] = np.ascontiguousarray(x[:,0])
    std = tr['x'].std()

    df = tr.groupby('x').agg({'y':['sum','count']})
    colx = 'count_y'
    df[colx] = df[colx].astype('float32')#/df[colx].max()

    df = df.sort_values('x')

    df = to_pandas(df)
    #df = df.set_index('x')
    cols = []
    df[col] = df[col].cumsum()
    df[colx] = df[colx].cumsum()

    """
    if xte is not None:
        x_all = np.vstack([xt,xte])[:,0]
    else:
        x_all = np.vstack([xt])[:,0]
    x_all = np.unique(x_all)

    dg = pd.DataFrame({'x':x_all})
    """

    for i in [2]:
        tm = getp_vec_sum(df['x'].values,df['x'].values,df[col].values,std,c=i)
        cm = getp_vec_sum(df['x'].values,df['x'].values,df[colx].values,std,c=i)+1
        df[str(i)] = tm/cm
        cols.append(str(i))
    #df = df.reset_index()
    for i in df.columns:
        df[i] = df[i].astype(np.float32)
    df = gd.from_pandas(df)
    #tr = merge(tr,df,on='x',how='left')

    te = gd.DataFrame()
    te['x'] = np.ascontiguousarray(xt[:,0])
    te = merge(te,df,on='x',how='left')

    col = cols[0]
    if xte is not None:
        tes = gd.DataFrame()
        tes['x'] = np.ascontiguousarray(xte[:,0])
        tes = merge(tes,df,on='x',how='left')
        xte = to_pandas(tes)
        ratio = xte[col].isnull().sum()*1.0/xte.shape[0]
        print('test null ratio %.4f'%(ratio))
        f = open('tmp','a')
        f.write('test null ratio %.4f\n'%(ratio))
        f.close()
        #if ratio>0.5:
        #    return None,None,None
        xte = xte[cols].values
        del tes
    xt = to_pandas(te)
    print('valid null ratio %.4f'%(xt[col].isnull().sum()*1.0/xt.shape[0]))
    f = open('tmp','a')
    f.write('valid null ratio %.4f\n\n'%(xt[col].isnull().sum()*1.0/xt.shape[0]))
    f.close()
    xt = xt[cols].values
    del df,tr,te
    return None,xt,xte

def getp_vec_sum(x,x_sort,y,std,c=0.5):
    # x is sorted
    left = x - std/c
    right = x + std/c
    p_left = np.searchsorted(x_sort,left)
    p_right = np.searchsorted(x_sort,right)
    p_right[p_right>=y.shape[0]] = y.shape[0]-1
    return (y[p_right]-y[p_left])#/(p_right-p_left+1)

def run_one_fold(X_train,y_train,X_test,y_test,model_name,Xt,leak,names,ysub):
    if leak:
        va=[X_test,y_test]
    else:
        va=None
    yp,ysx,model = _fit_predict(model_name,X_train,y_train,X_test,Xt,va,names)
    if ysx is not None:
        ysub += ysx
    return yp,ysub,model

def rank_norm(y):
    return y.argsort().argsort()*1.0/y.shape[0]

def _fit_predict(model_name,X,y,Xt,Xte,va,names):
    n1 = names.copy()
    model = get_model(n1,num_class=NUM_CLASS,model=model_name)
    model.fit(X,y,va=va)
    yp = model.predict(Xt)
    if Xte is not None:
        ys = model.predict(Xte)
    else:
        ys = None
    if RANKNORM:
        return rank_norm(yp),rank_norm(ys),model
    else:
        return yp,ys,model

def get_model(names,num_class,model='nn'):
    model = eval('get_%s_model(names,num_class)'%model)
    return model

def get_nn_model(names,num_class):
    from dense_nn import DenseNN
    H = 32
    catfeas = 0#len(get_categorical_columns(names))
    params = get_nn_params(H,catfeas,num_class)
    return DenseNN(**params)

def get_nn_params(H,catfeas,num_class):
    params = {
        'classes':num_class,
        'embedding_size':4,
        'catfeas':catfeas,
        'stratified':True,
        'metric':METRIC,
        'save_path':'weights',
        #'load_path':'weights/r67.npy',
        'epochs':1000,# if MODE=='sub' else 100,
        'Hs':[H],#,H//2,H//4],
        'drop_prob':0,#0.5,
        'early_stopping_epochs':5,
        'learning_rate':0.001,
        'batch_size':2048,
        #'verbosity':10,
        'folds':4,
    }
    return params

def get_bag_xgb_model(names,num_class):
    xgb_params = get_xgb_params(names,num_class)
    #xgb_params.update({
    #    'watch':False,
    #    'early_stopping_rounds':None,
    #})
    #xgb_params['folds'] = 4
    #xgb_params['num_round'] = 500
    xgb_params['aug'] = augment_fix 
    print(xgb_params)
    model = bag_xgb_model(**xgb_params)
    return model

def get_bag_sk_lgb_model(names,num_class):
    lgb_params = get_sk_lgb_params(names,num_class)
    lgb_params['folds'] = 4
    #lgb_params['n_estimators'] =30000 
    model = bag_sk_lgb_model(**lgb_params)
    return model

def get_sk_lgb_model(names,num_class):
    lgb_params = get_sk_lgb_params(names,num_class)
    model = sk_lgb_model(**lgb_params)
    return model

def get_lgb_model(names,num_class):
    lgb_params = get_sk_lgb_params(names,num_class)
    model = lgb_model(**lgb_params)
    return model

def get_sk_lgb_params(names,num_class):
    random_state = 42
    params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_threads": 16,
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
    "min_sum_hessian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state
    }
    return params

def get_xgb_cpu_params(names,num_class):
    print("# clases",num_class)
    #categorical = get_categorical_columns(names)
    params =  {
        'objective': 'binary:logistic',
        #'objective':'reg:linear',
        'tree_method': 'hist',
        'early_stopping_rounds':100,#None,
        'eta':0.1,
        'nthread': 16,
        'stratified':True,
        'folds':4,
        'watch':True,#False,
        'num_class':num_class,
        'num_round':120000,
        'max_depth': 1,#5 if 'stack' not in MODE else 1,
        'silent':1,
        'subsample':0.5,
        'colsample_bytree': 0.5,
        'min_child_weight':100,
        'feature_names':names,
        'maximize':True,
        'eval_metric':METRIC,
        'verbose_eval':1000,
    }
    return params

def get_xgb_gpu_params(names,num_class):
    params = get_xgb_cpu_params(names,num_class)
    params.update({'tree_method': 'gpu_hist',
    })
    return params

def get_xgb_params(names,num_class):
    #params = get_xgb_cpu_params(names,num_class)
    params = get_xgb_gpu_params(names,num_class)
    return params

def get_xgb_model(names,num_class):
    xgb_params = get_xgb_params(names,num_class)
    model = xgb_model(**xgb_params)
    return model

@timer
def preprocess(X,Xt,names):
    scaler = StandardScaler()
    N = 0#len(get_categorical_columns(names))
    X = pd.DataFrame(X).fillna(0).values
    Xt = pd.DataFrame(Xt).fillna(0).values
    X[:,N:] = scaler.fit_transform(X[:,N:])
    Xt[:,N:] = scaler.transform(Xt[:,N:])
    return X,Xt

def main(mode='cv',model='lgb',leak=False):
    global MODE
    MODE = mode
    cname = sys.argv[0].replace('.py','_%d.py'%GPU)
    assert os.path.exists(cname)==0
    cmd = 'cp %s %s'%(sys.argv[0],cname)
    os.system(cmd)
    start = time.time()
    tag = 'leak' if leak else 'non-leak'
    tag = '%s_%s_%s_%d'%(mode,tag,model,GPU)
    out = '%s.csv.gz'%tag
    bl = [1]
    cl = [0]*len(bl)
    if 0 in bl:
        tag = 'stack_%s'%tag
    out = '%s.csv.gz'%tag
    if model in ['nn']:
        cl = [1]*len(bl)
    X,Xt,y,names,tr_id,te_id = build(mode,build_list=bl,cache=cl)
    if model in ['nn']:
        X,Xt = preprocess(X,Xt,names)
    scores,score,yp,yps,model = run_cv_sub(X,y,FOLDS,names,tr_id,model_name=model,Xt=Xt,leak=leak)
    try:
        model.get_importance("%s_importance.csv"%tag)
    except:
        pass
    print(METRIC,score)
    if mode.startswith('sub'):
        s = pd.DataFrame({IDCOL:te_id,YCOL:yps})
        s.to_csv(out,index=False,compression='gzip',float_format='%.6f')
        print(s.head())
    s = pd.DataFrame({YCOL:yp})
    s.to_csv(out.replace('sub_','cv_'),index=False,compression='gzip',float_format='%.6f')
    print(s.head())
    duration = time.time()-start
    files = [i for i in os.listdir('.') if i.startswith(tag)]+[cname]+[out.replace('sub_','cv_')]
    write_log(duration,score,"{} {}".format(tag,scores),mfiles=files)

if __name__ == '__main__':
    model = ['xgb','lgb','bag_xgb','bag_lgb','nn','sk_lgb','bag_sk_lgb'][2]
    main(mode='sub',model=model,leak=False)
    #main(mode='sub',model=model,leak=True)
