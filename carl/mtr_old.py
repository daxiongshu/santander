def mtr_encodes(x,y,xt,xte,names,window_sizes=[500,1000]):
    names = names.copy()
    x0,xt0,xte0 = x.copy(),xt.copy(),xte.copy()
    x,xt,xte = [x0],[xt0],[xte0]
    for i in range(x0.shape[1]):
        print('feature mtr encoding',names[i])
        a,b,c = mtr_encode(x0[:,i],y,xt0[:,i],xte0[:,i],window_sizes)
        x.append(a)
        xt.append(b)
        xte.append(c)
        names.extend(['%s_mtr_%d'%(names[i],j) for j in window_sizes])
    x = np.hstack(x)
    xt = np.hstack(xt)
    xte = np.hstack(xte)
    return x,xt,xte,names

def mtr_encode(x,y,xt,xte,window_sizes):
    ids = np.arange(x.shape[0])
    x1,x2,y1,y2 = train_test_split(ids,y, test_size=0.5, random_state=42,stratify=y)

    xnew = np.zeros([x.shape[0],len(window_sizes)])
    _,xnew[x2],_ = mtr_pd(x[x1],y[x1],x[x2],None,window_sizes)
    _,xnew[x1],_ = mtr_pd(x[x2],y[x2],x[x1],None,window_sizes)
    _,xt,xte = mtr_pd(x,y,xt,xte,window_sizes)
    return xnew,xt,xte

def mtr_pd(x,y,xt,xte,window_sizes=[500,1000]):
    col = 'mean_y'
    tr = pd.DataFrame()
    tr['y'] = y.astype(np.float32)
    tr['x'] = x
    df = tr.groupby('x').agg({'y':'mean'})
    df.columns = [col]
    df = df.reset_index()
    df = df.sort_values('x')

    df[col] = df[col].interpolate()
    cols = []
    for i in window_sizes:
        df['mtr_%d'%i] = df[col].rolling(i,min_periods=1).mean()
        cols.append('mtr_%d'%i)
    tr = tr.merge(df,on='x',how='left')
    te = pd.DataFrame()
    te['x'] = xt
    te = te.merge(df,on='x',how='left')

    if xte is not None:
        tes = pd.DataFrame()
        tes['x'] = xte
        tes = tes.merge(df,on='x',how='left')
        #print('test null ratio %.4f'%(tes[cols[0]].isnull().sum()*1.0/tes.shape[0]))
        f = open('mtr_null.txt','a')
        f.write('test null ratio %.4f\n'%(tes[cols[0]].isnull().sum()*1.0/tes.shape[0]))
        xte = tes[cols].values
        del tes
        #print('valid null ratio %.4f'%(te[cols[0]].isnull().sum()*1.0/te.shape[0]))
        f.write('valid null ratio %.4f\n\n'%(te[cols[0]].isnull().sum()*1.0/te.shape[0]))
        f.close()
    x,xt = tr[cols].values,te[cols].values
    return x,xt,xte