# coding: utf-8


import numpy as np
import pandas as pd
import os
import copy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import layers
from keras import backend as K
from keras import regularizers
from keras.constraints import max_norm
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.models import load_model
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D,AveragePooling2D,MaxPooling2D,Dropout
from sklearn import preprocessing

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import time

from datetime import datetime
import pickle

import gc
gc.collect()


# In[2]:


def nn_binary_stack(nn_model, train_x, train_y, test_x, kfolds, random_state=42, verbose=1, n_stacks=1,
                    stratified=True):

    if stratified:
        kf = StratifiedKFold(n_splits=kfolds, shuffle=True,
                             random_state=random_state)
        kf_ids = list(kf.split(train_x, train_y))
    else:
        kf = KFold(n_splits=kfolds, random_state=random_state)
        kf_ids = kf.split(train_y)

    train_blend_x = np.zeros((train_x.shape[0], n_stacks))
    test_blend_x = np.zeros((test_x.shape[0], n_stacks))
    blend_scores = np.zeros((kfolds, n_stacks))

    if verbose > 0:
        print("Start stacking.")
    for j in range(n_stacks):
        model_time = time.strftime("%Y%m%d%H%M")
        if verbose > 0:
            print("Stacking model", j+1)
        test_blend_x_j = np.zeros((test_x.shape[0]))
        for i, (train_ids, val_ids) in enumerate(kf_ids):
            start = time.time()
            if verbose > 0:
                print("Model %d fold %d" % (j+1, i+1))
            train_x_fold = train_x[train_ids, :]
            train_y_fold = train_y[train_ids]
            val_x_fold = train_x[val_ids, :]
            val_y_fold = train_y[val_ids]
            
            save_model_name_1 = '../model/' + model_time + '_fold_' + str(i) + '_step_1.model'
            
            model = Convnet()
            if verbose > 0:
                print(i, model.summary())
            
            c = optimizers.adam(lr = 0.001)
            model.compile(loss='binary_crossentropy', optimizer=c, metrics=['accuracy', 'binary_crossentropy'])

            early_stop = EarlyStopping(monitor='val_loss', mode='auto', patience=10, verbose=1)
            model_checkpoint = ModelCheckpoint(save_model_name_1,monitor='val_loss', 
                                       mode = 'min', save_best_only=True, verbose=1)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode = 'min',factor=0.5, 
                                          patience=5, min_lr=0.0001, verbose=1)
            
            history = model.fit(train_x_fold, train_y_fold, 
                                epochs=EPOCHS, batch_size=BATCH_SIZE, 
                                validation_data=(val_x_fold, val_y_fold), 
                                callbacks=[model_checkpoint, reduce_lr, early_stop], 
                                verbose=verbose)
            
               
            ## Load weights    
            model.load_weights(save_model_name_1)
            val_y_predict_fold = model.predict(val_x_fold).ravel()
            score = roc_auc_score(val_y_fold, val_y_predict_fold)
            if verbose > 0:
                print("Score for the best Model %d fold %d: %f " % (j+1, i+1, score))
            
            blend_scores[i, j] = score
            train_blend_x[val_ids, j] = val_y_predict_fold
            test_blend_x_j = test_blend_x_j + model.predict(test_x).ravel()
            if verbose > 0:
                print("Model %d fold %d finished in %d seconds." %
                      (j+1, i+1, time.time()-start))
            del model
            gc.collect()

        test_blend_x[:, j] = test_blend_x_j/kfolds
        if verbose > 0:
            print("Score for model %d is %f" %
                  (j+1, np.mean(blend_scores[:, j])))
    return train_blend_x, test_blend_x, blend_scores





## From https://www.kaggle.com/bochuanwu/cnn-model-auto-featured-test
## Deleted unused layers
def Convnet(input_shape = (20,20,2),classes = 1):

    X_input = Input(input_shape)
 
    # Stage 3
    X = Conv2D(64,kernel_size=(2,2),strides=(2,2),name="conv1",kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization()(X)
    X = Activation("tanh")(X)
 
    X = Flatten()(X)
    X = Dense(classes, activation='sigmoid')(X)
    model = Model(inputs=X_input,outputs=X)
 
    return model





train_df =  pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

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

print(train_tran_df.shape,test_tran_df.shape)



EPOCHS = 300
BATCH_SIZE = 256
N_STACKS = 1

full_vars = num_vars + p2_num_vars + p3_num_vars + p4_num_vars 
train_x = train_tran_df[full_vars].values
train_y = train_tran_df['target'].values

test_x = test_tran_df[full_vars].values

train_x=np.reshape(train_x,(train_x.shape[0],20,20,2))
test_x=np.reshape(test_x,(test_x.shape[0],20,20,2))


train_stack_x_nn, test_stack_x_nn, _ =     nn_binary_stack(Convnet, train_x, train_y,
                    test_x, 5,
                    random_state=42,
                    stratified=True, n_stacks=1)


from datetime import datetime
import pickle
nn_stack_score = roc_auc_score(train_y, train_stack_x_nn.mean(axis=1))
print ('Score: %f ' % (nn_stack_score))
if nn_stack_score>0.8900:
    stack_score = str(int(nn_stack_score * 10000))
    time_str= datetime.now().strftime("%Y%m%d%H%M")
    pickle.dump(train_stack_x_nn, open('../stacking/'+stack_score + '_' + time_str+'_train_stack_x_nn.pkl', 'wb'))
    pickle.dump(test_stack_x_nn, open('../stacking/'+stack_score + '_' + time_str+'_test_stack_x_nn.pkl', 'wb'))
    sub = pd.DataFrame({'ID_code':test_id,'target':test_stack_x_nn.mean(axis=1)})
    sub.to_csv('../output/%s_%s.csv' % (stack_score, time_str),index=False)
    print('../output/%s_%s.csv created.' % (stack_score, time_str))

