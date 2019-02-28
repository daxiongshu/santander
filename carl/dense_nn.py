import tensorflow as tf
from random import shuffle
import numpy as np
from scipy import sparse
from ml_robot import SKTFModel
import warnings
warnings.filterwarnings("ignore")

class DenseNN(SKTFModel):

    def _build(self):
        netname = 'DenseNN'

        Hs = self.params['Hs']
        D = self.params.get('drop_prob',0.25)
        C = self.params.get('classes',1)

        As = ['relu' for i in Hs]
        Ds = [D for i in Hs]
        Ds[-1] = 0

        print(self.X.shape)
        SF = self.X.shape[1]
        self.inputs = tf.placeholder(tf.float32,shape=[None,SF])
        net = self.inputs

        with tf.variable_scope(netname):
            for c,(h,a,d) in enumerate(zip(Hs,As,Ds)):
                net = self.fcblock(net,h,netname,c+1,a,d)
            net = self._fc(net, C, layer_name='%s/out'%(netname))
        return tf.squeeze(net)

    def fcblock(self,net,H,name,idx,activation,drop_prob,norm=False,tag=''):
        net = self._fc(net, H, layer_name='%s/%sfc%d'%(name,tag,idx))#,use_mask=True,thresh=1e-4)
        if norm:
            net = self._batch_normalization(net, layer_name='%s/%sbn%d'%(name,tag,idx))
        net = self._activate(net, activation)
        if drop_prob>0:
            net = self._dropout(net,1-drop_prob)
        return net

if __name__ == '__main__':
    x = np.array([[0,0,1,1],[1,0,0,0],[1,1,0,0],[1,1,1,0]])
    y = np.array([1,0,1,0])
    print(x.shape,y.shape,type(x),isinstance(x,sparse.csr_matrix))
    nn = DenseNN(batch_size=2,objective='regression',metric='rmse')
    nn.fit(x,y)
