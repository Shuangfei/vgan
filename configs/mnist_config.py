import sys
sys.path.append('..')

import cPickle as pkl
import gzip
import cPickle as pkl
import numpy as np
import theano
import theano.tensor as T
import lasagne
from collections import OrderedDict
from utils import get_net

data_dir = '../datasets/mnist/mnist.pkl.gz'

with gzip.open(data_dir) as f:
    data = pkl.load(f)
train_data, valid_data, test_data = data
trX, trY = train_data
vaX, vaY = valid_data
teX, teY = test_data

trY = trY.astype('int32')
vaY = vaY.astype('int32')
teY = teY.astype('int32')    

def transform(X):
    return X.reshape(-1, 1, 28, 28)

def iter_data(subset='train', iter_y=False, size=128, shuffle=True, num=None):
    if subset == 'train':
        data_x = trX
        data_y = trY
    elif subset == 'valid':
        data_x = vaX
        data_y = vaY
    elif subset == 'test':
        data_x = teX
        data_y = teY
        
    if num:
        data_x = data_x[:num]
        data_y = data_y[:num]

    if shuffle:
        idx = np.random.permutation(data_x.shape[0])
        data_x = data_x[idx]
        data_y = data_y[idx]

    data_x = transform(data_x)
    n = data_x.shape[0]
    batches = n / size
    if n % size != 0:
        batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        if not iter_y:
            yield data_x[start:end]
        else:
            yield (data_x[start:end], data_y[start:end])

X = T.tensor4()
Z = T.tensor4()
ny = 10
    
discrim_opt = OrderedDict()
discrim_opt['input'] = (1, 28, 28)
discrim_opt['conv_relu_1'] = (64, 5, 5)
discrim_opt['pool_1'] = 2
discrim_opt['conv_relu_2'] = (64, 5, 5)
discrim_opt['pool_2'] = 2
discrim_opt['dense_relu'] = (1024,)
discrim_opt['dense_sigmoid'] = (128,)

print 'disciminator configuration'
print discrim_opt.keys()
discrim_net, discrim_NET = get_net(X, discrim_opt)

cond_opt = OrderedDict()
cond_opt['input'] = (1, 28, 28)
cond_opt['noise'] = 0.1
cond_opt['conv_batchnorm_relu_1'] = (64, 3, 3)
cond_opt['conv_batchnorm_relu_2'] = (64, 3, 3)
cond_opt['pool_1'] = 2
cond_opt['dropout_1'] = 0.5
cond_opt['conv_batchnorm_relu_3'] = (64, 3, 3)
cond_opt['conv_batchnorm_relu_4'] = (64, 3, 3)
cond_opt['pool_2'] = 2
cond_opt['dropout_2'] = 0.5
cond_opt['conv_batchnorm_relu_5'] = (64, 3, 3)
cond_opt['conv_batchnorm_relu_6'] = (64, 3, 3)
cond_opt['pool_3'] = -1
cond_opt['dense_softmax'] = (ny,)

cond_net, cond_NET = get_net(X, cond_opt)

gen_opt = OrderedDict()
gen_opt['input'] = (1, 28, 28)
gen_opt['conv_batchnorm_relu_1'] = (64, 5, 5)
gen_opt['pool_1'] = 2
gen_opt['conv_batchnorm_relu_2'] = (64, 5, 5)
gen_opt['pool_2'] = 2
gen_opt['dense_batchnorm_tanh'] = (256,)
gen_opt['noise2'] = 0.5
gen_opt['dense_batchnorm_relu_2'] = (64*7*7,)
gen_opt['reshape'] = (64, 7, 7)
gen_opt['deconv_batchnorm_relu_1'] = (64, 5, 5)
gen_opt['deconv_sigmoid'] = (1, 5, 5)

print 'generator configuration'
print gen_opt.keys()
gen_net, gen_NET = get_net(Z, gen_opt)
