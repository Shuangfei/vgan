import sys
sys.path.append('..')

import cPickle as pkl
import numpy as np
import theano
import theano.tensor as T
import lasagne
from collections import OrderedDict
from utils import get_net

data_dir = '../datasets/cifar-10-batches-py/'
train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4']
valid_file = 'data_batch_5'
test_file = 'test_batch'

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pkl.load(f)
    dict['data'] = np.asarray(dict['data'], dtype=theano.config.floatX)
    return dict

train_data = [unpickle(data_dir + file) for file in train_files]
valid_data = unpickle(data_dir + valid_file)
test_data = unpickle(data_dir + test_file)

trX = np.concatenate([batch['data'] for batch in train_data], axis=0) / 255.
trY = np.concatenate([batch['labels'] for batch in train_data], axis=0).astype('int32')
vaX = valid_data['data'] / 255.
vaY = np.array(valid_data['labels']).astype('int32')
teX = test_data['data'] / 255.
teY = np.array(test_data['labels']).astype('int32')

def transform(X):
    return X.reshape(-1, 3, 32, 32)

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
discrim_opt['input'] = (3, 32, 32)
discrim_opt['conv_relu_1'] = (128, 5, 5)
discrim_opt['pool_1'] = 2
discrim_opt['conv_relu_2'] = (128, 5, 5)
discrim_opt['pool_2'] = 2
discrim_opt['dense_relu'] = (1024,)
discrim_opt['dense_sigmoid'] = (128,)

print 'disciminator configuration'
print discrim_opt.keys()
discrim_net, discrim_NET = get_net(X, discrim_opt)

cond_opt = OrderedDict()
cond_opt['input'] = (3, 32, 32)
cond_opt['noise'] = 0.1
cond_opt['conv_batchnorm_relu_1_1'] = (96, 3, 3)
cond_opt['conv_batchnorm_relu_1_2'] = (96, 3, 3)
cond_opt['conv_batchnorm_relu_1_3'] = (96, 3, 3)
cond_opt['pool_1'] = 2
cond_opt['dropout_1'] = 0.5
cond_opt['conv_batchnorm_relu_2_1'] = (192, 3, 3)
cond_opt['conv_batchnorm_relu_2_2'] = (192, 3, 3)
cond_opt['conv_batchnorm_relu_2_3'] = (192, 3, 3)
cond_opt['pool_2'] = 2
cond_opt['dropout_2'] = 0.5
cond_opt['conv_batchnorm_relu_3_1'] = (192, 3, 3)
cond_opt['nin_batchnorm_relu_3_2'] = (192,)
cond_opt['nin_batchnorm_relu_3_3'] = (192,)
cond_opt['pool_3'] = -1
cond_opt['dense_softmax'] = (ny,)

cond_net, cond_NET = get_net(X, cond_opt)

gen_opt = OrderedDict()
gen_opt['input'] = (3, 32, 32)
gen_opt['conv_batchnorm_relu_1'] = (256, 5, 5)
gen_opt['pool_1'] = 2
gen_opt['conv_batchnorm_relu_2'] = (256, 5, 5)
gen_opt['pool_2'] = 2
gen_opt['dense_batchnorm_tanh'] = (2048,)
gen_opt['noise2'] = 0.5
gen_opt['dense_batchnorm_relu_2'] = (256*8*8,)
gen_opt['reshape'] = (256, 8, 8)
gen_opt['deconv_batchnorm_relu_1'] = (256, 5, 5)
gen_opt['deconv_sigmoid'] = (3, 5, 5)

print 'generator configuration'
print gen_opt.keys()
gen_net, gen_NET = get_net(Z, gen_opt)

