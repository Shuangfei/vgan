import os
import sys
import json
from time import time
import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
from sklearn.externals import joblib


import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng, t_rng
from lib.ops import conv_cond_concat, deconv, dropout
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data

import lasagne
from lasagne.layers import *
from collections import OrderedDict

class DeconvLayer(lasagne.layers.conv.BaseConvLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                     crop=0, untie_biases=False,
                     W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
                     nonlinearity=lasagne.nonlinearities.rectify, flip_filters=False, **kwargs):
        super(DeconvLayer, self).__init__(incoming, num_filters, filter_size, stride, crop,
                                          untie_biases, W, b, nonlinearity, flip_filters,
                                          n=2, **kwargs)

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        return (num_input_channels, self.num_filters) + self.filter_size

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_filters, 2*input_shape[2], 2*input_shape[3])

    def convolve(self, input, **kwargs):
        return deconv(input, self.W, subsample=(2, 2), border_mode='half')

class Noise2Layer(Layer):
    def __init__(self, incoming, ratio, **kwargs):
        super(Noise2Layer, self).__init__(incoming, **kwargs)
        self.rng = t_rng
        self.ratio = ratio
        
    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            return input
        else:
            mask = self.rng.binomial(input.shape, p=T.constant(1)-self.ratio, dtype=input.dtype)
            noise = self.rng.uniform(input.shape, -1., 1.)
            return mask * input + (1 - mask) * noise
        

def get_net(input, opt):
    NET = OrderedDict()
    nonlinearity_dict = {
        'relu': T.nnet.relu,
        'sigmoid':T.nnet.sigmoid,
        'softplus':T.nnet.softplus,
        'softmax':T.nnet.softmax,
        'lrelu':lasagne.nonlinearities.leaky_rectify,
        'exp':T.exp,
        'tanh':T.tanh,
    }
    for key, value in opt.items():
        for k in key.split('_'):
            if k == 'input':
                net = InputLayer((None,)+ value, input)
            elif k == 'reshape':
                net = ReshapeLayer(net, (-1,) + value)

            elif k == 'dense':
                net = DenseLayer(net, value[0], nonlinearity=None)
            elif k == 'nin':
                net = NINLayer(net, value[0], nonlinearity=None)
            elif k == 'conv':
                net = Conv2DLayer(net, value[0], value[1:], stride=1, pad='same', nonlinearity=None)
            elif k == 'deconv':
                net = DeconvLayer(net, value[0], value[1:], nonlinearity=None)
            elif k in nonlinearity_dict:
                net = NonlinearityLayer(net, nonlinearity_dict[k])
            elif k == 'batchnorm':
                net = BatchNormLayer(net)
            elif k == 'dropout':
                net = DropoutLayer(net, value)
            elif k == 'noise':
                net = GaussianNoiseLayer(net, value)
            elif k == 'noise2':
                net = Noise2Layer(net, value)
            elif k == 'pool':
                if value == -1:
                    net = GlobalPoolLayer(net)
                else:
                    net = MaxPool2DLayer(net, value)
            elif k == 'concat':
                net = ConcatLayer([net, InputLayer((None, value[0]), value[1])])
        NET[key] = net
    return net, NET


