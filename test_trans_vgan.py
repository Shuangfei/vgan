import sys
import os
import json
from time import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import cPickle as pkl

import theano
import theano.tensor as T

from lib.vis import color_grid_vis
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle

import lasagne
from lasagne.layers import *
from collections import OrderedDict

from utils import *

from train_trans_vgan import build_model, train_clf

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import svm

def test_energy(fns, iter_data, opt):
    energy_train = []
    for imb in iter_data(subset='train', size=200, shuffle=False):
        energy_train.append(fns['energy'](imb))
    energy_train = np.mean(energy_train)

    energy_test = []
    for imb in iter_data(subset='test', size=200, shuffle=False):
        energy_test.append(fns['energy'](imb))
    energy_test = np.mean(energy_test)
    
    print 'train energy %.3f, test energy %.3f' % (energy_train, energy_test)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='cifar10')
    parser.add_argument('-nbatch', type=int, default=100)
    parser.add_argument('-niter', type=int, default=100)
    parser.add_argument('-niterdecay', type=int, default=0)
    parser.add_argument('-k', type=int, default=1)
    parser.add_argument('-lr', type=float, default=0.1)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-augment', type=int, default=0)
    parser.add_argument('-desc', type=str, default='vgan')
    parser.add_argument('-save_model', type=int, default=0)
    parser.add_argument('-load_model', type=str, default='')
    parser.add_argument('-num_for_clf', type=int, default=None)
    opt = vars(parser.parse_args())

    if opt['dataset'] == 'cifar10':
        from configs.cifar10_config import X, Z, discrim_net, cond_net, gen_net, iter_data, ny
        
    elif opt['dataset'] == 'mnist':
        from configs.mnist_config import X, Z, discrim_net, cond_net, gen_net, iter_data, ny
        
    elif opt['dataset'] == 'svhn':
        from configs.svhn_config import X, Z, discrim_net, cond_net, gen_net, iter_data, ny

    fns = build_model(X, Z, discrim_net, cond_net, gen_net, opt, ny)

    if opt['load_model']:
        print 'loading model...'
        sys.stdout.flush()
        with open(opt['load_model'], 'r') as f:
            params = pkl.load(f)
        set_all_param_values([discrim_net, gen_net], params)

    if ny:
        train_clf(fns, iter_data, opt)        
