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

def build_model(X, Z, discrim_net, cond_net, gen_net, opt, ny=None):
    def _clip(x):
        epsilon = 1e-4
        return T.clip(x, epsilon, 1 - epsilon)
    def get_energy(p):
        return T.sum(lasagne.objectives.binary_crossentropy(p, p), axis=1).mean()
    def get_entropy(p):
        return T.sum(lasagne.objectives.binary_crossentropy(p.mean(axis=0), p.mean(axis=0)))

    gX = get_output(gen_net, deterministic=False)
    reconZ = _clip(get_output(gen_net, deterministic=True,
                              batch_norm_use_averages=False,
                              batch_norm_update_averages=True
                          ))
    reconErr = lasagne.objectives.binary_crossentropy(reconZ, Z).mean()
    p_x = _clip(get_output(discrim_net, deterministic=False))
    p_gx = _clip(get_output(discrim_net, gX, deterministic=False))    
    energy_x = get_energy(p_x)
    energy_gx = get_energy(p_gx)
    entropy_x = get_entropy(p_x)
    entropy_gx = get_entropy(p_gx)
    
    d_cost = energy_x - energy_gx - entropy_x
    g_cost = opt['gamma'] * (energy_gx - entropy_gx) + (1 - opt['gamma']) * reconErr

    discrim_params = get_all_params(discrim_net, trainable=True)
    gen_params = get_all_params(gen_net, trainable=True)

    lr = T.scalar()
    if ny:
        Y = T.ivector()
        clf = cond_net
        params = get_all_params(clf, trainable=True)
        pred = get_output(clf, deterministic=False)
        loss = lasagne.objectives.categorical_crossentropy(pred, Y).mean()
        if opt['augment']:
            X_tilde = get_output(gen_net, X, deterministic=False)
            pred_tilde = get_output(clf, X_tilde, deterministic=False)
            loss += lasagne.objectives.categorical_crossentropy(pred_tilde, Y).mean()
            
        updates = lasagne.updates.adadelta(loss, params, learning_rate=lr)
        error = T.neq(T.argmax(pred, axis=1), Y)
        pred_static = get_output(clf, deterministic=True)
        error_static = T.neq(T.argmax(pred_static, axis=1), Y)
        _train_clf = theano.function([X, Y, lr], error, updates=updates)
        _test_clf = theano.function([X, Y], error_static)

    print 'COMPILING'
    sys.stdout.flush()
    t = time()

    g_updates = lasagne.updates.adadelta(g_cost, gen_params, learning_rate=lr)
    _train_g = theano.function([X, Z, lr], [energy_x, energy_gx],
                               updates=g_updates, on_unused_input='ignore')
    d_updates = lasagne.updates.adadelta(d_cost, discrim_params, learning_rate=lr)
    _train_d = theano.function([X, Z, lr], [energy_x, energy_gx],
                               updates=d_updates, on_unused_input='ignore')

    _gen = theano.function([Z], gX, on_unused_input='ignore')
    _recon = theano.function([Z], reconZ)
    print '%.2f seconds to compile theano functions'%(time()-t)
    sys.stdout.flush()

    fns = {}
    fns['train_g'] = _train_g
    fns['train_d'] = _train_d
    fns['gen'] = _gen
    fns['recon'] = _recon
    if ny:
        fns['train_clf'] = _train_clf
        fns['test_clf'] = _test_clf
    
    return fns

def train_vgan(fns, iter_data, opt, **kwargs):
    desc = opt['desc']
    samples_dir = 'samples/%s'%desc
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
        
    for sample_imb in iter_data(subset='test', size=100): break
    color_grid_vis(sample_imb.transpose(0, 2, 3, 1), (10, 10), 'samples/%s/inputs.png'%(desc))    
    print desc.upper()
    sys.stdout.flush()
    n_updates = 0
    n_updates = 0
    n_examples = 0
    t = time()
    energy_x = 0
    energy_gx = 0
    entropy_x = 0
    entropy_gx = 0

    for epoch in range(opt['niter'] + opt['niterdecay']):
        if epoch <= opt['niter']:
            lr = opt['lr']
        else:
            lr = opt['lr'] * (opt['niter'] + opt['niterdecay'] -  epoch + 1.) / opt['niterdecay']
        for imb in iter_data(size=2*opt['nbatch'], shuffle=True):
            this_x = imb[:len(imb)/2]
            this_z = imb[len(imb)/2:]
            for _ in range(opt['k']):
                this_energy_x, this_energy_gx = fns['train_g'](this_x, this_z, lr)
                energy_gx = 0.9 * energy_gx + 0.1 * this_energy_gx
            this_energy_x, this_energy_gx = fns['train_d'](this_x, this_z, lr)
            energy_x = 0.9 * energy_x + 0.1 * this_energy_x
            energy_gx = 0.9 * energy_gx + 0.1 * this_energy_gx
            
            n_updates += 1
            n_examples += len(imb)

        samples = np.asarray(fns['gen'](sample_imb))
        color_grid_vis(samples.transpose(0, 2, 3, 1), (10, 10), 'samples/%s/%d.png'%(desc, epoch))
        recons = np.asarray(fns['recon'](sample_imb))
        color_grid_vis(recons.transpose(0, 2, 3, 1), (10, 10), 'samples/%s/recon%d.png'%(desc, epoch))
        print 'epoch %d, energy_gx %.4f' % (epoch, energy_gx)
        print 'epoch %d, energy_x %.4f\n' % (epoch, energy_x)

        sys.stdout.flush()

def train_clf(fns, iter_data, opt):
    print 'training classifier ...'
    sys.stdout.flush()
    num = opt['num_for_clf']
    lr = opt['lr']
    trHist = []
    vaHist = []
    teHist = []
    for epoch in range(opt['niter']):
        if (epoch == int(0.6*opt['niter'])) or (epoch == int(0.9*opt['niter'])):
            lr *= 0.1
            print 'reducing learning rate to %.2f' % lr
        trErr = []
        vaErr = []
        teErr = []
        for x, y in iter_data(size=opt['nbatch'], shuffle=True, subset='train', iter_y=True, num=num):
            trErr = np.append(trErr, fns['train_clf'](x, y, lr))
        for x, y in iter_data(size=2*opt['nbatch'], shuffle=False, subset='valid', iter_y=True):
            vaErr = np.append(vaErr, fns['test_clf'](x, y))
        for x, y in iter_data(size=2*opt['nbatch'], shuffle=False, subset='test', iter_y=True):
            teErr = np.append(teErr, fns['test_clf'](x, y))
        trHist.append(trErr.mean())
        vaHist.append(vaErr.mean())
        teHist.append(teErr.mean())
        print 'epoch %d train error %.4f valid error %.4f test error %.4f' \
            % (epoch, trHist[-1], vaHist[-1], teHist[-1])
        sys.stdout.flush()
    best_epoch = np.argmin(vaHist)
    print 'best epoch %d train error %.4f valid error %.4f test error %.4f' \
        % (best_epoch, trHist[best_epoch], vaHist[best_epoch], teHist[best_epoch])
    sys.stdout.flush()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='cifar10')
    parser.add_argument('-nbatch', type=int, default=100)
    parser.add_argument('-niter', type=int, default=100)
    parser.add_argument('-niterdecay', type=int, default=0)
    parser.add_argument('-k', type=int, default=2)
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

    train_vgan(fns, iter_data, opt)
    if opt['save_model']:
        print 'saving model...'
        sys.stdout.flush()
        params = get_all_param_values([discrim_net, gen_net])
        with open('models/%s.pkl'%opt['desc'], 'w') as f:
            pkl.dump(params, f)
