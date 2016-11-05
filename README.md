Experimental code for paper: Generative Adversarial Networks as Variational Training of Energy Based Models, under review at ICLR 2017

requirements: Theano and Lasagne

This code implements the variational contrastive divergence (VCD) in the paper.
Usage:

1. Download datasets (e.g., train 32x32.mat and test 32x32.mat http://ufldl.stanford.edu/housenumbers/), save under ./datasets

2. Train a VGAN with VCD by running (gamma == \rho in the paper):

python train_trans_vgan.py -dataset svhn -k 1 -lr 0.1 -niter 100 -gamma 0.1 -save_model 1

3. To run semisupervised learning with trained generator with data agumentation (set -augment to 0 to run without augmentation):

python test_trans_vgan.py -dataset svhn -load_model models/[modelname] -augment 1 -lr 1 -niter 200
