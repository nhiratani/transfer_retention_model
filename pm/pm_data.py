#
# Model of tranfer learning and catastrophic forgetting
#
# permuted MNIST Dataset with latent
#

import sys
from math import *
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random


# Data loading 
def load_datafile(fstr):
    xtmp = []
    for line in open(fstr, 'r'):
        ltmps = line[:-2].split(" ")
        xtmp.append([])
        for i in range( len(ltmps) ):
            xtmp[-1].append( float(ltmps[i]) )

    return jnp.array(xtmp)


def load_mnist_data():
    xtrain = load_datafile("dataset/mnist_x_train.txt")
    xtest = load_datafile("dataset/mnist_x_test.txt")
    ytrain_digit = load_datafile("dataset/mnist_y_train.txt")
    ytest_digit = load_datafile("dataset/mnist_y_test.txt")
    
    ytrain_latent = y_latent(ytrain_digit)
    ytest_latent = y_latent(ytest_digit)
    
    return xtrain, ytrain_latent, xtest, ytest_latent


#convert label y to latent variables
def y_latent(y_digit):
	l_to_d = [\
			[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],\
			[0, 0, 1, 1, 0, 0, 1, 1, 0, 0],\
			[0, 0, 0 ,0, 1, 1, 1, 1, 0, 0],\
			[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
	
	l_to_d = jnp.array(l_to_d)
	ld_ones = jnp.ones( jnp.shape(l_to_d) )
	
	#print( jnp.shape(l_to_d), jnp.shape(y_digit) )
	return jnp.dot(l_to_d - 0.5*ld_ones, y_digit.T).T


#latent to output space projection
def generate_Bs(key, hy_params):
	layer_sizes = hy_params['layer_sizes']
	Ny = layer_sizes[-1]
	
	rhoB = hy_params['rhoB']
	latent_size = hy_params['latent_size']
	
	b1key, b2key, gkey = random.split(key, 3)
	B1 = random.normal(b1key, (Ny, latent_size))

	B2tmp = random.normal(b2key, (Ny, latent_size))
	Gtmp = random.bernoulli(gkey, rhoB, (Ny, latent_size))
	Bones = jnp.ones( jnp.shape(B1) )
	
	B2 = jnp.multiply( Gtmp, B1 ) + jnp.multiply( Bones-Gtmp, B2tmp )
	
	return B1, B2


#latent to output space projection (generate three matrices instead of two)
def generate_B3s(key, hy_params):
	layer_sizes = hy_params['layer_sizes']
	Ny = layer_sizes[-1]
	
	rhoB = hy_params['rhoB']
	latent_size = hy_params['latent_size']
	
	b0key, b1key, b2key, g1key, g2key = random.split(key, 5)
	B0 = random.normal(b0key, (Ny, latent_size))
	Bones = jnp.ones( jnp.shape(B0) )
	
	B1tmp = random.normal(b1key, (Ny, latent_size))
	G1tmp = random.bernoulli(g1key, rhoB, (Ny, latent_size))
	B1 = jnp.multiply( G1tmp, B0 ) + jnp.multiply( Bones-G1tmp, B1tmp )
	
	B2tmp = random.normal(b2key, (Ny, latent_size))
	G2tmp = random.bernoulli(g2key, rhoB, (Ny, latent_size))
	B2 = jnp.multiply( G2tmp, B1 ) + jnp.multiply( Bones-G2tmp, B2tmp )
	
	return B0, B1, B2


def check_perm(test_perm):
	if jnp.mean( test_perm == jnp.arange(len(test_perm)) ) > 0.0:
		return True
	else:
		return False


def permute_arrays(key, xtrain, xtest, rho):
	Nx = len(xtrain[0])
	Nperm = int(round( Nx*(1.0-rho) ))
	
	ckey, pkey = random.split(key)
	
	perm_idxs = random.choice(ckey, Nx, shape=(Nperm,), replace=False) 

	# choose a permutation that fully shuffle indices
	test_perm = random.permutation(pkey, jnp.arange(Nperm))
	while check_perm(test_perm):
		pkey, key = random.split(pkey)
		test_perm = random.permutation(pkey, jnp.arange(Nperm))
	
	#permute the training and test arrays with the same key
	permed_subarrays = random.permutation(pkey, xtrain[:,perm_idxs].T)
	xtrain = xtrain.at[:,perm_idxs].set( permed_subarrays.T )
	
	permed_subarrays = random.permutation(pkey, xtest[:,perm_idxs].T)
	xtest = xtest.at[:,perm_idxs].set( permed_subarrays.T )

	return xtrain, xtest


# input: permute the input x for the previous task
# randomly project latent to output space with Btmp
def generate_tasks_xperm(key, xtrain, lt_train, xtest, lt_test, Btmp, hy_params):
	new_xtrain, new_xtest = permute_arrays(key, xtrain, xtest, hy_params['rhoA'])
	ytrain = jnp.dot(lt_train, Btmp.T)
	ytest = jnp.dot(lt_test, Btmp.T)
	
	#print( jnp.shape(new_xtrain), jnp.shape(ytrain), jnp.shape(new_xtest), jnp.shape(ytest) )
	return new_xtrain, ytrain, new_xtest, ytest

