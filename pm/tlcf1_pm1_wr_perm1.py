#
# Model of tranfer learning and catastrophic forgetting
#
# Fashion MNIST Dataset
#
# Permutation of input and output
#
# A model with weight regularization
#

import sys
from math import *
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from tlcf1_pm1_model import init_network_params, copy_params_value, loss, update_wr_euc, update_wr_diag, update_wr_fim, calc_coeff_params_diag, calc_coeff_params_fim
from tlcf1_pm1_data import load_mnist_data, generate_tasks_xperm, generate_Bs, generate_B3s

import matplotlib.pyplot as plt


def run_exp(x0train, lt_train, x0test, lt_test, hy_params):
	key = random.PRNGKey( hy_params['ik'] )
	bkey, d1key, d2key, wkey = random.split(key, 4)
	
	fstr = 'data/tlcf1_pm1_wr_perm1_accuracy_lr' + str(hy_params['learning_rate']) + '_mbs' + str(hy_params['batch_size'])\
	 + '_rhoA' + str(hy_params['rhoA']) + '_rhoB' + str(hy_params['rhoB'])  + '_lmbd' + str(hy_params['lmbd'])\
	 + '_wrtype_' + hy_params['wrtype'] + '_ik' + str(ik) + ".txt"
	fw = open(fstr, 'w')
	
	B0, B1, B2 = generate_B3s(bkey, hy_params)
	#B1, B2 = generate_Bs(bkey, hy_params)
	
	y0train = jnp.dot(lt_train, B0.T)
	x1train, y1train, x1test, y1test = generate_tasks_xperm(d1key, x0train, lt_train, x0test, lt_test, B1, hy_params)
	x2train, y2train, x2test, y2test = generate_tasks_xperm(d2key, x1train, lt_train, x1test, lt_test, B2, hy_params)

	params = init_network_params(hy_params['layer_sizes'], wkey)

	sample_size = len(x1train); 
	num_epochs = hy_params['num_epochs']
	batch_size = hy_params['batch_size']
	wrtype = hy_params['wrtype']
	mb_iter = sample_size//batch_size
	
	test_loss = np.zeros((2, num_epochs*2))
	for epoch in range(2*num_epochs):
		test_loss[0, epoch] = loss(params, x1test, y1test)
		test_loss[1, epoch] = loss(params, x2test, y2test)
		
		if epoch == 0 or epoch == num_epochs:
			target_params = copy_params_value(params)
			if wrtype == 'diag':
				if epoch == 0:	
					coeff_params = calc_coeff_params_diag(params, x0train)
				else:
					coeff_params = calc_coeff_params_diag(params, x1train)
				
				print( jnp.min(coeff_params[0]), jnp.max(coeff_params[0]), jnp.min(coeff_params[1]), jnp.max(coeff_params[1]) )
				
			if wrtype == 'fim':
				if epoch == 0:	
					xxt, dyxt, phixt, phiphit, dphi_mat, zws = calc_coeff_params_fim(params, x0train, y0train)
				else:
					xxt, dyxt, phixt, phiphit, dphi_mat, zws = calc_coeff_params_fim(params, x1train, y1train)
				print( zws[0], zws[1], jnp.max(xxt), jnp.max(dyxt), jnp.max(phixt), jnp.max(phiphit) )
				
		train_loss = 0.0		
		for t in range(mb_iter):
			if epoch < num_epochs:
				xmb = x1train[t*batch_size:(t+1)*batch_size]
				ymb = y1train[t*batch_size:(t+1)*batch_size]
			else:
				xmb = x2train[t*batch_size:(t+1)*batch_size]
				ymb = y2train[t*batch_size:(t+1)*batch_size]
			if wrtype == 'euc':
				params = update_wr_euc(params, target_params, xmb, ymb, hy_params['learning_rate'], hy_params['lmbd'])
			elif wrtype == 'diag':
				params = update_wr_diag(params, target_params, coeff_params, xmb, ymb, hy_params['learning_rate'], hy_params['lmbd'])
			elif wrtype == 'fim':
				params = update_wr_fim(params, target_params, xxt, dyxt, phixt, phiphit, dphi_mat, zws, xmb, ymb, hy_params['learning_rate'], hy_params['lmbd'])

			train_loss += (1.0/mb_iter)*loss(params, xmb, ymb)
		fw.write( str(epoch) + " " + str(test_loss[0, epoch]) + " " + str(test_loss[1, epoch]) + " " + str(train_loss) + "\n" )

		
if __name__ == "__main__":
	stdins = sys.argv # standard inputs
	
	learning_rate = float(stdins[1]) # learning_rate
	rhoA = float(stdins[2]) # input similarity
	rhoB = float(stdins[3]) # output similarity 
	lmbd = float(stdins[4]) # amplitude of the weight regularization
	wrtype = str(stdins[5]) # type of weight regularization
	ikmax = int(stdins[6]) # the number of simulations

	#hyper parameters
	hy_params = {
	'layer_sizes': [784, 1500, 10], #layer sizes
	'latent_size': 4, #the dimensionality of the latent space
	'learning_rate': learning_rate, #learning rate
	'num_epochs': 100, #100, #number of epochs per task
	'batch_size': 300, #batch size
	'rhoA': rhoA, #3000, #input layer width
	'rhoB': rhoB, #output layer width
	'lmbd': lmbd, #amplitude of the weight regularization
	'wrtype': wrtype, # type of weight regularization: 'euc'/'fim'/'diag'
	'ik': 0 #simulation id
	}
	
	print( hy_params['layer_sizes'] )
	
	x0train, lt_train, x0test, lt_test = load_mnist_data()
	for ik in range(ikmax):
		hy_params['ik'] = ik
		run_exp(x0train, lt_train, x0test, lt_test, hy_params)

		
