#
# Model of tranfer learning and catastrophic forgetting
#
# Fashion MNIST Dataset
#
# Permutation of input and output
#
# Learning with context-dependent gating (random/adaptive)
#

import sys
from math import *
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from tlcf1_pm1_model import init_network_params, init_gates, g_loss, g_update, get_adaptive_g
from tlcf1_pm1_data import load_mnist_data, generate_tasks_xperm, generate_Bs

#import matplotlib.pyplot as plt


def run_exp(x0train, lt_train, x0test, lt_test, hy_params):
	key = random.PRNGKey( hy_params['ik'] )
	bkey, d1key, d2key, wkey, g1key, g2key, gckey = random.split(key, 7)
	
	fstr = 'data/tlcf1_pm1_gated_perm1_accuracy_lr' + str(hy_params['learning_rate']) + '_mbs' + str(hy_params['batch_size'])\
	 + '_alpha' + str(hy_params['alpha']) + '_gtype_' + str(hy_params['gtype']) + '_rhoA' + str(hy_params['rhoA']) + \
	 '_rhoB' + str(hy_params['rhoB']) + '_ik' + str(ik) + ".txt"
	fw = open(fstr, 'w')
	
	B1, B2 = generate_Bs(bkey, hy_params)
	x1train, y1train, x1test, y1test = generate_tasks_xperm(d1key, x0train, lt_train, x0test, lt_test, B1, hy_params)
	x2train, y2train, x2test, y2test = generate_tasks_xperm(d2key, x1train, lt_train, x1test, lt_test, B2, hy_params)

	params = init_network_params(hy_params['layer_sizes'], wkey)
	g1 = init_gates(hy_params, g1key)
	g2 = init_gates(hy_params, g2key)

	sample_size = len(x1train); 
	num_epochs = hy_params['num_epochs']
	batch_size = hy_params['batch_size']
	mb_iter = sample_size//batch_size
	
	test_loss = np.zeros((2, num_epochs*2))
	for epoch in range(2*num_epochs):
		test_loss[0, epoch] = g_loss(params, g1, x1test, y1test)
		test_loss[1, epoch] = g_loss(params, g2, x2test, y2test)
	
		if epoch == num_epochs and hy_params['gtype'] == 'ada':
			probe_loss = g_loss(params, g1, x2test, y2test) # loss on task 2 under g1
			base_loss = test_loss[1, 0]
			min_loss = test_loss[0, epoch]
			if base_loss > min_loss:
				rho_g = np.clip( (base_loss - probe_loss)/(base_loss - min_loss), 0.0, 1.0 )
			else:
				rho_g = np.clip( (base_loss - probe_loss)/base_loss, 0.0, 1.0 )
			g2 = get_adaptive_g(g1, g2, rho_g, gckey)
			test_loss[1, epoch] = g_loss(params, g2, x2test, y2test) #error under g2
			print( rho_g, base_loss, probe_loss, test_loss[1, epoch] )
			
		train_loss = 0.0		
		for t in range(mb_iter):
			if epoch < num_epochs:
				xmb = x1train[t*batch_size:(t+1)*batch_size]
				ymb = y1train[t*batch_size:(t+1)*batch_size]
			else:
				xmb = x2train[t*batch_size:(t+1)*batch_size]
				ymb = y2train[t*batch_size:(t+1)*batch_size]
			
			if epoch < num_epochs:
				params = g_update(params, g1, xmb, ymb, hy_params['learning_rate'])
				train_loss += (1.0/mb_iter)*g_loss(params, g1, xmb, ymb)
			else:
				params = g_update(params, g2, xmb, ymb, hy_params['learning_rate'])
				train_loss += (1.0/mb_iter)*g_loss(params, g2, xmb, ymb)
			
		fw.write( str(epoch) + " " + str(test_loss[0, epoch]) + " " + str(test_loss[1, epoch]) + " " + str(train_loss) + "\n" )

		

if __name__ == "__main__":
	stdins = sys.argv # standard inputs
	
	learning_rate = float(stdins[1]) # learning_rate
	rhoA = float(stdins[2]) # input similarity
	rhoB = float(stdins[3]) # output similarity 
	alpha = float(stdins[4]) # activity sparsity
	gtype = str(stdins[5]) # activity sparsity
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
	'alpha': alpha, #activity sparsity
	'gtype': gtype, #type of gating (rnd or ada)
	'ik': 0 #simulation id
	}
	
	print( hy_params['layer_sizes'] )
	
	x0train, lt_train, x0test, lt_test = load_mnist_data()
	for ik in range(ikmax):
		hy_params['ik'] = ik
		run_exp(x0train, lt_train, x0test, lt_test, hy_params)

		
