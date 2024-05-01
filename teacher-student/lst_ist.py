#
# Model of tranfer learning and catastrophic forgetting
#
# Linear student-teacher model 
#
# with Input-soft thresholding
#

import sys
from math import *
import numpy as np
import jax.numpy as jnp
import jax.scipy.special as jscisp
from jax import grad, jit, vmap
from jax import random

from lst_model import generate_tasks, calc_dW_ist, fnorm2, calc_error_ist

import matplotlib.pyplot as plt
	

def run_exp(params):
	num_epochs = params['num_epochs']
	learning_rate = params['learning_rate']
	batch_size = params['batch_size']
	
	key = random.PRNGKey(params['ik'])
	key, abkey = random.split(key, num=2)
	
	Aseq, Bseq = generate_tasks(abkey, params)	
	A1 = Aseq[0]; B1 = Bseq[0];
	A2 = Aseq[1]; B2 = Bseq[1];

	Nx = params['Nx']; Ny = params['Ny']
	W = jnp.zeros((Ny, Nx))
	h = jnp.sqrt(2)*jscisp.erfinv( 1.0 - params['alpha'] )
	
	errors = np.zeros((2, 2*num_epochs))
	for t in range( 2*num_epochs ):
		key, wkey, ekey = random.split(key, num=3)

		S = random.normal(wkey, (params['Ns'], batch_size))	
		if t < num_epochs:
			W = W - learning_rate*calc_dW_ist(W, A1, B1, S, h, batch_size)
		else:
			W = W - learning_rate*calc_dW_ist(W, A2, B2, S, h, batch_size)
		
		S = random.normal(ekey, (params['Ns'], batch_size))	
		errors[0,t] = calc_error_ist(W, A1, B1, S, h, batch_size)/Ny
		errors[1,t] = calc_error_ist(W, A2, B2, S, h, batch_size)/Ny

	return errors


def simul(params):
	fstr = 'data/tlcf1_lst2_ist_errors_Nx' + str(params['Nx']) + '_lr' + str(params['learning_rate']) + '_bs' + str(params['batch_size'])\
	 + '_nep' + str(params['num_epochs']) + '_rhoA' + str(params['rhoA']) + '_rhoB' + str(params['rhoB']) + '_alpha' + str(params['alpha']) + '_ikm' + str(params['ikmax']) + ".txt"
	fw = open(fstr, 'w')

	num_epochs = params['num_epochs']
	for ik in range(params['ikmax']):
		params['ik'] = ik
		errs = run_exp(params)
		fw.write( str(ik) + " " + str(errs[0,num_epochs-1]) + " " + str(errs[1,num_epochs-1]) + " " + str(errs[0, 2*num_epochs-1]) + " " + str(errs[1, 2*num_epochs-1]) + "\n" )
		
		#if np.random.random() < 0.1:
		#	clrs = ['C0', 'C1']
		#	for q in range(2):
		#		plt.plot(errs[q], color=clrs[q])
		#	plt.show()


def simul_set1(params):
	params['rhoA'] = 1.0
	rhoBs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	
	alen = len(alphas)
	rBlen = len(rhoBs)
	for alidx in range(alen):
		params['alpha'] = alphas[alidx]
		for rbidx in range(rBlen):
			params['rhoB'] = rhoBs[rbidx]
			simul(params)


def simul_set2(params):
	params['rhoB'] = 0.9#1.0
	rhoAs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	
	alen = len(alphas)
	rAlen = len(rhoAs)
	for alidx in range(alen):
		params['alpha'] = alphas[alidx]
		for raidx in range(rAlen):
			params['rhoA'] = rhoAs[raidx]
			simul(params)


def simul_set3(params):
	alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	key = random.PRNGKey( np.random.choice( range(100) ) )
	
	alen = len(alphas)
	for alidx in range(alen):
		params['alpha'] = alphas[alidx]
		fstr = 'data/tlcf1_lst2_ist_mean_errors_Nx' + str(params['Nx']) + '_lr' + str(params['learning_rate']) + '_bs' + str(params['batch_size'])\
		+ '_nep' + str(params['num_epochs']) + '_alpha' + str(params['alpha']) + '_ikm' + str(params['ikmax']) + ".txt"
		fw = open(fstr, 'w')
		
		for ik in range(params['ikmax']):
			akey, bkey, key = random.split(key, num=3)
			params['rhoA'] = random.uniform(akey)
			params['rhoB'] = random.uniform(bkey)
			
			params['ik'] = ik
			errs = run_exp(params)
			num_epochs = params['num_epochs']
			fw.write( str(ik) + " " + str(errs[0,num_epochs-1]) + " " + str(errs[1,num_epochs-1]) + " " + str(errs[0, 2*num_epochs-1]) + " " + str(errs[1, 2*num_epochs-1]) + "\n" )




if __name__ == "__main__":
	stdins = sys.argv # standard inputs

	ikmax = int(stdins[1]) # simulation id

	#env parameters
	params = {
	'Ns': 30, #dimensionality of feature space
	'Nx': 3000, #3000, #input layer width
	'Ny': 10, #output layer width
	'Nsess': 2, #the number of sessions
	'rhoA': 0.0, #task similarity (input)
	'rhoB': 0.0, #task similarity (output)
	'learning_rate': 0.01, #0.001, #learning rate
	'num_epochs': 5000, #number of epochs
	'alpha': 0.0, #sparsity of soft-thresholding
	'batch_size': 10000, #batch size for update (for soft thresholding model)
	'ikmax': ikmax, #simulation id
	}

	#simul_set1(params)
	simul_set2(params)
	#simul_set3(params)
	
		
		
