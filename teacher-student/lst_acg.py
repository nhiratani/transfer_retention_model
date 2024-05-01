#
# Model of tranfer learning and catastrophic forgetting
#
# Linear student-teacher model 
#
# with adaptive context-dependent gating of input activity
#

import sys
from math import *
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from lst_model import generate_tasks, calc_dW_cg, fnorm2, generate_g

#import matplotlib.pyplot as plt

def run_exp(params):
	num_epochs = params['num_epochs']
	learning_rate = params['learning_rate']
	alpha = params['alpha']
	
	key = random.PRNGKey(params['ik'])
	key, abkey = random.split(key, num=2)
	
	Aseq, Bseq = generate_tasks(abkey, params)	
	A1 = Aseq[0]; B1 = Bseq[0];
	A2 = Aseq[1]; B2 = Bseq[1];
	
	key, g1key, g2key, gckey = random.split(key, num=4)
	g1 = generate_g(g1key, params)
	#g2 = generate_g(g2key, params)
	
	DA1 = jnp.dot( jnp.diag(g1), A1 )
	D1A2 = jnp.dot( jnp.diag(g1), A2 )
	#DA2 = jnp.dot( jnp.diag(g2), A2 )
	
	Nx = params['Nx']; Ny = params['Ny']
	W = jnp.zeros((Ny, Nx))
	
	errors = np.zeros((2, 2*num_epochs))
	for t in range( 2*num_epochs ):
		if t == num_epochs:
			probe_error = fnorm2(B2 - jnp.dot(W, D1A2))/Ny
			base_error = fnorm2(B2)/Ny
			#rho_g = np.clip( (base_error - probe_error)/( alpha*(2-alpha) ), 0.0, 1.0 )
			rho_g = np.clip( base_error - probe_error, 0.0, 1.0 )
			
			if params['ik'] == 0:
				rho_g_theory = np.clip( params['rhoA']*(2*params['rhoB'] - params['rhoA']), 0.0, 1.0 )
				print( params['rhoA'], params['rhoB'], alpha, base_error, probe_error, rho_g, rho_g_theory)
			
			g2tmp = generate_g(g2key, params)
			gg = random.bernoulli(gckey, rho_g, (Nx,))
			Nxones = jnp.ones((Nx))
			
			g2 = jnp.multiply(gg, g1) + jnp.multiply(Nxones - gg, g2tmp)
			DA2 = jnp.dot( jnp.diag(g2), A2 )
			
			#retrospective update of the transfer error
			errors[1,t-1] = fnorm2(B2 - jnp.dot(W, DA2))/Ny 
			
		if t < num_epochs:
			W = W - learning_rate*calc_dW_cg(W, DA1, B1)
		else:
			W = W - learning_rate*calc_dW_cg(W, DA2, B2)
			
		errors[0,t] = fnorm2(B1 - jnp.dot(W, DA1))/Ny
		if t < num_epochs:
			errors[1,t] = fnorm2(B2 - jnp.dot(W, D1A2))/Ny
		else:
			errors[1,t] = fnorm2(B2 - jnp.dot(W, DA2))/Ny

	return errors


def simul(params):
	fstr = 'data/tlcf1_lst2_acg_errors_Nx' + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
	 + '_nep' + str(params['num_epochs']) + '_rhoA' + str(params['rhoA']) + '_rhoB' + str(params['rhoB']) + '_alpha' + str(params['alpha']) + '_ikm' + str(params['ikmax']) + ".txt"
	fw = open(fstr, 'w')

	num_epochs = params['num_epochs']
	for ik in range(params['ikmax']):
		params['ik'] = ik
		errs = run_exp(params)
		fw.write( str(ik) + " " + str(errs[0, num_epochs-1]) + " " + str(errs[1, num_epochs-1]) + " " + str(errs[0, 2*num_epochs-1]) + " " + str(errs[1, 2*num_epochs-1]) + "\n" )
		
		#if np.random.random() < 0.01:
		#	clrs = ['C0', 'C1']
		#	for q in range(2):
		#		plt.plot(errs[q], color=clrs[q])
		#	plt.show()


def simul_set1(params):
	params['rhoB'] = 1.0
	rhoAs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
	
	alen = len(alphas)
	rAlen = len(rhoAs)
	for alidx in range(alen):
		params['alpha'] = alphas[alidx]
		for raidx in range(rAlen):
			params['rhoA'] = rhoAs[raidx]
			simul(params)


def simul_set2(params):
	alphas = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
	key = random.PRNGKey( np.random.choice( range(100) ) )
	
	alen = len(alphas)
	for alidx in range(alen):
		params['alpha'] = alphas[alidx]
		fstr = 'data/tlcf1_lst2_acg_mean_errors_Nx' + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
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
	'learning_rate': 0.001, #learning rate
	'num_epochs': 500, #number of epochs
	'alpha': 0.0, #sparsity of gating
	'ikmax': ikmax, #simulation id
	}

	#simul_set1(params)
	simul_set2(params)
	
		
		
