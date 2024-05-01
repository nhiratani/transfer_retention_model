#
# Model of tranfer learning and catastrophic forgetting
#
# Linear student-teacher model 
#
# with weight regularization in the Fisher information metric, but with diagonal approximation
#

import sys
from math import *
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from tlcf1_lst2_model import generate_tasks, calc_dW, calc_dW_wn, calc_dW_cwn_diag, fnorm2

	
def run_exp_vnl_cwn_diag(params):
	num_epochs = params['num_epochs']
	learning_rate = params['learning_rate']
	
	key = random.PRNGKey(params['ik'])
	key, abkey = random.split(key, num=2)
	
	Aseq, Bseq = generate_tasks(abkey, params)	
	A1 = Aseq[0]; B1 = Bseq[0];
	A2 = Aseq[1]; B2 = Bseq[1];
	Dm1 = jnp.diag( jnp.diag( jnp.dot(A1, A1.T) ) )
	
	Nx = params['Nx']; Ny = params['Ny']
	W = jnp.zeros((Ny, Nx))
	W0 = jnp.zeros((Ny, Nx))
	
	lmbd = params['lmbd']; lmbd_c = params['lmbd_c']
	errors = np.zeros((2, 2*num_epochs))
	for t in range( 2*num_epochs ):
		if t == num_epochs:
			W1 = jnp.zeros((Ny, Nx))
			W1 = W1.at[:,:].set( W[:,:] )
		if t < num_epochs:
			W = W - learning_rate*calc_dW(W, A1, B1)
		else:
			#W = W - learning_rate*calc_dW_cwn_diag(W, A2, B2, W1, Dm1, lmbd_c)
			W = W - learning_rate*calc_dW_cwn_diag(W, A2, B2, W1, Dm1, lmbd)

		errors[0,t] = fnorm2(B1 - jnp.dot(W, A1))/Ny
		errors[1,t] = fnorm2(B2 - jnp.dot(W, A2))/Ny

	return errors
	

def run_exp_cwn_diag(params):
	num_epochs = params['num_epochs']
	learning_rate = params['learning_rate']
	
	key = random.PRNGKey(params['ik'])
	key, abkey = random.split(key, num=2)
	
	Aseq, Bseq = generate_tasks(abkey, params)	
	A1 = Aseq[0]; B1 = Bseq[0];
	A2 = Aseq[1]; B2 = Bseq[1];
	Dm1 = jnp.diag( jnp.diag( jnp.dot(A1, A1.T) ) )
	
	Nx = params['Nx']; Ny = params['Ny']; 
	W = jnp.zeros((Ny, Nx))
	W0 = jnp.zeros((Ny, Nx))
	
	Ns = params['Ns']
	a0key, key = random.split(key, num=2)
	A0 = (1.0/sqrt(Ns))*random.normal(a0key, (Nx, Ns))
	Dm0 = jnp.diag( jnp.diag( jnp.dot(A0, A0.T) ) )
	
	lmbd = params['lmbd']; lmbd_c = params['lmbd_c']
	errors = np.zeros((2, 2*num_epochs))
	for t in range( 2*num_epochs ):
		if t == num_epochs:
			W1 = jnp.zeros((Ny, Nx))
			W1 = W1.at[:,:].set( W[:,:] )
		if t < num_epochs:
			W = W - learning_rate*calc_dW_cwn_diag(W, A1, B1, W0, Dm0, lmbd)
		else:
			W = W - learning_rate*calc_dW_cwn_diag(W, A2, B2, W1, Dm1, lmbd)
		errors[0,t] = fnorm2(B1 - jnp.dot(W, A1))/Ny
		errors[1,t] = fnorm2(B2 - jnp.dot(W, A2))/Ny

	return errors


def simul(params):
	fstr = 'data/tlcf1_lst2_cwn_diag_errors_Nx' + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
	 + '_nep' + str(params['num_epochs']) + '_rhoA' + str(params['rhoA']) + '_rhoB' + str(params['rhoB']) + '_gm' + str(params['gm']) + '_ikm' + str(params['ikmax']) + ".txt"
	fw = open(fstr, 'w')

	num_epochs = params['num_epochs']
	for ik in range(params['ikmax']):
		params['ik'] = ik
		#errs = run_exp_vnl_cwn_diag(params)
		errs = run_exp_cwn_diag(params)
		fw.write( str(ik) + " " + str(errs[0,num_epochs-1]) + " " + str(errs[1,num_epochs-1]) + " " + str(errs[0, 2*num_epochs-1]) + " " + str(errs[1, 2*num_epochs-1]) + "\n" )


def simul_set1(params):
	params['rhoA'] = 1.0
	rhoBs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	gms = [0.2, 0.4, 0.6, 0.8, 1.0]
	
	rBlen = len(rhoBs)
	gmlen = len(gms)
	for rbidx in range(rBlen):
		params['rhoB'] = rhoBs[rbidx]
		for gmidx in range(gmlen):
			params['gm'] = gms[gmidx]
			params['lmbd'] = (params['Nx']/params['Ns'])*( 1.0/params['gm'] - 1.0 )
			params['lmbd_c'] = 1.0/params['gm'] - 1.0 #(params['Nx']/params['Ns'])*( 1.0/params['gm'] - 1.0 )
			simul(params)
			

def simul_set2(params):
	params['rhoB'] = 1.0
	rhoAs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	gms = [0.2, 0.4, 0.6, 0.8, 1.0]
	
	rAlen = len(rhoAs)
	gmlen = len(gms)
	for raidx in range(rAlen):
		params['rhoA'] = rhoAs[raidx]
		for gmidx in range(gmlen):
			params['gm'] = gms[gmidx]
			params['lmbd'] = (params['Nx']/params['Ns'])*( 1.0/params['gm'] - 1.0 )
			params['lmbd_c'] = 1.0/params['gm'] - 1.0 #(params['Nx']/params['Ns'])*( 1.0/params['gm'] - 1.0 )
			simul(params)


def simul_set3(params):
	gms = [0.2, 0.4, 0.6, 0.8, 1.0]
	key = random.PRNGKey( np.random.choice( range(100) ) )
	
	gmlen = len(gms)
	for gmidx in range(gmlen):
		params['gm'] = gms[gmidx]
		params['lmbd'] = (params['Nx']/params['Ns'])*( 1.0/params['gm'] - 1.0 )
		params['lmbd_c'] = 1.0/params['gm'] - 1.0 #(params['Nx']/params['Ns'])*( 1.0/params['gm'] - 1.0 )
		
		fstr = 'data/tlcf1_lst2_cwn_diag_mean_errors_Nx' + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
		+ '_nep' + str(params['num_epochs']) + '_gm' + str(params['gm']) + '_ikm' + str(params['ikmax']) + ".txt"
		fw = open(fstr, 'w')
		
		for ik in range(params['ikmax']):
			akey, bkey, key = random.split(key, num=3)
			params['rhoA'] = random.uniform(akey)
			params['rhoB'] = random.uniform(bkey)
			
			params['ik'] = ik
			#errs = run_exp_vnl_cwn_diag(params)
			errs = run_exp_cwn_diag(params)
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
	'lmbd': 0.0, #amplitude of the weight regularization 
	'ikmax': ikmax, #simulation id
	}

	#simul_set1(params)
	#simul_set2(params)
	simul_set3(params)
	#simul_set4(params)
		
		
