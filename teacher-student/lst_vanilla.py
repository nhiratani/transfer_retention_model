#
# Model of tranfer learning and catastrophic forgetting
#
# Linear student-teacher model with context-dependent gating of synaptic plasticity
#

import sys
from math import *
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from lst_model import generate_tasks, calc_dW, fnorm2

import matplotlib.pyplot as plt

def run_exp(params):
	num_epochs = params['num_epochs']
	learning_rate = params['learning_rate']
	
	key = random.PRNGKey(params['ik'])
	key, abkey = random.split(key, num=2)
	
	Aseq, Bseq = generate_tasks(abkey, params)	
	A1 = Aseq[0]; B1 = Bseq[0];
	A2 = Aseq[1]; B2 = Bseq[1];
	
	Nx = params['Nx']; Ny = params['Ny']
	W = jnp.zeros((Ny, Nx))
	
	errors = np.zeros((2, 2*num_epochs))
	for t in range( 2*num_epochs ):
		if t < num_epochs:
			W = W - learning_rate*calc_dW(W, A1, B1)
		else:
			W = W - learning_rate*calc_dW(W, A2, B2)
		errors[0,t] = fnorm2(B1 - jnp.dot(W, A1))/Ny
		errors[1,t] = fnorm2(B2 - jnp.dot(W, A2))/Ny

	return errors


def simul(params):
	fstr = 'data/tlcf1_lst2_vanilla_errors_Nx' + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
	 + '_nep' + str(params['num_epochs']) + '_rhoA' + str(params['rhoA']) + '_rhoB' + str(params['rhoB'])+ '_ikm' + str(params['ikmax']) + ".txt"
	fw = open(fstr, 'w')

	num_epochs = params['num_epochs']
	for ik in range(params['ikmax']):
		params['ik'] = ik
		errs = run_exp(params)
		fw.write( str(ik) + " " + str(errs[0,num_epochs-1]) + " " + str(errs[1,num_epochs-1]) + " " + str(errs[0, 2*num_epochs-1]) + " " + str(errs[1, 2*num_epochs-1]) + "\n" )


def simul_set1(params):
	rhoAs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	rhoBs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	
	rAlen = len(rhoAs)
	rBlen = len(rhoBs)
	for raidx in range(rAlen):
		params['rhoA'] = rhoAs[raidx]
		for rbidx in range(rBlen):
			params['rhoB'] = rhoBs[rbidx]
			simul(params)


def simul_set2(params):
	rhoAs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	rhoBs = [0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
	
	rAlen = len(rhoAs)
	rBlen = len(rhoBs)
	for raidx in range(rAlen):
		params['rhoA'] = rhoAs[raidx]
		for rbidx in range(rBlen):
			params['rhoB'] = rhoBs[rbidx]
			simul(params)	


def simul_trajectory(params):
	params['rhoA'] = 0.8
	params['rhoB'] = 0.8
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	T = params['num_epochs']
	t1s = range(0, T)
	t2s = range(T-1, 2*T)
	
	for ik in range(params['ikmax']):
		params['ik'] = ik
		errs = run_exp(params)
		
		svfg1 = plt.figure()
		plt.plot(t1s, errs[0, :T], color='C0', lw=2.0)
		plt.plot(t1s, errs[1, :T], color='C1', ls='--', lw=2.0)
		
		plt.plot(t2s, errs[0, T-1:], color='C0', ls='--', lw=2.0)
		plt.plot(t2s, errs[1, T-1:], color='C1', lw=2.0)
		plt.ylim(-0.01, 1.0)
		plt.show()
		svfg1.savefig("fig_tlcf1_lst2_vanilla_trajectory_Nx" + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
					 + "_rhoA" + str(params['rhoA']) + "_rhoA" + str(params['rhoA']) + '_nep' + str(params['num_epochs'])\
					 + "_ik" + str(params['ik']) + ".pdf")
		
	

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
	'num_epochs': 100, #number of epochs
	'ikmax': ikmax, #simulation id
	}

	#simul_set1(params)
	#simul_set2(params)
	simul_trajectory(params)
	
		
		
