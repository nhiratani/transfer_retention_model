#
# Model of tranfer learning and catastrophic forgetting
#
# Linear student-teacher model
#
# Vanilla model, plotting
#

import sys
from math import *
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

import matplotlib.pyplot as plt
from matplotlib import cm

clrs = []
cnum = 5
for cidx in range(cnum):
	clrs.append( cm.viridis( (0.5+cidx)/cnum ) )


def calc_eTF_eRT_vanilla_mf(params):
	rhoA = params['rhoA']; rhoB = params['rhoB']
	eTF = rhoA*(2*rhoB - rhoA)
	eRT = 1.0 - rhoA*rhoA*(rhoA*rhoA - 2*rhoA*rhoB + 1)
	return eTF, eRT


# Plot transfer and retention performance as a function of task similarity
def plot_simul_set1(params):
	rhoAs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	
	rhoBs = [0.92, 0.94, 0.96, 0.98, 1.0] # [0.2, 0.4, 0.6, 0.8, 1.0] # 

	rAlen = len(rhoAs)
	rBlen = len(rhoBs)
	
	num_epochs = params['num_epochs']
	ikmax = params['ikmax'] 
	
	eTFs = np.zeros((rAlen, rBlen, ikmax))
	eRTs = np.zeros((rAlen, rBlen, ikmax))
	
	eTF_means = np.zeros((rAlen, rBlen)); eTF_stds = np.zeros((rAlen, rBlen))
	eRT_means = np.zeros((rAlen, rBlen)); eRT_stds = np.zeros((rAlen, rBlen))
	for raidx in range(rAlen):
		params['rhoA'] = rhoAs[raidx]
		for rbidx in range(rBlen):
			params['rhoB'] = rhoBs[rbidx]
			
			fstr = 'data/tlcf1_lst2_vanilla_errors_Nx' + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
			+ '_nep' + str(params['num_epochs']) + '_rhoA' + str(params['rhoA']) + '_rhoB' + str(params['rhoB'])+ '_ikm' + str(params['ikmax']) + ".txt"

			lidx = 0
			for line in open(fstr, 'r'):
				ltmps = line[:-1].split(" ")
				eTFs[raidx, rbidx, lidx] = 1.0 - float( ltmps[2] )
				eRTs[raidx, rbidx, lidx] = 1.0 - float( ltmps[3] )
				lidx += 1
			
			eTF_means[raidx, rbidx] = np.mean(eTFs[raidx, rbidx, :])
			eTF_stds[raidx, rbidx] = np.std(eTFs[raidx, rbidx, :])
			
			eRT_means[raidx, rbidx] = np.mean(eRTs[raidx, rbidx, :])
			eRT_stds[raidx, rbidx] = np.std(eRTs[raidx, rbidx, :])
			
	
	rhoA_mfs = np.arange(0.0, 1.005, 0.01)
	raflen = len(rhoA_mfs)
	eTF_mfs = np.zeros((rBlen, raflen))
	eRT_mfs = np.zeros((rBlen, raflen))
	
	for rbidx in range(rBlen):
		params['rhoB'] = rhoBs[rbidx]
		for raidx in range(raflen):
			params['rhoA'] = rhoA_mfs[raidx]
			eTFtmp, eRTtmp = calc_eTF_eRT_vanilla_mf(params)
			eTF_mfs[rbidx, raidx] = eTFtmp
			eRT_mfs[rbidx, raidx] = eRTtmp
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg1 = plt.figure()
	for rbidx in range(rBlen):
		plt.errorbar(rhoAs, eTF_means[:, rbidx], eTF_stds[:, rbidx], color=clrs[rbidx], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
		plt.plot(rhoA_mfs, eTF_mfs[rbidx], '-', color=clrs[rbidx])
	plt.show()
	svfg1.savefig("fig_tlcf1_lst2_vanilla_simul_set2_eTF_Nx" + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
					+ '_nep' + str(params['num_epochs']) + "_ikm" + str(params['ikmax']) + ".pdf")
		
	svfg2 = plt.figure()
	for rbidx in range(rBlen):
		plt.errorbar(rhoAs, eRT_means[:, rbidx], eRT_stds[:, rbidx], color=clrs[rbidx], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
		plt.plot(rhoA_mfs, eRT_mfs[rbidx], '-', color=clrs[rbidx])
	plt.show()
	svfg2.savefig("fig_tlcf1_lst2_vanilla_simul_set2_eRT_Nx" + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
					+ '_nep' + str(params['num_epochs']) + "_ikm" + str(params['ikmax']) + ".pdf")


if __name__ == "__main__":
	stdins = sys.argv # standard inputs

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
	'ikmax': 10, #simulation id
	}

	plot_simul_set1(params)

	
		
		
