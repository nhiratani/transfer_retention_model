#
# Model of tranfer learning and catastrophic forgetting
#
# Linear student-teacher model
#
# Adaptive context-dependent gating model
#

import sys
from math import *
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection

clrs = []
cnum = 5
for cidx in range(cnum):
	clrs.append( cm.viridis( (0.5+cidx)/cnum ) )


def calc_eTF_eRT_cg_mf(params):
	rhoA = params['rhoA']; rhoB = params['rhoB']; alpha = params['alpha']
	eTF = alpha*rhoA*(2*rhoB - alpha*rhoA)
	eRT = 1.0 - alpha*rhoA*alpha*rhoA*(alpha*rhoA*alpha*rhoA - 2*alpha*rhoA*rhoB + 1)
	return eTF, eRT
	
	
def calc_eTF_eRT_adacg_mf(params):
	rhoA = params['rhoA']; rhoB = params['rhoB']; alpha = params['alpha']
	
	rhog = np.clip( rhoA*(2*rhoB - rhoA), 0, 1 )
	alphat = rhog + (1 - rhog)*alpha
	
	eTF = alphat*rhoA*(2*rhoB - alphat*rhoA)
	eRT = 1.0 - alphat*rhoA*alphat*rhoA*(alphat*rhoA*alphat*rhoA - 2*alphat*rhoA*rhoB + 1)
	return eTF, eRT


def calc_mean_eTF_eRT_cg_mf(params):
	alpha = params['alpha']
	mean_eTF = alpha/2 - alpha*alpha/3
	mean_eRT = 1.0 - (alpha*alpha*alpha*alpha/5 - alpha*alpha*alpha/4 + alpha*alpha/3)
	return mean_eTF, mean_eRT


def calc_mean_eTF_eRT_adacg_mf(params):
	alpha = params['alpha']
	
	#Nmax = 10000
	#mean_eTF = 0.0; mean_eRT = 0.0
	#for idx in range(Nmax):
	#	rhoA, rhoB = np.random.uniform(size=2)
	#	rhog = np.clip( rhoA*(2*rhoB - rhoA), 0, 1 )
	#	alphat = rhog + (1 - rhog)*alpha
	#	mean_eTF += ( alphat/2 - alphat*alphat/3 )/Nmax
	#	mean_eRT += ( 1.0 - alphat*rhoA*alphat*rhoA*(alphat*rhoA*alphat*rhoA - 2*alphat*rhoA*rhoB + 1) )/Nmax
	
	mean_eTF = (149 + 451*alpha - 390*alpha*alpha)/1260
	mean_eRT = (341974 + 63729*alpha - 220296*alpha*alpha + 210551*alpha*alpha*alpha -137700*alpha*alpha*alpha*alpha)/360360
	return mean_eTF, mean_eRT


def plot_simul_set1(params):
	params['rhoB'] = 1.0
	
	rhoAs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
	alen = len(alphas)
	rAlen = len(rhoAs)
	
	num_epochs = params['num_epochs']
	ikmax = params['ikmax'] 
	
	eTFs = np.zeros((alen, rAlen, ikmax))
	eRTs = np.zeros((alen, rAlen, ikmax))
	
	eTF_means = np.zeros((alen, rAlen)); eTF_stds = np.zeros((alen, rAlen))
	eRT_means = np.zeros((alen, rAlen)); eRT_stds = np.zeros((alen, rAlen))

	for alidx in range(alen):
		params['alpha'] = alphas[alidx]
		for raidx in range(rAlen):
			params['rhoA'] = rhoAs[raidx]
			
			fstr = 'data/tlcf1_lst2_acg_errors_Nx' + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
			+ '_nep' + str(params['num_epochs']) + '_rhoA' + str(params['rhoA']) + '_rhoB' + str(params['rhoB']) + '_alpha' + str(params['alpha'])+ '_ikm' + str(params['ikmax']) + ".txt"

			lidx = 0
			for line in open(fstr, 'r'):
				ltmps = line[:-1].split(" ")
				eTFs[alidx, raidx, lidx] = 1.0 - float( ltmps[2] )
				eRTs[alidx, raidx, lidx] = 1.0 - float( ltmps[3] )
				lidx += 1
			
			eTF_means[alidx, raidx] = np.mean(eTFs[alidx, raidx, :])
			eTF_stds[alidx, raidx] = np.std(eTFs[alidx, raidx, :])
			
			eRT_means[alidx, raidx] = np.mean(eRTs[alidx, raidx, :])
			eRT_stds[alidx, raidx] = np.std(eRTs[alidx, raidx, :])
			
	
	alpha_mfs = np.arange(0.01, 1.005, 0.01)
	aflen = len(alpha_mfs)
	eTF_nonada_mfs = np.zeros((rAlen, aflen))
	eRT_nonada_mfs = np.zeros((rAlen, aflen))
	
	eTF_ada_mfs = np.zeros((rAlen, aflen))
	eRT_ada_mfs = np.zeros((rAlen, aflen))
	
	for raidx in range(rAlen):
		params['rhoA'] = rhoAs[raidx]
		for afidx in range(aflen):
			params['alpha'] = alpha_mfs[afidx]
			
			eTFtmp, eRTtmp = calc_eTF_eRT_adacg_mf(params)
			eTF_ada_mfs[raidx, afidx] = eTFtmp
			eRT_ada_mfs[raidx, afidx] = eRTtmp
			
			eTFtmp, eRTtmp = calc_eTF_eRT_cg_mf(params)
			eTF_nonada_mfs[raidx, afidx] = eTFtmp
			eRT_nonada_mfs[raidx, afidx] = eRTtmp
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg1 = plt.figure()
	for raidx in range(1, rAlen, 2):
		plt.errorbar(alphas, eTF_means[:, raidx], eTF_stds[:, raidx], color=clrs[raidx-1], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
		plt.plot(alpha_mfs, eTF_nonada_mfs[raidx], '--', color=clrs[raidx-1])
		plt.plot(alpha_mfs, eTF_ada_mfs[raidx], '-', color=clrs[raidx-1])
	plt.show()
	svfg1.savefig("fig_tlcf1_lst2_acg_simul_set1_eTF_Nx" + str(params['Nx']) + '_rhoB' + str(params['rhoB']) + '_lr' + str(params['learning_rate'])\
					+ '_nep' + str(params['num_epochs']) + "_ikm" + str(params['ikmax']) + ".pdf")
		
	svfg2 = plt.figure()
	for raidx in range(1, rAlen, 2):
		plt.errorbar(alphas, eRT_means[:, raidx], eRT_stds[:, raidx], color=clrs[raidx-1], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
		plt.plot(alpha_mfs, eRT_nonada_mfs[raidx], '--', color=clrs[raidx-1])
		plt.plot(alpha_mfs, eRT_ada_mfs[raidx], '-', color=clrs[raidx-1])
	plt.show()
	svfg2.savefig("fig_tlcf1_lst2_acg_simul_set1_eRT_Nx" + str(params['Nx']) + '_rhoB' + str(params['rhoB']) + '_lr' + str(params['learning_rate'])\
					+ '_nep' + str(params['num_epochs']) + "_ikm" + str(params['ikmax']) + ".pdf")


def plot_simul_set2(params):
	alphas = [0.2, 0.4, 0.6, 0.8, 1.0] # [0.1, 0.2, 0.4, 0.6, 0.8, 1.0] #
	alen = len(alphas)
	
	num_epochs = params['num_epochs']
	ikmax = params['ikmax'] 
	
	eTFs = np.zeros((alen, ikmax))
	eRTs = np.zeros((alen, ikmax))
	
	eTF_means = np.zeros((alen)); eTF_stds = np.zeros((alen))
	eRT_means = np.zeros((alen)); eRT_stds = np.zeros((alen))

	for alidx in range(alen):
		params['alpha'] = alphas[alidx]
		fstr = 'data/tlcf1_lst2_acg_mean_errors_Nx' + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
		+ '_nep' + str(params['num_epochs']) + '_alpha' + str(params['alpha'])+ '_ikm' + str(params['ikmax']) + ".txt"

		lidx = 0
		for line in open(fstr, 'r'):
			ltmps = line[:-1].split(" ")
			eTFs[alidx, lidx] = 1.0 - float( ltmps[2] )
			eRTs[alidx, lidx] = 1.0 - float( ltmps[3] )
			lidx += 1
			
		eTF_means[alidx] = np.mean(eTFs[alidx, :])
		eTF_stds[alidx] = np.std(eTFs[alidx, :])
			
		eRT_means[alidx] = np.mean(eRTs[alidx, :])
		eRT_stds[alidx] = np.std(eRTs[alidx, :])
			
	
	alpha_mfs = np.arange(0.01, 1.005, 0.01)
	aflen = len(alpha_mfs)
	eTF_mfs = np.zeros((aflen)); eRT_mfs = np.zeros((aflen))
	eTF_nonada_mfs = np.zeros((aflen)); eRT_nonada_mfs = np.zeros((aflen))
	
	for afidx in range(aflen):
		params['alpha'] = alpha_mfs[afidx]
		eTFtmp, eRTtmp = calc_mean_eTF_eRT_adacg_mf(params)
		eTF_mfs[afidx] = eTFtmp
		eRT_mfs[afidx] = eRTtmp
		
		eTFtmp, eRTtmp = calc_mean_eTF_eRT_cg_mf(params)
		eTF_nonada_mfs[afidx] = eTFtmp
		eRT_nonada_mfs[afidx] = eRTtmp
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg1 = plt.figure()
	#plt.errorbar(alphas, eTF_means, eTF_stds[:, raidx], color=clrs[raidx], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
	plt.plot(alphas, eTF_means, 'o', color='C0')
	plt.plot(alpha_mfs, eTF_mfs, '-', color='C0')
	plt.plot(alpha_mfs, eTF_nonada_mfs, '--', color='C0')
	
	plt.plot(alphas, eRT_means, 'o', color='C1')
	plt.plot(alpha_mfs, eRT_mfs, '-', color='C1')
	plt.plot(alpha_mfs, eRT_nonada_mfs, '--', color='C1')
	
	plt.show()
	svfg1.savefig("fig_tlcf1_lst2_acg_simul_set3_mean_eTF_eTR_Nx" + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
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
	'num_epochs': 500, #number of epochs
	'ikmax': 10, #simulation id
	}

	plot_simul_set1(params)
	#plot_simul_set2(params)
		
