#
# Model of tranfer learning and catastrophic forgetting
#
# Linear student-teacher model
#
# Vanilla model
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
cnum = 6
for cidx in range(cnum):
	clrs.append( cm.viridis( (0.5+cidx)/cnum ) )



def calc_eTF_eRT_cg_mf(params):
	rhoA = params['rhoA']; rhoB = params['rhoB']; alpha = params['alpha']
	eTF = alpha*rhoA*(2*rhoB - alpha*rhoA)
	eRT = 1.0 - alpha*rhoA*alpha*rhoA*(alpha*rhoA*alpha*rhoA - 2*alpha*rhoA*rhoB + 1)
	return eTF, eRT


def calc_mean_eTF_eRT_cg_mf(params):
	alpha = params['alpha']
	mean_eTF = alpha/2 - alpha*alpha/3
	mean_eRT = 1.0 - (alpha*alpha*alpha*alpha/5 - alpha*alpha*alpha/4 + alpha*alpha/3)
	return mean_eTF, mean_eRT


def plot_simul_set1(params):
	params['rhoA'] = 1.0
	
	rhoBs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
	alen = len(alphas)
	rBlen = len(rhoBs)
	
	num_epochs = params['num_epochs']
	ikmax = params['ikmax'] 
	
	eTFs = np.zeros((alen, rBlen, ikmax))
	eRTs = np.zeros((alen, rBlen, ikmax))
	
	eTF_means = np.zeros((alen, rBlen)); eTF_stds = np.zeros((alen, rBlen))
	eRT_means = np.zeros((alen, rBlen)); eRT_stds = np.zeros((alen, rBlen))

	for alidx in range(alen):
		params['alpha'] = alphas[alidx]
		for rbidx in range(rBlen):
			params['rhoB'] = rhoBs[rbidx]
			
			fstr = 'data/tlcf1_lst2_cg_errors_Nx' + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
			+ '_nep' + str(params['num_epochs']) + '_rhoA' + str(params['rhoA']) + '_rhoB' + str(params['rhoB']) + '_alpha' + str(params['alpha'])+ '_ikm' + str(params['ikmax']) + ".txt"

			lidx = 0
			for line in open(fstr, 'r'):
				ltmps = line[:-1].split(" ")
				eTFs[alidx, rbidx, lidx] = 1.0 - float( ltmps[2] )
				eRTs[alidx, rbidx, lidx] = 1.0 - float( ltmps[3] )
				lidx += 1
			
			eTF_means[alidx, rbidx] = np.mean(eTFs[alidx, rbidx, :])
			eTF_stds[alidx, rbidx] = np.std(eTFs[alidx, rbidx, :])
			
			eRT_means[alidx, rbidx] = np.mean(eRTs[alidx, rbidx, :])
			eRT_stds[alidx, rbidx] = np.std(eRTs[alidx, rbidx, :])
			
	
	alpha_mfs = np.arange(0.0, 1.005, 0.01)
	aflen = len(alpha_mfs)
	eTF_mfs = np.zeros((rBlen, aflen))
	eRT_mfs = np.zeros((rBlen, aflen))
	
	for rbidx in range(rBlen):
		params['rhoB'] = rhoBs[rbidx]
		for afidx in range(aflen):
			params['alpha'] = alpha_mfs[afidx]
			eTFtmp, eRTtmp = calc_eTF_eRT_cg_mf(params)
			eTF_mfs[rbidx, afidx] = eTFtmp
			eRT_mfs[rbidx, afidx] = eRTtmp
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg1 = plt.figure()
	for rbidx in range(rBlen):
		plt.errorbar(alphas, eTF_means[:, rbidx], eTF_stds[:, rbidx], color=clrs[rbidx], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
		plt.plot(alpha_mfs, eTF_mfs[rbidx], '-', color=clrs[rbidx])
	plt.show()
	svfg1.savefig("fig_tlcf1_lst2_cg_simul_set1_eTF_Nx" + str(params['Nx']) + '_rhoA' + str(params['rhoA']) + '_lr' + str(params['learning_rate'])\
					+ '_nep' + str(params['num_epochs']) + "_ikm" + str(params['ikmax']) + ".pdf")
		
	svfg2 = plt.figure()
	for rbidx in range(rBlen):
		plt.errorbar(alphas, eRT_means[:, rbidx], eRT_stds[:, rbidx], color=clrs[rbidx], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
		plt.plot(alpha_mfs, eRT_mfs[rbidx], '-', color=clrs[rbidx])
	plt.show()
	svfg2.savefig("fig_tlcf1_lst2_cg_simul_set1_eRT_Nx" + str(params['Nx']) + '_rhoA' + str(params['rhoA']) + '_lr' + str(params['learning_rate'])\
					+ '_nep' + str(params['num_epochs']) + "_ikm" + str(params['ikmax']) + ".pdf")


def plot_simul_set2(params):
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
			
			fstr = 'data/tlcf1_lst2_cg_errors_Nx' + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
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
			
	
	alpha_mfs = np.arange(0.0, 1.005, 0.01)
	aflen = len(alpha_mfs)
	eTF_mfs = np.zeros((rAlen, aflen))
	eRT_mfs = np.zeros((rAlen, aflen))
	
	for raidx in range(rAlen):
		params['rhoA'] = rhoAs[raidx]
		for afidx in range(aflen):
			params['alpha'] = alpha_mfs[afidx]
			eTFtmp, eRTtmp = calc_eTF_eRT_cg_mf(params)
			eTF_mfs[raidx, afidx] = eTFtmp
			eRT_mfs[raidx, afidx] = eRTtmp
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg1 = plt.figure()
	for raidx in range(rAlen):
		plt.errorbar(alphas, eTF_means[:, raidx], eTF_stds[:, raidx], color=clrs[raidx], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
		plt.plot(alpha_mfs, eTF_mfs[raidx], '-', color=clrs[raidx])
	plt.show()
	svfg1.savefig("fig_tlcf1_lst2_cg_simul_set2_eTF_Nx" + str(params['Nx']) + '_rhoB' + str(params['rhoB']) + '_lr' + str(params['learning_rate'])\
					+ '_nep' + str(params['num_epochs']) + "_ikm" + str(params['ikmax']) + ".pdf")
		
	svfg2 = plt.figure()
	for raidx in range(rAlen):
		plt.errorbar(alphas, eRT_means[:, raidx], eRT_stds[:, raidx], color=clrs[raidx], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
		plt.plot(alpha_mfs, eRT_mfs[raidx], '-', color=clrs[raidx])
	plt.show()
	svfg2.savefig("fig_tlcf1_lst2_cg_simul_set2_eRT_Nx" + str(params['Nx']) + '_rhoB' + str(params['rhoB']) + '_lr' + str(params['learning_rate'])\
					+ '_nep' + str(params['num_epochs']) + "_ikm" + str(params['ikmax']) + ".pdf")


def plot_simul_set3(params):
	alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
	alen = len(alphas)
	
	num_epochs = params['num_epochs']
	ikmax = params['ikmax'] 
	
	eTFs = np.zeros((alen, ikmax))
	eRTs = np.zeros((alen, ikmax))
	
	eTF_means = np.zeros((alen)); eTF_stds = np.zeros((alen))
	eRT_means = np.zeros((alen)); eRT_stds = np.zeros((alen))

	for alidx in range(alen):
		params['alpha'] = alphas[alidx]
		fstr = 'data/tlcf1_lst2_cg_mean_errors_Nx' + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
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
			
	
	alpha_mfs = np.arange(0.0, 1.005, 0.01)
	aflen = len(alpha_mfs)
	eTF_mfs = np.zeros((aflen))
	eRT_mfs = np.zeros((aflen))
	
	for afidx in range(aflen):
		params['alpha'] = alpha_mfs[afidx]
		eTFtmp, eRTtmp = calc_mean_eTF_eRT_cg_mf(params)
		eTF_mfs[afidx] = eTFtmp
		eRT_mfs[afidx] = eRTtmp
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg1 = plt.figure()
	#plt.errorbar(alphas, eTF_means, eTF_stds[:, raidx], color=clrs[raidx], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
	plt.axhline( 1/6.0, color='C0', ls='--' )
	plt.plot(alphas, eTF_means, 'o', color='C0')
	plt.plot(alpha_mfs, eTF_mfs, '-', color='C0')
	
	plt.axhline( 1.0 - (1/5.0 - 1/4.0 + 1/3.0), color='C1', ls='--' )
	plt.plot(alphas, eRT_means, 'o', color='C1')
	plt.plot(alpha_mfs, eRT_mfs, '-', color='C1')
	
	plt.show()
	svfg1.savefig("fig_tlcf1_lst2_cg_simul_set3_mean_eTF_eTR_Nx" + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
					+ '_nep' + str(params['num_epochs']) + "_ikm" + str(params['ikmax']) + ".pdf")
		

def plot_simul_set4(params):
	rhoAs = [0.75]#[0.50, 0.75, 0.75, 0.75, 0.90, 1.00]
	rhoBs = [0.25]#[0.75, 0.25, 0.50, 0.98, 0.98, 0.98]
	alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
	
	alen = len(alphas)
	rlen = len(rhoAs)
	
	num_epochs = params['num_epochs']
	ikmax = params['ikmax'] 
	
	eTFs = np.zeros((rlen, alen, ikmax))
	eRTs = np.zeros((rlen, alen, ikmax))
	
	eTF_means = np.zeros((rlen, alen)); eTF_stds = np.zeros((rlen, alen))
	eRT_means = np.zeros((rlen, alen)); eRT_stds = np.zeros((rlen, alen))

	for ridx in range(rlen):
		params['rhoA'] = rhoAs[ridx]
		params['rhoB'] = rhoBs[ridx]
		
		for alidx in range(alen):
			params['alpha'] = alphas[alidx]
			
			fstr = 'data/tlcf1_lst2_cg_errors_Nx' + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
			+ '_nep' + str(params['num_epochs']) + '_rhoA' + str(params['rhoA']) + '_rhoB' + str(params['rhoB']) + '_alpha' + str(params['alpha'])+ '_ikm' + str(params['ikmax']) + ".txt"

			lidx = 0
			for line in open(fstr, 'r'):
				ltmps = line[:-1].split(" ")
				eTFs[ridx, alidx, lidx] = 1.0 - float( ltmps[2] )
				eRTs[ridx, alidx, lidx] = 1.0 - float( ltmps[3] )
				lidx += 1
			
			eTF_means[ridx, alidx] = np.mean(eTFs[ridx, alidx, :])
			eTF_stds[ridx, alidx] = np.std(eTFs[ridx, alidx, :])
			
			eRT_means[ridx, alidx] = np.mean(eRTs[ridx, alidx, :])
			eRT_stds[ridx, alidx] = np.std(eRTs[ridx, alidx, :])
			
	
	alpha_mfs = np.arange(0.0, 1.005, 0.01)
	aflen = len(alpha_mfs)
	eTF_mfs = np.zeros((rlen, aflen))
	eRT_mfs = np.zeros((rlen, aflen))
	
	for ridx in range(rlen):
		params['rhoA'] = rhoAs[ridx]
		params['rhoB'] = rhoBs[ridx]
		for afidx in range(aflen):
			params['alpha'] = alpha_mfs[afidx]
			eTFtmp, eRTtmp = calc_eTF_eRT_cg_mf(params)
			eTF_mfs[ridx, afidx] = eTFtmp
			eRT_mfs[ridx, afidx] = eRTtmp
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	#svfg1 = plt.figure()
	
	for ridx in range(rlen):
		fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
		
		points = np.array([eTF_mfs[ridx], eRT_mfs[ridx]]).T.reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)
		norm = plt.Normalize(alpha_mfs.min(), alpha_mfs.max())

		lc = LineCollection(segments, cmap='rainbow', norm=norm)
		lc.set_array(alpha_mfs)
		lc.set_linewidth(2)
		line = axs.add_collection(lc)
		
		climit = 5
		clrs = []
		for q in range(climit):
			clrs.append( cm.rainbow( alphas[q] ) )
		for aidx in range(alen):
			axs.errorbar(eTF_means[ridx, aidx], eRT_means[ridx, aidx], xerr=eTF_stds[ridx, aidx], yerr=eRT_stds[ridx, aidx],\
			color=clrs[aidx], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
		
		if min(eTF_mfs[ridx]) >= 0:
			axs.set_xlim(0, 1.25*max(eTF_mfs[ridx]))
		else:
			axs.set_xlim(2.0*min(eTF_mfs[ridx]), 2.0*max(eTF_mfs[ridx]))

		if min(eRT_mfs[ridx]) > 0.9:
			axs.set_ylim( min(eRT_mfs[ridx])-0.03, 1.01)
		else:
			axs.set_ylim( 0.75*min(eRT_mfs[ridx]), 1.01)
		
		fig.colorbar(line)

		plt.show()
		fig.savefig("fig_tlcf1_lst2_cg_simul_set4_eTF_eRT_Nx" + str(params['Nx']) + "_rhoA" + str(rhoAs[ridx]) + "_rhoB" + str(rhoBs[ridx]) + "_ikm" + str(params['ikmax']) + ".pdf")
		


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
	'ikmax': 100, #simulation id
	}

	#plot_simul_set1(params)
	#plot_simul_set2(params)
	plot_simul_set3(params)
	#plot_simul_set4(params)
		
