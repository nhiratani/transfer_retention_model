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


def calc_eTF_eRT_cwn_mf(params):
	rhoA = params['rhoA']; rhoB = params['rhoB']; gm = params['gm']
	eTF = rhoA*(2*rhoB - rhoA)
	if rhoA == 1.0 or gm == 1.0:
		eRTres = 2*gm*gm*(1 - rhoB)
	else:
		eRTres = 0.0
	eRT = 1.0 - eRTres
	return eTF, eRT


def calc_mean_eTF_eRT_cwn_mf(params):
	gm = params['gm']
	mean_eTF = 1.0/2.0 - 1.0/3.0
	mean_eRT = 1.0
	return mean_eTF, mean_eRT


def plot_simul_set1(params):
	params['rhoA'] = 0.9#1.0
	
	rhoBs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	gms = [0.2, 0.4, 0.6, 0.8, 1.0]
	gmlen = len(gms)
	rBlen = len(rhoBs)
	
	num_epochs = params['num_epochs']
	ikmax = params['ikmax'] 
	
	eTFs = np.zeros((gmlen, rBlen, ikmax))
	eRTs = np.zeros((gmlen, rBlen, ikmax))
	perfs = np.zeros((2, gmlen, rBlen, ikmax))
	
	eTF_means = np.zeros((gmlen, rBlen)); eTF_stds = np.zeros((gmlen, rBlen))
	eRT_means = np.zeros((gmlen, rBlen)); eRT_stds = np.zeros((gmlen, rBlen))
	
	perf_means = np.zeros((2, gmlen, rBlen)); perf_stds = np.zeros((2, gmlen, rBlen))
	
	for gmidx in range(gmlen):
		params['gm'] = gms[gmidx]
		for rbidx in range(rBlen):
			params['rhoB'] = rhoBs[rbidx]
			
			fstr = 'data/tlcf1_lst2_cwn_errors_Nx' + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
			+ '_nep' + str(params['num_epochs']) + '_rhoA' + str(params['rhoA']) + '_rhoB' + str(params['rhoB']) + '_gm' + str(params['gm'])+ '_ikm' + str(params['ikmax']) + ".txt"

			lidx = 0
			for line in open(fstr, 'r'):
				ltmps = line[:-1].split(" ")
				eTFs[gmidx, rbidx, lidx] = 1.0 - float( ltmps[2] )
				eRTs[gmidx, rbidx, lidx] = 1.0 - float( ltmps[3] )
				perfs[0, gmidx, rbidx, lidx] = 1.0 - float( ltmps[1] )
				perfs[1, gmidx, rbidx, lidx] = 1.0 - float( ltmps[4] )
				lidx += 1
			
			eTF_means[gmidx, rbidx] = np.mean(eTFs[gmidx, rbidx, :])
			eTF_stds[gmidx, rbidx] = np.std(eTFs[gmidx, rbidx, :])
			
			eRT_means[gmidx, rbidx] = np.mean(eRTs[gmidx, rbidx, :])
			eRT_stds[gmidx, rbidx] = np.std(eRTs[gmidx, rbidx, :])
			
			for q in range(2):
				perf_means[q, gmidx, rbidx] = np.mean(perfs[q, gmidx, rbidx, :])
				perf_stds[q, gmidx, rbidx] = np.std(perfs[q, gmidx, rbidx, :])
			
	
	print(perf_means)
	
	gm_mfs = np.arange(0.01, 1.005, 0.01)
	gmflen = len(gm_mfs)
	eTF_mfs = np.zeros((rBlen, gmflen))
	eRT_mfs = np.zeros((rBlen, gmflen))
	
	for rbidx in range(rBlen):
		params['rhoB'] = rhoBs[rbidx]
		for gmfidx in range(gmflen):
			params['gm'] = gm_mfs[gmfidx]
			eTFtmp, eRTtmp = calc_eTF_eRT_cwn_mf(params)
			eTF_mfs[rbidx, gmfidx] = eTFtmp
			eRT_mfs[rbidx, gmfidx] = eRTtmp
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg1 = plt.figure()
	for rbidx in range(rBlen):
		plt.errorbar(gms, eTF_means[:, rbidx], eTF_stds[:, rbidx], color=clrs[rbidx], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
		plt.plot(gm_mfs, eTF_mfs[rbidx], '-', color=clrs[rbidx])
	plt.show()
	svfg1.savefig("fig_tlcf1_lst2_cwn_simul_set1_eTF_Nx" + str(params['Nx']) + '_rhoA' + str(params['rhoA']) + '_lr' + str(params['learning_rate'])\
					+ '_nep' + str(params['num_epochs']) + "_ikm" + str(params['ikmax']) + ".pdf")
		
	svfg2 = plt.figure()
	for rbidx in range(rBlen):
		plt.errorbar(gms, eRT_means[:, rbidx], eRT_stds[:, rbidx], color=clrs[rbidx], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
		plt.plot(gm_mfs, eRT_mfs[rbidx], '-', color=clrs[rbidx])
	plt.show()
	svfg2.savefig("fig_tlcf1_lst2_cwn_simul_set1_eRT_Nx" + str(params['Nx']) + '_rhoA' + str(params['rhoA']) + '_lr' + str(params['learning_rate'])\
					+ '_nep' + str(params['num_epochs']) + "_ikm" + str(params['ikmax']) + ".pdf")
					
	svfg3 = plt.figure()
	for q in range(2):
		plt.subplot(2,1,1+q)
		for rbidx in range(rBlen):
			plt.errorbar(gms, perf_means[q, :, rbidx], perf_stds[q, :, rbidx], color=clrs[rbidx], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
			#plt.plot(gm_mfs, eRT_mfs[rbidx], '-', color=clrs[rbidx])
	plt.show()
	svfg3.savefig("fig_tlcf1_lst2_cwn_simul_set1_perfs_Nx" + str(params['Nx']) + '_rhoA' + str(params['rhoA']) + '_lr' + str(params['learning_rate'])\
					+ '_nep' + str(params['num_epochs']) + "_ikm" + str(params['ikmax']) + ".pdf")


def plot_simul_set2(params):
	params['rhoB'] = 1.0
	
	rhoAs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	gms = [0.2, 0.4, 0.6, 0.8, 1.0]
	gmlen = len(gms)
	rAlen = len(rhoAs)
	
	num_epochs = params['num_epochs']
	ikmax = params['ikmax'] 
	
	eTFs = np.zeros((gmlen, rAlen, ikmax))
	eRTs = np.zeros((gmlen, rAlen, ikmax))
	perfs = np.zeros((2, gmlen, rAlen, ikmax))
	
	eTF_means = np.zeros((gmlen, rAlen)); eTF_stds = np.zeros((gmlen, rAlen))
	eRT_means = np.zeros((gmlen, rAlen)); eRT_stds = np.zeros((gmlen, rAlen))
	perf_means = np.zeros((2, gmlen, rAlen)); perf_stds = np.zeros((2, gmlen, rAlen))

	for gmidx in range(gmlen):
		params['gm'] = gms[gmidx]
		for raidx in range(rAlen):
			params['rhoA'] = rhoAs[raidx]
			
			fstr = 'data/tlcf1_lst2_cwn_errors_Nx' + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
			+ '_nep' + str(params['num_epochs']) + '_rhoA' + str(params['rhoA']) + '_rhoB' + str(params['rhoB']) + '_gm' + str(params['gm'])+ '_ikm' + str(params['ikmax']) + ".txt"

			lidx = 0
			for line in open(fstr, 'r'):
				ltmps = line[:-1].split(" ")
				eTFs[gmidx, raidx, lidx] = 1.0 - float( ltmps[2] )
				eRTs[gmidx, raidx, lidx] = 1.0 - float( ltmps[3] )
				perfs[0, gmidx, raidx, lidx] = 1.0 - float( ltmps[1] )
				perfs[1, gmidx, raidx, lidx] = 1.0 - float( ltmps[4] )
				lidx += 1
			
			eTF_means[gmidx, raidx] = np.mean(eTFs[gmidx, raidx, :])
			eTF_stds[gmidx, raidx] = np.std(eTFs[gmidx, raidx, :])
			
			eRT_means[gmidx, raidx] = np.mean(eRTs[gmidx, raidx, :])
			eRT_stds[gmidx, raidx] = np.std(eRTs[gmidx, raidx, :])
			
			for q in range(2):
				perf_means[q, gmidx, raidx] = np.mean(perfs[q, gmidx, raidx, :])
				perf_stds[q, gmidx, raidx] = np.std(perfs[q, gmidx, raidx, :])
				
	print( perf_means )
			
	
	gm_mfs = np.arange(0.01, 1.005, 0.01)
	gmflen = len(gm_mfs)
	eTF_mfs = np.zeros((rAlen, gmflen))
	eRT_mfs = np.zeros((rAlen, gmflen))
	
	for raidx in range(rAlen):
		params['rhoA'] = rhoAs[raidx]
		for gmfidx in range(gmflen):
			params['gm'] = gm_mfs[gmfidx]
			eTFtmp, eRTtmp = calc_eTF_eRT_cwn_mf(params)
			eTF_mfs[raidx, gmfidx] = eTFtmp
			eRT_mfs[raidx, gmfidx] = eRTtmp
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg1 = plt.figure()
	for raidx in range(rAlen):
		plt.errorbar(gms, eTF_means[:, raidx], eTF_stds[:, raidx], color=clrs[raidx], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
		plt.plot(gm_mfs, eTF_mfs[raidx], '-', color=clrs[raidx])
	plt.show()
	svfg1.savefig("fig_tlcf1_lst2_cwn_simul_set2_eTF_Nx" + str(params['Nx']) + '_rhoB' + str(params['rhoB']) + '_lr' + str(params['learning_rate'])\
					+ '_nep' + str(params['num_epochs']) + "_ikm" + str(params['ikmax']) + ".pdf")
		
	svfg2 = plt.figure()
	for raidx in range(rAlen):
		plt.errorbar(gms, eRT_means[:, raidx], eRT_stds[:, raidx], color=clrs[raidx], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
		plt.plot(gm_mfs, eRT_mfs[raidx], '-', color=clrs[raidx])
	print(eRT_means)
	plt.ylim(0.9, 1.005)
	plt.show()
	svfg2.savefig("fig_tlcf1_lst2_cwn_simul_set2_eRT_Nx" + str(params['Nx']) + '_rhoB' + str(params['rhoB']) + '_lr' + str(params['learning_rate'])\
					+ '_nep' + str(params['num_epochs']) + "_ikm" + str(params['ikmax']) + ".pdf")

	svfg3 = plt.figure()
	for q in range(2):
		plt.subplot(2,1,1+q)
		for rbidx in range(rAlen):
			plt.errorbar(gms, perf_means[q, :, raidx], perf_stds[q, :, raidx], color=clrs[raidx], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
			#plt.plot(gm_mfs, eRT_mfs[rbidx], '-', color=clrs[rbidx])
	plt.show()
	svfg3.savefig("fig_tlcf1_lst2_cwn_simul_set2_perfs_Nx" + str(params['Nx']) + '_rhoA' + str(params['rhoA']) + '_lr' + str(params['learning_rate'])\
					+ '_nep' + str(params['num_epochs']) + "_ikm" + str(params['ikmax']) + ".pdf")
					

def plot_simul_set3(params):
	gms = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]#[0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
	gmlen = len(gms)
	
	num_epochs = params['num_epochs']
	ikmax = params['ikmax'] 
	
	eTFs = np.zeros((gmlen, ikmax))
	eRTs = np.zeros((gmlen, ikmax))
	perfs = np.zeros((2, gmlen, ikmax))
	
	eTF_means = np.zeros((gmlen)); eTF_stds = np.zeros((gmlen))
	eRT_means = np.zeros((gmlen)); eRT_stds = np.zeros((gmlen))
	perf_means = np.zeros((2, gmlen)); perf_stds = np.zeros((2, gmlen))

	for gmidx in range(gmlen):
		params['gm'] = gms[gmidx]
		fstr = 'data/tlcf1_lst2_cwn_mean_errors_Nx' + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
		+ '_nep' + str(params['num_epochs']) + '_gm' + str(params['gm'])+ '_ikm' + str(params['ikmax']) + ".txt"

		lidx = 0
		for line in open(fstr, 'r'):
			ltmps = line[:-1].split(" ")
			eTFs[gmidx, lidx] = 1.0 - float( ltmps[2] )
			eRTs[gmidx, lidx] = 1.0 - float( ltmps[3] )
			perfs[0, gmidx, lidx] = 1.0 - float(ltmps[1])
			perfs[1, gmidx, lidx] = 1.0 - float(ltmps[4])
			lidx += 1
			
		eTF_means[gmidx] = np.mean(eTFs[gmidx, :])
		eTF_stds[gmidx] = np.std(eTFs[gmidx, :])
			
		eRT_means[gmidx] = np.mean(eRTs[gmidx, :])
		eRT_stds[gmidx] = np.std(eRTs[gmidx, :])
		
		for q in range(2):
			perf_means[q, gmidx] = np.mean(perfs[q, gmidx, :])
			perf_stds[q, gmidx] = np.std(perfs[q, gmidx, :])
			
	
	gm_mfs = np.arange(0.0, 1.005, 0.01)
	gmflen = len(gm_mfs)
	eTF_mfs = np.zeros((gmflen))
	eRT_mfs = np.zeros((gmflen))
	
	for gmfidx in range(gmflen):
		params['gm'] = gm_mfs[gmfidx]
		eTFtmp, eRTtmp = calc_mean_eTF_eRT_cwn_mf(params)
		eTF_mfs[gmfidx] = eTFtmp
		eRT_mfs[gmfidx] = eRTtmp
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg1 = plt.figure()
	#plt.errorbar(alphas, eTF_means, eTF_stds[:, raidx], color=clrs[raidx], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
	plt.axhline( 1/6.0, color='C0', ls='--' )
	plt.plot(gms, eTF_means, 'o', color='C0')
	plt.plot(gm_mfs, eTF_mfs, '-', color='C0')
	
	plt.axhline( 1.0 - (1/5.0 - 1/4.0 + 1/3.0), color='C1', ls='--' )
	plt.plot(gms, eRT_means, 'o', color='C1')
	plt.plot(gm_mfs, eRT_mfs, '-', color='C1')
	
	plt.ylim(0.0, 1.025)
	
	plt.show()
	svfg1.savefig("fig_tlcf1_lst2_cwn_simul_set3_mean_eTF_eTR_Nx" + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
					+ '_nep' + str(params['num_epochs']) + "_ikm" + str(params['ikmax']) + ".pdf")
	
	#print(perf_means)
	svfg2 = plt.figure()
	#plt.errorbar(alphas, eTF_means, eTF_stds[:, raidx], color=clrs[raidx], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
	#plt.axhline( 1/6.0, color='C0', ls='--' )
	plt.plot(gms, perf_means[0], 'o', color='C2')
	#plt.plot(gm_mfs, eTF_mfs, '-', color='C0')
	
	#plt.axhline( 1.0 - (1/5.0 - 1/4.0 + 1/3.0), color='C1', ls='--' )
	plt.plot(gms, perf_means[1], 'o', color='C3')
	#plt.plot(gm_mfs, eRT_mfs, '-', color='C1')
	
	plt.ylim(0.0, 1.025)
	
	plt.show()
	svfg2.savefig("fig_tlcf1_lst2_cwn_simul_set3_mean_perfs_Nx" + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
					+ '_nep' + str(params['num_epochs']) + "_ikm" + str(params['ikmax']) + ".pdf")
		

def plot_simul_set4(params):
	rhoAs = [0.3, 0.6, 0.9]
	rhoBs = [0.3, 0.6, 0.9]
	gms = [0.2, 0.4, 0.6, 0.8, 1.0]
	
	gmlen = len(gms)
	ralen = len(rhoAs)
	rblen = len(rhoBs)
	
	num_epochs = params['num_epochs']
	ikmax = params['ikmax'] 
	
	eTFs = np.zeros((gmlen, ralen, rblen, ikmax))
	eRTs = np.zeros((gmlen, ralen, rblen, ikmax))
	
	eTF_means = np.zeros((gmlen, ralen, rblen)); eTF_stds = np.zeros((gmlen, ralen, rblen))
	eRT_means = np.zeros((gmlen, ralen, rblen)); eRT_stds = np.zeros((gmlen, ralen, rblen))

	for raidx in range(ralen):
		params['rhoA'] = rhoAs[raidx]
		for rbidx in range(rblen):
			params['rhoB'] = rhoBs[rbidx]
		
			for gmidx in range(gmlen):
				params['gm'] = gms[gmidx]
				
				fstr = 'data/tlcf1_lst2_wn_errors_Nx' + str(params['Nx']) + '_lr' + str(params['learning_rate'])\
				+ '_nep' + str(params['num_epochs']) + '_rhoA' + str(params['rhoA']) + '_rhoB' + str(params['rhoB']) + '_gm' + str(params['gm'])+ '_ikm' + str(params['ikmax']) + ".txt"

				lidx = 0
				for line in open(fstr, 'r'):
					ltmps = line[:-1].split(" ")
					eTFs[gmidx, raidx, rbidx, lidx] = 1.0 - float( ltmps[2] )
					eRTs[gmidx, raidx, rbidx, lidx] = 1.0 - float( ltmps[3] )
					lidx += 1
				
				eTF_means[gmidx, raidx, rbidx] = np.mean(eTFs[gmidx, raidx, rbidx, :])
				eTF_stds[gmidx, raidx, rbidx] = np.std(eTFs[gmidx, raidx, rbidx, :])
				
				eRT_means[gmidx, raidx, rbidx] = np.mean(eRTs[gmidx, raidx, rbidx, :])
				eRT_stds[gmidx, raidx, rbidx] = np.std(eRTs[gmidx, raidx, rbidx, :])
			
	
	gm_mfs = np.arange(0.01, 1.005, 0.01)
	gmflen = len(gm_mfs)
	eTF_mfs = np.zeros((gmflen, ralen, rblen))
	eRT_mfs = np.zeros((gmflen, ralen, rblen))
	
	for raidx in range(ralen):
		params['rhoA'] = rhoAs[raidx]
		for rbidx in range(rblen):
			params['rhoB'] = rhoBs[rbidx]
			for gmfidx in range(gmflen):
				params['gm'] = gm_mfs[gmfidx]
				eTFtmp, eRTtmp = calc_eTF_eRT_wn_mf(params)
				eTF_mfs[gmfidx, raidx, rbidx] = eTFtmp
				eRT_mfs[gmfidx, raidx, rbidx] = eRTtmp
	
	plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	#svfg1 = plt.figure()
	
	for raidx in range(ralen):
		for rbidx in range(rblen):
			fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
			
			points = np.array([eTF_mfs[:, raidx, rbidx], eRT_mfs[:, raidx, rbidx]]).T.reshape(-1, 1, 2)
			segments = np.concatenate([points[:-1], points[1:]], axis=1)
			norm = plt.Normalize(gm_mfs.min(), gm_mfs.max())

			lc = LineCollection(segments, cmap='rainbow', norm=norm)
			lc.set_array(gm_mfs)
			lc.set_linewidth(2)
			line = axs.add_collection(lc)
			
			climit = 5
			clrs = []
			for q in range(climit):
				clrs.append( cm.rainbow( gms[q] ) )
			for gmidx in range(gmlen):
				axs.errorbar(eTF_means[gmidx, raidx, rbidx], eRT_means[gmidx, raidx, rbidx], xerr=eTF_stds[gmidx, raidx, rbidx], yerr=eRT_stds[gmidx, raidx, rbidx],\
				color=clrs[gmidx], lw=0.0, elinewidth=2.0, capthick=2.0, capsize=4.0, marker='o', markersize=7.5)
			
			axs.set_xlim(0.0, min(1.25*max(eTF_mfs[:,raidx,rbidx]), 1.0))
			axs.set_ylim(0.0, min(1.25*max(eRT_mfs[:,raidx,rbidx]), 1.0))
			"""
			if min(eTF_mfs[ridx]) >= 0:
				axs.set_xlim(0, 1.25*max(eTF_mfs[ridx]))
			else:
				axs.set_xlim(2.0*min(eTF_mfs[ridx]), 2.0*max(eTF_mfs[ridx]))
			if min(eRT_mfs[ridx]) > 0.9:
				axs.set_ylim( min(eRT_mfs[ridx])-0.03, 1.01)
			else:
				axs.set_ylim( 0.75*min(eRT_mfs[ridx]), 1.01)
			"""
			fig.colorbar(line)

			plt.show()
			fig.savefig("fig_tlcf1_lst2_wn_simul_set4_eTF_eRT_Nx" + str(params['Nx']) + "_rhoA" + str(rhoAs[raidx]) + "_rhoB" + str(rhoBs[rbidx]) + "_ikm" + str(params['ikmax']) + ".pdf")
			


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
	'ikmax': 500, #simulation id
	}

	#plot_simul_set1(params)
	#plot_simul_set2(params)
	plot_simul_set3(params)
	#plot_simul_set4(params)
		
