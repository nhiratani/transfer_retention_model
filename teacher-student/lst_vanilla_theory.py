#
# Model of tranfer learning and catastrophic forgetting
#
# Linear student-teacher model
#
# Illustrating the analytical results on the vanilla model
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
cnum = 6
for cidx in range(cnum):
	clrs.append( cm.rainbow( (0.5+cidx)/cnum ) )


def diagram1():
	drhoA = 0.02; 
	drhoB = 0.02

	rhoAs = np.arange(0.5*drhoA, 1.00, drhoA)
	rhoBs = np.arange(0.5*drhoB, 1.00, drhoB)
	Alen = len(rhoAs)
	Blen = len(rhoBs)

	Ns = 30
	Nx = 10000
	ex_ratio = Ns/Nx

	eTFs = np.zeros(( Alen, Blen ))
	eRTs = np.zeros(( Alen, Blen ))
	for aidx in range(Alen):
		rhoA = rhoAs[aidx]
		for bidx in range(Blen):
			rhoB = rhoBs[bidx]
			eTFs[aidx, bidx] = rhoA*(2*rhoB - rhoA)
			eRTs[aidx, bidx] = 1.0 - rhoA*rhoA*(rhoA*rhoA - 2*rhoA*rhoB + 1)


	#plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg1 = plt.figure()
	X, Y = np.meshgrid(np.arange(0.0, 1.01, drhoA), np.arange(0.0, 1.01, drhoB))
	plt.pcolor(X, Y, eTFs.T, cmap='seismic', vmax = 1.0, vmin = -1.0)

	xs = np.arange(0.0, 1.01, 0.02)
	#plt.plot(xs, xs, color='gray', ls='-', lw=1.0)
	#plt.plot(xs, 0.5*xs, color='gray', ls='-', lw=1.0)
	
	plt.colorbar()
	plt.show()
	svfg1.savefig("fig_tlcf1_lst2_vanilla_theory_eTF_task_similarity_nl" + ".pdf")
	
	svfg2 = plt.figure()
	X, Y = np.meshgrid(np.arange(0.0, 1.01, drhoA), np.arange(0.0, 1.01, drhoB))
	plt.pcolor(X, Y, eRTs.T, cmap='seismic', vmax = 1.0, vmin = -1.0)
	
	xs = np.arange(0.01, 1.03, 0.02)
	xones = np.ones(( len(xs) ))
	y1s = np.divide( np.multiply(xs, xs) + xones, 2*xs ) - 0.5*np.divide( xones, np.multiply(np.multiply(xs, xs), xs) )
	y2s = (2/3)*xs + (1/3)*np.divide(xones, xs)
	plt.plot(xs, y2s, color='white', ls='--', lw=2.0)
	#plt.plot(xs, 0.5*xs, color='k', ls='-', lw=1.0)
	plt.xlim(0,1)
	plt.ylim(0.9,1)
	plt.colorbar()
	plt.show()
	svfg2.savefig("fig_tlcf1_lst2_vanilla_theory_eRT_task_similarity" + ".pdf")


def diagram2():
	drhoA = 0.02; 
	drhoB = 0.002

	rhoAs = np.arange(0.5*drhoA, 1.00, drhoA)
	rhoBs = np.arange(0.9 + 0.5*drhoB, 1.00, drhoB)
	Alen = len(rhoAs)
	Blen = len(rhoBs)

	Ns = 30
	Nx = 10000
	ex_ratio = Ns/Nx

	eTFs = np.zeros(( Alen, Blen ))
	eRTs = np.zeros(( Alen, Blen ))
	for aidx in range(Alen):
		rhoA = rhoAs[aidx]
		for bidx in range(Blen):
			rhoB = rhoBs[bidx]
			eTFs[aidx, bidx] = rhoA*(2*rhoB - rhoA)
			eRTs[aidx, bidx] = 1.0 - rhoA*rhoA*(rhoA*rhoA - 2*rhoA*rhoB + 1)

	print( np.min(eRTs) )
	#plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg2 = plt.figure()
	X, Y = np.meshgrid(np.arange(0.0, 1.01, drhoA), np.arange(0.9, 1.001, drhoB))
	plt.pcolor(X, Y, eRTs.T, cmap='Reds', vmax = 1.0, vmin = 0.8)
	
	xs = np.arange(0.01, 1.03, 0.02)
	xones = np.ones(( len(xs) ))
	y1s = np.divide( np.multiply(xs, xs) + xones, 2*xs ) - 0.5*np.divide( xones, np.multiply(np.multiply(xs, xs), xs) )
	y2s = (2/3)*xs + (1/3)*np.divide(xones, xs)
	plt.plot(xs, y2s, color='white', ls='--', lw=2.0)
	#plt.plot(xs, 0.5*xs, color='k', ls='-', lw=1.0)
	plt.xlim(0,1)
	plt.ylim(0.9,1)
	plt.colorbar()
	plt.show()
	svfg2.savefig("fig_tlcf1_lst2_vanilla_theory_eRT_task_similarity" + ".pdf")


def diagram3():
	drhoA = 0.02; 
	drhoB = 0.02

	rhoAs = np.arange(-1.00+0.5*drhoA, 1.00, drhoA)
	rhoBs = np.arange(-1.00+0.5*drhoB, 1.00, drhoB)
	Alen = len(rhoAs)
	Blen = len(rhoBs)

	Ns = 30
	Nx = 10000
	ex_ratio = Ns/Nx

	eTFs = np.zeros(( Alen, Blen ))
	eRTs = np.zeros(( Alen, Blen ))
	for aidx in range(Alen):
		rhoA = rhoAs[aidx]
		for bidx in range(Blen):
			rhoB = rhoBs[bidx]
			eTFs[aidx, bidx] = rhoA*(2*rhoB - rhoA)
			eRTs[aidx, bidx] = 1.0 - rhoA*rhoA*(rhoA*rhoA - 2*rhoA*rhoB + 1)

	print( np.min(eTFs), np.min(eRTs) )
	#plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg1 = plt.figure()
	X, Y = np.meshgrid(np.arange(-1.0, 1.01, drhoA), np.arange(-1.0, 1.01, drhoB))
	plt.pcolor(X, Y, eTFs.T, cmap='seismic', vmax = 3.0, vmin = -3.0)

	#xs = np.arange(0.0, 1.01, 0.02)
	#plt.plot(xs, xs, color='gray', ls='-', lw=1.0)
	#plt.plot(xs, 0.5*xs, color='gray', ls='-', lw=1.0)
	
	plt.colorbar()
	plt.show()
	svfg1.savefig("fig_tlcf1_lst2_vanilla_theory_eTF_task_similarity_full" + ".pdf")
	
	svfg2 = plt.figure()
	X, Y = np.meshgrid(np.arange(-1, 1.01, drhoA), np.arange(-1, 1.01, drhoB))
	plt.pcolor(X, Y, eRTs.T, cmap='seismic', vmax = 3.0, vmin = -3.0)
	
	#xs = np.arange(0.01, 1.03, 0.02)
	#xones = np.ones(( len(xs) ))
	#y1s = np.divide( np.multiply(xs, xs) + xones, 2*xs ) - 0.5*np.divide( xones, np.multiply(np.multiply(xs, xs), xs) )
	#y2s = (2/3)*xs + (1/3)*np.divide(xones, xs)
	#plt.plot(xs, y2s, color='white', ls='--', lw=2.0)
	#plt.plot(xs, 0.5*xs, color='k', ls='-', lw=1.0)
	plt.xlim(-1,1)
	plt.ylim(-1,1)
	plt.colorbar()
	plt.show()
	svfg2.savefig("fig_tlcf1_lst2_vanilla_theory_eRT_task_similarity_full" + ".pdf")



if __name__ == "__main__":
	#stdins = sys.argv # standard inputs
	#diagram1()
	#diagram2()
	diagram3()
	
	
