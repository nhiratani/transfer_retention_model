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

	alpha_opt_eTFs = np.zeros(( Alen, Blen )) 
	eTF_opts = np.zeros(( Alen, Blen ))
	alpha_opt_eRTs = np.zeros(( Alen, Blen ))
	eRT_opts = np.zeros(( Alen, Blen ))
	for aidx in range(Alen):
		rhoA = rhoAs[aidx]
		for bidx in range(Blen):
			rhoB = rhoBs[bidx]
			
			alpha = min(1, rhoB/rhoA)
			alpha_opt_eTFs[aidx, bidx] = alpha
			eTF_opts[aidx, bidx] = alpha*rhoA*(2*rhoB - alpha*rhoA)
			
			#if rhoB > 2*sqrt(2)/3:
			#	alpha = (3*rhoB - sqrt(9*rhoB*rhoB - 8))/(4*rhoA)
			#else:
			alpha = 0.0
			alpha_opt_eRTs[aidx, bidx] = alpha
			eRT_opts[aidx, bidx] = 1.0 - alpha*rhoA*alpha*rhoA*(alpha*rhoA*alpha*rhoA - 2*alpha*rhoA*rhoB + 1)


	#plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg1 = plt.figure()
	X, Y = np.meshgrid(np.arange(0.0, 1.01, drhoA), np.arange(0.0, 1.01, drhoB))
	plt.pcolor(X, Y, alpha_opt_eTFs.T, cmap='viridis', vmax = 1.0, vmin = 0.0)

	plt.colorbar()
	plt.show()
	svfg1.savefig("fig_tlcf1_lst2_cg_theory_alpha_opt_eTFs" + ".pdf")
	
	svfg2 = plt.figure()
	X, Y = np.meshgrid(np.arange(0.0, 1.01, drhoA), np.arange(0.0, 1.01, drhoB))
	plt.pcolor(X, Y, eTF_opts.T, cmap='seismic', vmax = 1.0, vmin = -1.0)
	
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.colorbar()
	plt.show()
	svfg2.savefig("fig_tlcf1_lst2_cg_theory_opt_eTFs" + ".pdf")
	
	svfg3 = plt.figure()
	X, Y = np.meshgrid(np.arange(0.0, 1.01, drhoA), np.arange(0.0, 1.01, drhoB))
	plt.pcolor(X, Y, alpha_opt_eRTs.T, cmap='viridis', vmax = 1.0, vmin = 0.0)
	
	xs = np.arange(0.01, 1.03, 0.02)
	ys = np.zeros(( len(xs) ))
	for xidx in range( len(xs) ):
		x = xs[xidx]
		if x < 1.0/sqrt(2.0):
			ys[xidx] = 2*x/3 + 1/(3*x)
		else:
			ys[xidx] = 2*sqrt(2)/3
	
	plt.plot(xs, ys, color='white', ls='--', lw=2.0)
	#plt.plot(xs, 0.5*xs, color='k', ls='-', lw=1.0)
	plt.xlim(0,1)
	plt.ylim(0,1)
	
	plt.colorbar()
	plt.show()
	svfg3.savefig("fig_tlcf1_lst2_cg_theory_alpha_opt_eRTs" + ".pdf")
	
	svfg4 = plt.figure()
	X, Y = np.meshgrid(np.arange(0.0, 1.01, drhoA), np.arange(0.0, 1.01, drhoB))
	plt.pcolor(X, Y, eRT_opts.T, cmap='seismic', vmax = 1.0, vmin = -1.0)
	
	plt.colorbar()
	plt.show()
	svfg4.savefig("fig_tlcf1_lst2_cg_theory_opt_eRTs" + ".pdf")


def diagram2():
	plt.rcParams.update({'font.size':16})
	svfg1 = plt.figure()

	x1s = np.arange(0.5, 1.005, 0.01)
	y1cs = []; y1fs = []
	for x in x1s:
		y1cs.append(1.0)
		if x < 1.0/sqrt(2.0):
			y1fs.append( 2*x/3 + 1/(3*x) )
		else:
			y1fs.append( 2*sqrt(2.0)/3 )
	plt.plot(x1s, y1fs, color='k')
	plt.fill_between(x1s, y1cs, y1fs, color='r', alpha=0.25)

	x2s = np.arange(0.0, 1.005, 0.01)
	y2cs = []; y2fs = []
	for x2 in x2s:
		y2cs.append( x2 ); y2fs.append(0.0)
	plt.plot(x2s, y2cs, color='k')
	plt.fill_between(x2s, y2cs, y2fs, color='b', alpha=0.25)

	#x3s = np.arange(0.0, 1.005, 0.01)
	#y3cs = []; y3fs = []
	#for x in x3s:
	#	y3cs.append( 0.5*x ); y3fs.append(0.0)
	#plt.plot(x3s, y3cs, color='k')
	#plt.fill_between(x3s, y3cs, y3fs, color='b', alpha=0.5)

	plt.plot([0.5], [0.75], 'o', markersize=10.0, color='green')
	plt.plot([0.75], [0.5], 'o', markersize=10.0, color='blue')
	plt.plot([0.75], [0.98], 'o', markersize=10.0, color='red')
	#plt.plot(xs, 0.5*xs, color='k', ls='-', lw=1.0)
	plt.xlim(0,1.0)
	plt.ylim(0,1.0)

	plt.show()
	svfg1.savefig("fig_tlcf1_lst2_theory_diagram2_rhoArhoB" + ".pdf")
			

if __name__ == "__main__":
	#stdins = sys.argv # standard inputs
	#diagram1()
	diagram2()
	
	
