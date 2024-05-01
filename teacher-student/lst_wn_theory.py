#
# Model of tranfer learning and catastrophic forgetting
#
# Linear student-teacher model with weight regularization in Euclidean metric
#
# Plotting theoretical results

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


def calc_eRT(gm, rhoA, rhoB):
	eRTres = (1.0 - gm)*(1.0 - gm) - 2*gm*rhoA*( rhoB - gm*(rhoA + rhoB) + gm*gm*rhoA ) + gm*gm*rhoA*rhoA*( 1.0 - 2*gm*rhoA*rhoB + gm*gm*rhoA*rhoA )
	return 1.0 - eRTres
	

def calc_gm_opt_for_eRT(rhoA, rhoB):
	gms = np.arange(0.0, 1.0005, 0.001)
	eRTs = np.zeros((len(gms)))
	for gidx in range(len(gms)):
		eRTs[gidx] = calc_eRT(gms[gidx], rhoA, rhoB)
	
	return gms[ np.argmax(eRTs) ]


def diagram1():
	drhoA = 0.02; 
	drhoB = 0.02

	rhoAs = np.arange(0.5*drhoA, 1.00, drhoA)
	rhoBs = np.arange(0.5*drhoB, 1.00, drhoB)
	Alen = len(rhoAs)
	Blen = len(rhoBs)

	Ns = 30
	Nx = 3000#10000
	ex_ratio = Ns/Nx

	gm_opt_eTFs = np.zeros(( Alen, Blen )) 
	eTF_opts = np.zeros(( Alen, Blen ))
	gm_opt_eRTs = np.zeros(( Alen, Blen ))
	eRT_opts = np.zeros(( Alen, Blen ))
	for aidx in range(Alen):
		rhoA = rhoAs[aidx]
		for bidx in range(Blen):
			rhoB = rhoBs[bidx]
			
			gmTF = min(1, rhoB/rhoA)
			gm_opt_eTFs[aidx, bidx] = gmTF
			eTF_opts[aidx, bidx] = gmTF*rhoA*(2*rhoB - gmTF*rhoA)
			
			gmRT = calc_gm_opt_for_eRT(rhoA, rhoB)
			gm_opt_eRTs[aidx, bidx] = gmRT
			eRT_opts[aidx, bidx] = calc_eRT(gmRT, rhoA, rhoB)


	#plt.style.use("ggplot")
	plt.rcParams.update({'font.size':16})
	
	svfg1 = plt.figure()
	X, Y = np.meshgrid(np.arange(0.0, 1.01, drhoA), np.arange(0.0, 1.01, drhoB))
	plt.pcolor(X, Y, gm_opt_eTFs.T, cmap='viridis', vmax = 1.0, vmin = 0.0)

	plt.colorbar()
	plt.show()
	svfg1.savefig("fig_tlcf1_lst2_wn_theory_gm_opt_eTFs" + ".pdf")
	
	svfg2 = plt.figure()
	X, Y = np.meshgrid(np.arange(0.0, 1.01, drhoA), np.arange(0.0, 1.01, drhoB))
	plt.pcolor(X, Y, eTF_opts.T, cmap='seismic', vmax = 1.0, vmin = -1.0)
	
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.colorbar()
	plt.show()
	svfg2.savefig("fig_tlcf1_lst2_wn_theory_opt_eTFs" + ".pdf")
	
	svfg3 = plt.figure()
	X, Y = np.meshgrid(np.arange(0.0, 1.01, drhoA), np.arange(0.0, 1.01, drhoB))
	plt.pcolor(X, Y, gm_opt_eRTs.T, cmap='viridis', vmax = 1.0, vmin = 0.0)
	
	plt.xlim(0,1)
	plt.ylim(0,1)
	
	plt.colorbar()
	plt.show()
	svfg3.savefig("fig_tlcf1_lst2_wn_theory_gm_opt_eRTs" + ".pdf")
	
	svfg4 = plt.figure()
	X, Y = np.meshgrid(np.arange(0.0, 1.01, drhoA), np.arange(0.0, 1.01, drhoB))
	plt.pcolor(X, Y, eRT_opts.T, cmap='seismic', vmax = 1.0, vmin = -1.0)
	
	plt.colorbar()
	plt.show()
	svfg4.savefig("fig_tlcf1_lst2_wn_theory_opt_eRTs" + ".pdf")

	

if __name__ == "__main__":
	#stdins = sys.argv # standard inputs
	diagram1()
	#diagram2()
	
	
