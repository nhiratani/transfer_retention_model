#
# Model of tranfer learning and catastrophic forgetting
#
# Linear student-teacher model
# 
# Data generation and weight update functions
#

import sys
from math import *
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from scipy import special as scisp

# Geneate feature weight A and readout weight B
def generate_AB(key, params):
	Ns = params['Ns']; Nx = params['Nx']; Ny = params['Ny'] 
	akey, bkey = random.split(key)
	A = (1.0/sqrt(Ns))*random.normal(akey, (Nx, Ns))
	B = (1.0/sqrt(Ns))*random.normal(bkey, (Ny, Ns))
	return A, B


def generate_AB_mix(key, Ao, Bo, params):
	rhoA = params['rhoA']; rhoB = params['rhoB']
	Atmp, Btmp = generate_AB(key, params)
	return rhoA*Ao + jnp.sqrt(1 - rhoA*rhoA)*Atmp, rhoB*Bo + jnp.sqrt(1 - rhoB*rhoB)*Btmp


def generate_g(key, params):
	Nx = params['Nx']; alpha = params['alpha']
	g = random.bernoulli(key, alpha, (Nx,))
	return g


def generate_tasks(key, params):
	Aseq = []; Bseq = []
	for sidx in range( params['Nsess'] ):
		key, abkey, gkey = random.split(key, num=3)
		if sidx == 0:
			A, B = generate_AB(abkey, params)
		else:
			A, B = generate_AB_mix(key, A, B, params)
		Aseq.append(A); Bseq.append(B)
		
	return Aseq, Bseq


# Calculate squared frobenius norm
def fnorm2(Mtmp):
	fnorm = jnp.linalg.norm( Mtmp, ord='fro' )
	return fnorm*fnorm


@jit
def calc_dW(W, A, B): #vanilla weight update (with GD)
	return -jnp.dot(B - jnp.dot(W, A), A.T) 


@jit
def calc_dW_cg(W, DA, B): #weight update with input gating
	return -jnp.dot(B - jnp.dot(W, DA), DA.T)

@jit
def calc_dW_cpg(W, A, B, D): #weight update with input gating
	DA = jnp.dot(D, A)
	return -jnp.dot(B - jnp.dot(W, A), DA.T)


@jit
def calc_dW_wn(W, A, B, Wo, lmbd): #weight update with weight regularization
	return -jnp.dot(B - jnp.dot(W, A), A.T) + lmbd*(W - Wo)


@jit
def calc_dW_cwn(W, A, B, Wo, Ao, lmbd_c): #weight update with weight regularization in the covariance space
	return -jnp.dot(B - jnp.dot(W, A), A.T) + lmbd_c*jnp.dot( jnp.dot(W - Wo, Ao), Ao.T )
	

@jit
def calc_dW_cwn_diag(W, A, B, Wo, Dm, lmbd): #weight update with weight regularization in the covariance space
	return -jnp.dot(B - jnp.dot(W, A), A.T) + lmbd*jnp.dot(W - Wo, Dm)


@jit
def calc_dW_ist(W, A, B, S, h, batch_size): #weight update with weight regularization in the covariance space
	X = jnp.dot(A, S)
	phiX = jnp.multiply( jnp.sign(X), jnp.clip( jnp.abs(X) - h*jnp.ones( jnp.shape(X) ), 0.0, None )  )
	Y = jnp.dot(B, S)
	return -(1.0/batch_size)*jnp.dot( Y - jnp.dot(W, phiX), phiX.T )

@jit
def calc_error_ist(W, A, B, S, h, batch_size): #weight update with weight regularization in the covariance space
	X = jnp.dot(A, S)
	phiX = jnp.multiply( jnp.sign(X), jnp.clip( jnp.abs(X) - h*jnp.ones( jnp.shape(X) ), 0.0, None )  )
	Y = jnp.dot(B, S)
	return (1.0/batch_size)*fnorm2( Y - jnp.dot(W, phiX) )



