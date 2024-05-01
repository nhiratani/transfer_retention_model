#
# Model of tranfer learning and catastrophic forgetting
#
# purmuted MNIST Dataset
#

import sys
from math import *
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from jax.scipy.special import logsumexp


def random_layer_params(m, n, key):
	w_key, b_key = random.split(key)
	# small initialization
	return (1.0/m) * random.normal(w_key, (n, m)), (1.0/m) * random.normal(b_key, (n,))
	# lazy initialization
	#return (1/jnp.sqrt(m)) * random.normal(w_key, (n, m)), (1/jnp.sqrt(m)) * random.normal(b_key, (n,))
	

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
	keys = random.split(key, len(sizes))
	return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
	
	
def copy_params_value(params):
	new_params = []
	for w,b in params:
		wtmp = jnp.zeros( jnp.shape(w) )
		btmp = jnp.zeros( jnp.shape(b) )
		wtmp = wtmp.at[:,:].set(w[:,:])
		btmp = btmp.at[:].set(b[:])
		new_params.append( (wtmp, btmp) )
	return new_params


#initialization of gating units
def init_gates(hy_params, key):
	layer_sizes = hy_params['layer_sizes']
	alpha = hy_params['alpha']
	
	gs = []
	for Lsize in layer_sizes[:-1]:
		gkey, key = random.split(key, 2)
		gs.append( random.bernoulli(gkey, alpha, (Lsize,)) )

	return gs
		

#Rectified linear function
def relu(x):
	return jnp.maximum(0, x)
	

#Derivative of rectified linear function
def drelu(x):
	return jnp.heaviside(x, 0.0)


def one_hot(x, k, dtype=jnp.float32):
	"""Create a one-hot encoding of x of size k."""
	return jnp.array(x[:, None] == jnp.arange(k), dtype)
  

@jit
def predict(params, image):
	# per-example predictions
	activations = image
	for w, b in params[:-1]:
		outputs = jnp.dot(w, activations) + b
		activations = relu(outputs)
  
	final_w, final_b = params[-1]
	return jnp.dot(final_w, activations) + final_b
	#logits = jnp.dot(final_w, activations) + final_b
	#return logits - logsumexp(logits)


# Make a batched version of the `predict` function
batched_predict = vmap(predict, in_axes=(None, 0))


#def accuracy(params, images, targets):
#	target_class = jnp.argmax(targets, axis=1)
#	predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
#	return jnp.mean(predicted_class == target_class)


def loss(params, images, targets):
	preds = batched_predict(params, images)
	return jnp.mean( jnp.multiply(preds - targets, preds - targets) )


@jit
def update(params, x, y, learning_rate):
	grads = grad(loss)(params, x, y)
	return [(w - learning_rate * dw, b - learning_rate * db) for (w, b), (dw, db) in zip(params, grads)]

## Context-dependent gating models

@jit
def g_predict(params, gs, image):
	# per-example predictions
	activations = image
	for (w, b), g in zip(params[:-1], gs[:-1]):
		#print( jnp.shape(w), jnp.shape(g) )
		g_activations = jnp.multiply( g, activations )
		outputs = jnp.dot(w, g_activations) + b
		activations = relu(outputs)
  
	final_w, final_b = params[-1]
	final_g_activations = jnp.multiply( gs[-1], activations )
	return jnp.dot(final_w, final_g_activations) + final_b


g_batched_predict = vmap(g_predict, in_axes=(None, None, 0))


def g_loss(params, gs, images, targets):
	preds = g_batched_predict(params, gs, images)
	return jnp.mean( jnp.multiply(preds - targets, preds - targets) )


def get_adaptive_g(g1, g2, rho_g, key):
	new_g2 = []
	for g1tmp, g2tmp in zip(g1, g2):
		key, gckey = random.split(key, 2) 
		gg = random.bernoulli(gckey, rho_g, jnp.shape(g2tmp))
		gones = jnp.ones( jnp.shape(g2tmp) )
		new_g2.append( jnp.multiply(gg, g1tmp) + jnp.multiply(gones - gg, g2tmp) )
	return new_g2


@jit
def g_update(params, gs, x, y, learning_rate):
	grads = grad(g_loss)(params, gs, x, y)
	return [(w - learning_rate * dw, b - learning_rate * db) for (w, b), (dw, db) in zip(params, grads)]


#weight regularization in the Euclidian space
@jit
def update_wr_euc(params, target_params, x, y, learning_rate, lmbd):  
	grads = grad(loss)(params, x, y)
	return [( w - learning_rate * (dw + lmbd*(w-wo)), b - learning_rate * (db + lmbd*(b-bo)) ) for (w, b), (wo, bo), (dw, db) in zip(params, target_params, grads)]



def calc_phi(W1, b1, x):
	return relu( jnp.dot(W1, x) + b1 )

def calc_dphi(W1, b1, x):
	return drelu( jnp.dot(W1, x) + b1 )

batched_calc_phi = vmap(calc_phi, in_axes=(None, None, 0))
batched_calc_dphi = vmap(calc_dphi, in_axes=(None, None, 0))


#weight regularization in the Hessian (diagonal approximation)
@jit
def update_wr_diag(params, target_params, coeff_params, x, y, learning_rate, lmbd):
	grads = grad(loss)(params, x, y)
	#return [( w - learning_rate * (dw + lmbd*jnp.multiply(cw, w-wo)), b - learning_rate * (db + lmbd*jnp.multiply(cb, b-bo)) ) for (w, b), (wo, bo), (cw, cb), (dw, db) in zip(params, target_params, coeff_params, grads)]
	return [( w - learning_rate * (dw + lmbd*jnp.multiply(cw, w-wo)), b - learning_rate * db ) for (w, b), (wo, bo), cw, (dw, db) in zip(params, target_params, coeff_params, grads)]


@jit
def calc_coeff_params_diag(params, xs):
	W1, b1 = params[0]
	W2, b2 = params[1]
	
	phi = batched_calc_phi(W1, b1, xs)
	dphi = batched_calc_dphi(W1, b1, xs)

	cb2ones = jnp.ones( jnp.shape(b2) )
	phi_sq = jnp.multiply(phi, phi) 
	cW2 = jnp.outer( cb2ones, jnp.mean(phi_sq, axis=0) )
	
	dphixt = jnp.einsum('ij, ik->ijk', dphi, xs)
	dphixt_sq_m = jnp.mean( jnp.multiply(dphixt, dphixt), axis=0)
	w2m = np.sum( jnp.multiply(W2, W2), axis=0 )
	cW1 = jnp.einsum('i,ij->ij', w2m, dphixt_sq_m)
	
	cW2 = ( 1.0/jnp.max(cW2) )*cW2
	cW1 = ( 1.0/jnp.max(cW1) )*cW1
	
	coeff_params = []
	coeff_params.append(cW1); coeff_params.append(cW2)
	
	return coeff_params
	
	

@jit
def update_wr_fim(params, target_params, xxt, dyxt, phixt, phiphit, dphi_mat, zws, x, y, learning_rate, lmbd):
	W1, b1 = params[0]
	W1o, b1o = target_params[0]
	
	W2, b2 = params[1]
	W2o, b2o = target_params[1]
	
	dW1 = W1 - W1o
	dW2 = W2 - W2o

	W2tW2 = jnp.dot(W2o.T, W2o)
	W2tdW2 = jnp.dot(W2o.T, dW2)

	#dW2_reg = jnp.dot(dW2, phiphit) + jnp.dot( dyxt, jnp.dot(dW1.T, dphi_mat) )\
	#		+ jnp.dot( jnp.dot(W2o, dphi_mat), jnp.dot(dW1, phixt.T) )
	#dW1_reg = jnp.dot( jnp.dot( jnp.dot(dphi_mat, W2tW2), dphi_mat ), jnp.dot(dW1, xxt) ) \
	#		+ jnp.dot( jnp.dot(dphi_mat, dW2.T), dyxt) + jnp.dot( dphi_mat, jnp.dot(W2tdW2, phixt) )
	#layer-wise approximation
	dW2_reg = jnp.dot(dW2, phiphit)
	dW1_reg = jnp.dot( jnp.dot( jnp.dot(dphi_mat, W2tW2), dphi_mat ), jnp.dot(dW1, xxt) )
	
	Wregs = []
	Wregs.append( (1.0/zws[0])*dW1_reg ); Wregs.append( (1.0/zws[1])*dW2_reg )
	
	grads = grad(loss)(params, x, y)
	
	return [( w - learning_rate * (dw + lmbd*wreg), b - learning_rate * db ) for (w, b), wreg, (dw, db) in zip(params, Wregs, grads)]
	

@jit
def calc_coeff_params_fim(params, xs, ys_target):
	W1, b1 = params[0]
	W2, b2 = params[1]
	
	phi = batched_calc_phi(W1, b1, xs)
	dphi = batched_calc_dphi(W1, b1, xs)
	dys = batched_predict( params, xs ) - ys_target
	
	xxt = jnp.mean( jnp.einsum('ij,ik->ijk', xs, xs), axis=0 )
	dyxt = jnp.mean( jnp.einsum('ij,ik->ijk', dys, xs), axis=0 )
	phixt = jnp.mean( jnp.einsum('ij,ik->ijk', phi, xs), axis=0 )
	phiphit = jnp.mean( jnp.einsum('ij,ik->ijk', phi, phi), axis=0 )
	dphi_mat = jnp.diag( jnp.mean(dphi, axis=0) )
	
	W2tW2 = jnp.dot(W2.T, W2)
	zws = []
	zws.append( jnp.linalg.norm(W2tW2, 2)*jnp.linalg.norm(xxt, 2) )
	zws.append( jnp.linalg.norm(phiphit, 2) )
	
	return xxt, dyxt, phixt, phiphit, dphi_mat, zws



