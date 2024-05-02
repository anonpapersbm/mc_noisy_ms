import jax
import jax.numpy as jnp
# from jax.config import config
# config.update("jax_enable_x64", True)
import optax
import numpy as np
from scipy import linalg as LA
from sklearn.cluster import KMeans
import time
from itertools import chain, repeat, product
import matplotlib.pyplot as plt

from infrastructure import *

def output_A_MS(mask,n):
    #outputs a n*n, n,n tensor
    #specifically for converting mask to A for m=n**2
    n = mask.shape[0]
    A = np.zeros((n**2,n,n))
    for [i,j] in product(np.arange(n),repeat=2):
        temp = np.zeros((n,n))
        temp[i,j]=mask[i,j]
        temp = (temp+temp.T)/2
        A[i*n+j,:,:]=temp

    return A

def loss_MC(U,b,mask):
    #U is n by r, b is n by n, mask is n by n
    loss = jnp.sum(mask*(U@U.T-b)**2)
    
    return loss


def solve_MC(problem, init_mag = 1e-1, init_point = None, lr=0.01, optimizer = None, plot_gradnorm = False, plot_loss = False, epochs=2000, loss_epsilon=1e-8, gradnorm_epsilon=1e-8):

    #d is the number of neurons we want to use
    n,r,mask,b,loss_fnc = problem
    
    if init_point is None:
        init_point = jax.random.normal(jax.random.PRNGKey(np.random.randint(0,2**31-1)), (n,r))
    #normalization
    init_point = init_mag * init_point/jnp.linalg.norm(init_point)

    if optimizer is not None:
        opt_state = optimizer.init(init_point)


    gradnorms = np.zeros(epochs)
    losses = np.zeros(epochs)

    l_g_fn = jax.value_and_grad(loss_fnc)
    U = init_point
    jit_l_g_fn = jax.jit(l_g_fn)
    # the main training loop
    for epoch in range(epochs):
        loss, grad = jit_l_g_fn(U,b,mask)

        grad = np.nan_to_num(grad)

        gradnorms[epoch] = jnp.linalg.norm(grad)
        losses[epoch] = loss

        if loss < loss_epsilon:
            break
        if optimizer is None:
            U = U - lr*grad
        else:
            updates, opt_state = optimizer.update(grad, opt_state)
            U = optax.apply_updates(U, updates)
    
    if plot_gradnorm:
        plt.figure()
        plt.yscale("log")
        plt.plot(np.arange(epochs),gradnorms)

    if plot_loss:
        plt.figure()
        plt.yscale("log")
        plt.plot(np.arange(epochs),losses)
        

    return loss, U, (losses,gradnorms)

def build_gt(n):
    #for specific dimension
    z = np.zeros(n)
    for k in range(int(np.ceil(n/2))):
        z[2*k]=1

    M_star = np.outer(z,z)

    return M_star

#Replicates (4) in the paper. This is a instance that is hard to solve in BM.
def build_mc_mask(n,eps):
    mask = np.ones((n,n))*eps
    for i in range(n):
        mask[i,i]=1
        for k in (np.arange(int(np.floor(n/2)))+1):
            mask[i,2*k-1]=1
            mask[2*k-1,i]=1

    return mask

def build_mc_instance_1(M_star,n,eps):

    mask = build_mc_mask(n,0)
    b_mtr = mask*M_star #in matrix shape

    mask_eps = build_mc_mask(n,eps)

    return ( b_mtr.reshape(-1), mask, mask_eps)


#In this instance we assume bernoulli observation of entires, with probability p 

def build_mc_instance_2(M_star,n,p,eps):

    mask = np.random.binomial(1, p, (n,n))
    b_mtr = mask*M_star #in matrix shape

    mask_eps = np.minimum(mask+eps,1)

    return (b_mtr.reshape(-1), mask, mask_eps)