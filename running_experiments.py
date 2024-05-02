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
import cvxpy as cp

from infrastructure import *
from problem_building import *

#Construct MC Problem to see its original success rate
n=6

seed = int(1000 * time.time()) % 2**32
np.random.seed(seed)

M_star = build_gt(n)

(b, mask, mask_eps) = build_mc_instance_1(M_star,n,0)
#(b, mask, mask_eps) = build_mc_instance_2(n,0.2,0)

trial_num=20

dist_to_gt = np.zeros((2,trial_num))

optimizer = optax.adam(1e-2)

# np.random.seed(int(1000 * time.time()) % 2**32)
for trial in range(trial_num):
    _ , U_final,_= solve_MC((n,1,mask,b.reshape((n,n)),loss_MC), init_mag=1e-2,optimizer=optimizer,epochs=8000)
    dist_to_gt[0,trial] =LA.norm(U_final@U_final.T-M_star)

rate1 = len(dist_to_gt[0,dist_to_gt[0,:] <= 0.05])/len(dist_to_gt[0,:])
print(rate1)
# plt.figure()
# plt.hist(dist_to_gt[0,:], bins=20)


for n in [4,5,6,7,8]:

    M_star = build_gt(n)

    trial_num=1

    dist_to_gt = np.zeros((2,trial_num))

    for trial in range(trial_num):
        (b, mask, mask_eps) = build_mc_instance_1(M_star,n,0)
        #(b, mask, mask_eps) = build_mc_instance_2(M_star,n,0.15,0)

        X = cp.Variable((n, n), symmetric=True)
        constraints = [X >> 0]  # X is positive semidefinite
        constraints += [cp.multiply(mask,X) == b.reshape((n,n))]  # Trace of X equals 1

        # Define the objective
        objective = cp.Minimize(cp.trace(X))

        # Define the problem
        problem = cp.Problem(objective, constraints)

        # Solve the problem
        problem.solve(solver=cp.SCS)

        dist =LA.norm(X.value-M_star)
        print(dist)
        dist_to_gt[0,trial] =LA.norm(X.value-M_star)

    rate1 = len(dist_to_gt[0,dist_to_gt[0,:] <= 0.05])/len(dist_to_gt[0,:])
    print(rate1)


dist_to_gt_list = []
lr = 0.02
jit= True
# init_mag = 0.1
loss_list = []
level=2

n_list = np.array([4,5,6,7,8]) # when n=4,5,6 seems to be ok
p_list = np.array([0.15])
# init_mag_list = np.array([1e-2,1e-3,1e-4,1e-5])
init_mag = 1e-4
eps_list = np.array([5e-5])

success_rate_map = np.zeros((len(n_list),len(p_list),len(eps_list),2))

# eps = 1e-4

for (n,p,eps),(idx1,idx2,idx3) in zip(product(n_list,p_list,eps_list),product(np.arange(len(n_list)),np.arange(len(p_list)),np.arange(len(eps_list)))):
    
    M_star = build_gt(n)
    # z = np.random.normal(1,0.2,(n))
    # M_star = np.outer(z,z)

    prob_num=10
    trial_num = 20
    

    all_trajectories = []
    all_trajectories_lifted = []

    dist_to_gt = np.zeros((2,prob_num*trial_num))

    jit_level = level

    optimizer = optax.adam(lr)
    
    for prob in range(prob_num):
        b, mask, mask_eps = build_mc_instance_1(M_star,n,eps)

        for trial in range(trial_num):            
            A = output_A_MS(mask_eps,n)
            
            ##############################
            #unlifted problem
            _ , U_final,_= solve_MC((n,1,mask,b.reshape((n,n)),loss_MC), init_mag=1e-2,optimizer=optimizer)
            dist_to_gt[0,prob*trial_num+trial] =LA.norm(U_final@U_final.T-M_star)

            #############################
            #lifted problem 

            if jit == True:
                #needs to be changed for the noisy version of lifted problem
                this_get_grad = jax.jit(get_grad)
                this_get_grad(elevate_initialization(jnp.zeros(n),level),A,b,level).block_until_ready()
                this_loss = jax.jit(loss_fnc)
                this_loss(elevate_initialization(jnp.zeros(n),level),A,b,level).block_until_ready()
            else:
                this_get_grad = get_grad
                this_loss = loss_fnc
        
            
            _, w_final_lifted,(loss_vals,gradnormvals)= solve((this_get_grad,this_loss),(A,b,level,n), jax.random.PRNGKey(np.random.randint(10000)), init_mag,optimizer=optimizer,plot_gradnorm=False,plot_loss=False,loss_epsilon=1e-5)

            """
            this is to skip the solving step by just using 0 as the solution
            """
            dropped_sol = drop(w_final_lifted,epochs=2000)
            dist_to_gt[1,prob*trial_num+trial] = jnp.linalg.norm(np.outer(dropped_sol,dropped_sol)-M_star)

            loss_list.append(loss_vals)
            # dist_to_gt[1,trial] = min(LA.norm(w_final_lifted-elevate_initialization(z,level)),LA.norm(w_final_lifted-elevate_initialization(-z,level)))
    
    dist_to_gt_list.append(dist_to_gt)
    rate1 = len(dist_to_gt[0,dist_to_gt[0,:] <= 0.05])/len(dist_to_gt[0,:])
    rate2 = len(dist_to_gt[1,dist_to_gt[1,:] <= 0.05])/len(dist_to_gt[1,:])
    # plt.figure()
    # plt.hist(dist_to_gt[0,:])
    # plt.figure()
    # plt.hist(dist_to_gt[1,:])
    success_rate_map[idx1,idx2,idx3,0] = rate1
    success_rate_map[idx1,idx2,idx3,1] = rate2

print(success_rate_map)