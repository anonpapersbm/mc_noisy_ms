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

class vanillaGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def init(self, starting_point, lr = 0):
        if lr != 0:
            self.learning_rate = lr
        return self.learning_rate

    def update(self, gradients, opt_state):
        updates = -self.learning_rate * gradients
        return updates, opt_state

class vanillaPGD:
    def __init__(self, learning_rate, r, n , l, g_thres = 1e-5, t_thres = 200):
        self.learning_rate = learning_rate
        self.r = r
        self.n = n
        self.l = l
        self.g_thres = g_thres
        self.t_thres = t_thres

    def init(self, starting_point, lr=0):
        if lr != 0:
            self.learning_rate = lr
        return {'curr_iter':0, 't_noise':0}

    def update(self, gradients, opt_state):
        curr_iter = opt_state['curr_iter']+1
        t_noise = opt_state['t_noise']
        # if jnp.linalg.norm(gradients) < self.g_thres and curr_iter - t_noise > self.t_thres:
        #     t_noise = curr_iter
        #     randint = np.random.randint(0,2**31-1)
        #     noise = jax.random.normal(jax.random.PRNGKey(randint), shape=(self.n,))
        #     noise = elevate_initialization(noise, self.l)
        #     noise = self.r*noise / (jnp.sqrt(self.n*self.l)*jnp.linalg.norm(noise))
        #     updates = -self.learning_rate * (noise+gradients)
        # else:
        randint = np.random.randint(0,2**31-1)
        noise = jax.random.normal(jax.random.PRNGKey(randint), shape=(self.n,))
        noise = elevate_initialization(noise, self.l)
        noise = self.r*noise / (jnp.sqrt(self.n*self.l)*jnp.linalg.norm(noise))
        updates = -self.learning_rate * (noise+gradients)
        return updates, {'curr_iter':curr_iter, 't_noise':t_noise}

class customGD:
    def __init__(self, learning_rate, n, r, l, prob_params,loss,g_thres = 1e-5, buffer=100, beta=0.5, gamma=0.5, eta_0 = None):
        self.learning_rate = learning_rate
        self.n = n
        self.r = r
        self.l = l
        self.loss = loss
        self.A, self.b = prob_params #unlifted sensing matrices and measurements
        self.g_thres = g_thres #gradient threshold for saddle point detection
        self.escape_saddle = False
        self.buffer_limit = buffer
        self.buffer_step = 0
        self.beta = beta #for backtracking line search
        self.gamma = gamma #for backtracking line search
        if eta_0 is None:
            self.eta_0 = learning_rate #for backtracking line search
        else:
            self.eta_0 = eta_0


    def init(self, starting_point ,lr=0):
        if lr != 0:
            self.learning_rate = lr
        return {'curr_iter':0, 't_noise':0, 'curr_w':starting_point}

    def update(self, gradients, opt_state):
        curr_iter = opt_state['curr_iter']+1
        t_noise = opt_state['t_noise']
        curr_w = opt_state['curr_w']
        if jnp.linalg.norm(gradients) < self.g_thres and curr_iter > 100:
            if self.escape_saddle:
                t_noise = curr_iter
                dropped_x = jnp.nan_to_num(drop(curr_w,epochs=500)) #500 is good enough for n<10? But could be adjusted larger.
                dropped_x = unvec(dropped_x, self.n, self.r)
                xx_top = dropped_x@dropped_x.T
                grad_x = jnp.kron(jnp.eye(self.r),jnp.einsum(self.A,[0,1,2],xx_top,[1,2], self.A,[0,3,4]) - jnp.einsum(self.b, [0], self.A,[0,1,2]))
                eigs, eigvec = jnp.linalg.eigh(grad_x)
                direction = elevate_initialization(eigvec[:,0], self.l) #corresponding to the least eigenvalue (should be negatuve)
                #now we want to do backtracking line search
                this_eta = self.eta_0
                while self.loss(curr_w+this_eta*direction,self.A,self.b,self.l) > self.loss(curr_w,self.A,self.b,self.l) + self.beta*this_eta*jnp.inner(gradients.reshape(-1),direction.reshape(-1)):
                    this_eta = this_eta*self.gamma
                updates = this_eta * (direction)
                self.escape_saddle = False
            else:
                self.buffer_step += 1
                if self.buffer_step == self.buffer_limit:
                    self.escape_saddle = True
                    self.buffer_step = 0
                updates = -self.learning_rate * gradients
        else:
            self.escape_saddle = False
            updates = -self.learning_rate * gradients
        return updates, {'curr_iter':curr_iter, 't_noise':t_noise,'curr_w': curr_w+ updates}
    

def adam_optimize(problem,
          lr,
          epochs,
          gradnorm_epsilon,
          loss_epsilon = float('-inf')
          ):
    f, w, params = problem
    optimizer = optax.adam(lr)
    # Obtain the `opt_state` that contains statistics for the optimizer.
    opt_state = optimizer.init(w)

    l_g_fn = jax.value_and_grad(f)

    # the main training loop
    for _ in range(epochs):
        loss, grads = l_g_fn(w, params)
        grad1, grad2 = grads
        if jnp.abs(grad1) > 10:
            grad1 = 0
        if loss < loss_epsilon or jnp.linalg.norm(grad2) < gradnorm_epsilon:
             break
        updates, opt_state = optimizer.update((grad1,grad2), opt_state)
        w = optax.apply_updates(w, updates)

    return loss, grads, w

def tensor_PCA(tensor,
                        lr, #=0.05,
                        epochs, #=2000,
                        gradnorm_epsilon,
                        lambd_v=None, key=None):
    # either v or key must be not None

    def loss(eigenval_eigenvec, tensor):
        lambd, v = eigenval_eigenvec
        k = len(tensor.shape)
        for _ in range(len(tensor.shape)):
            tensor = jnp.inner(tensor, v)
        first_term = jnp.square(lambd)*jnp.power(jnp.linalg.norm(v), 2*k)
        res = first_term - 2*lambd*tensor
        return res

    s = tensor.shape[0]
    if lambd_v is None:
        key1, key2 = jax.random.split(key)
        v = jax.random.normal(key1, shape=(s,))/jnp.sqrt(s)
        lambd = 0.001*jax.random.normal(key2, shape=())
    else:
        lambd, v = lambd_v

    loss, grads, lambd_v = adam_optimize((loss, (lambd, v), tensor),
                  lr=lr,
                    epochs=epochs,
                    gradnorm_epsilon=gradnorm_epsilon)
    lambd, v = lambd_v
    sign = jnp.sign(lambd)

    return sign*jnp.power(jnp.abs(lambd), 1/len(tensor.shape))*v

def drop(w,
         lr=0.005,
         epochs=1000,
         gradnorm_epsilon=1e-6):
    n = w.shape[0]
    key=jax.random.PRNGKey(np.random.randint(0,2**31-1))
    w = tensor_PCA(w,
                            lr=lr,
                            epochs=epochs,
                            gradnorm_epsilon=gradnorm_epsilon,
                            key=key)
    return w.reshape(n,)


def elevate_initialization(w_in, level,flatten=False):
    '''
    w_in.shape = (n,)
    w_out.shape = (n, .. level+1 times .., n)
    '''
    w_in = w_in.reshape(w_in.shape[0],)
    einsum_w_args = list(chain(*[(w_in, (i,)) for i in range(level+1)]))
    w_norank = jnp.einsum(*einsum_w_args)
    if flatten:
        w_norank = w_norank.reshape(jnp.prod(jnp.asarray(w_norank.shape)))
    return w_norank

def elevate_matrices(X,level,flatten=False):
    '''
    X.shape = (n, m)
    X_lifted.shape = (n, .. level+1 times .., m)
    '''
    einsum_X_args = list(chain(*[(X, (i*2,i*2+1)) for i in range(level+1)]))
    X_lifted = jnp.einsum(*einsum_X_args)
    if flatten:
        X_lifted = X_lifted.reshape(jnp.prod(jnp.asarray(X_lifted.shape)))
    return X_lifted

def elevate_A(A, level,flatten=False):
    einsum_A_args = list(chain(*[(A, (i*3,i*3+1,i*3+2)) for i in range(level+1)]))
    A_lifted = jnp.einsum(*einsum_A_args)

    return A_lifted

def elevate_AA(A, level):
    A_topA = jnp.einsum(A,[0,1,2],A,[0,3,4])
    einsum_AA_args = list(chain(*[(A_topA, (i*4,i*4+1,i*4+2,i*4+3)) for i in range(level+1)]))
    A_topA_lifted = jnp.einsum(*einsum_AA_args)

    return A_topA_lifted

def data_loss_reference(w, z_lifted, A_lifted,lvl):
    '''w.shape = (n, .. level+1 times .., n)
    We want to take in flattened w, since we want hessian as matrix, but z and A can be tensors
    '''
    n = z_lifted.shape[0]
    if len(w.shape) == 1:
        w = jnp.reshape(w,tuple(n for i in range(lvl+1)))

    idx_A = [i for i in range(3*(lvl+1))]
    idx_1 = [1+3*i for i in range(lvl+1)]
    idx_2 = [2+3*i for i in range(lvl+1)]
    
    Aww = jnp.einsum(A_lifted,idx_A,w,idx_1,w,idx_2)

    Azz = jnp.einsum(A_lifted,idx_A,z_lifted,idx_1,z_lifted,idx_2)

    diff = jnp.reshape(Aww-Azz,-1)

    loss = 0.25*jnp.dot(diff,diff)

    return loss


def get_grad_reference(w, z_lifted, A_topA_lifted,lvl):
    #Get the gradient in closed form, without using auto-diff package
    n = z_lifted.shape[0]
    if len(w.shape) == 1:
        w = jnp.reshape(w,tuple(n for _ in range(lvl+1)))

    #uncomment this line if you want jit acceleration
    #lvl = level
    
    # idx_A = [i for i in range(4*(lvl+1))]
    # idx_1 = [4*i for i in range(lvl+1)]
    # idx_2 = [1+4*i for i in range(lvl+1)]
    # idx_3 = [2+4*i for i in range(lvl+1)]
    idx_A = list(np.arange(4*(lvl+1)))
    idx_1 = list(4*np.arange(lvl+1))
    idx_2 = list(4*np.arange(lvl+1)+1.0)
    idx_3 = list(4*np.arange(lvl+1)+2.0)
    
    Awww = jnp.einsum(A_topA_lifted,idx_A,w,idx_1,w,idx_2,w,idx_3)

    Azzw = jnp.einsum(A_topA_lifted,idx_A,z_lifted,idx_1,z_lifted,idx_2,w,idx_3)

    grad = Awww - Azzw

    return grad


def vec(X):
    #return the vectorized version of X, column stacked, as congruent to math notation
    return X.T.reshape(-1)

def unvec(X,n,r):
    #return the unvectorized version of X, column stacked, as congruent to math notation
    return X.reshape(r,n).T

def constructP(n,r):
    #this shape for easier indexing, but will need to be transposed
    P = np.zeros((r,n,n*r))
    for i in range(r):
        P[i,:,i*n:(i+1)*n] = np.eye(n)

    return np.swapaxes(P,0,1)

def loss_func_highrank(w,A,b,lvl):
    # A is tensor of shape m by n by n
    if jit == True:
        lvl = jit_level
        
    m,n,_ = A.shape
    nr = w.shape[0]
    r = int(nr/n)

    P = constructP(n,r)

    args = []
    for i in range(lvl+1):
        args += [P,(i*3,i*3+1,i*3+2)]

    args += [w,tuple(np.arange(lvl+1)*3+2)]
    X_lifted = jnp.einsum(*args)

    idx = np.arange(2*(lvl+1))*2
    idx[np.arange(lvl+1)*2+1] += 1
    ww_top = jnp.einsum(X_lifted,tuple(np.arange(2*(lvl+1))*2+1),X_lifted,tuple(idx))

    args1 = [ww_top,tuple(np.arange(2*(lvl+1)))]
    for i in range(lvl+1):
        args1 += [A,(2*(lvl+1)+i,i*2,i*2+1)]
    Aww_top = jnp.einsum(*args1)

    args2 = []
    for i in range(lvl+1):
        args2 += [b,[i]]
    b_lifted = jnp.einsum(*args2)
    return 0.5*jnp.linalg.norm(Aww_top-b_lifted)**2


#loss function for general lifted matrix sensing problem
def loss_fnc(w, A,b,lvl):
    #calculate the loss in a memory-saving way
    #Does not take in flattened w, but w as a tensor

    if jit == True:
        lvl = jit_level
    
    A_topA = jnp.einsum(A,[0,1,2],A,[0,3,4])
    b_lifted_args = list(chain(*[(b, (i,)) for i in range(lvl+1)]))
    b_lifted = jnp.einsum(*b_lifted_args)

    # args1 = list(chain(*[(A_topA,(i*4,i*4+1,i*4+2,i*4+3)) for i in range(lvl+1)]))

    args1 = []
    for i in range(lvl+1):
        args1 += [A_topA,(i*4,i*4+1,i*4+2,i*4+3)]
    args2 = [w,tuple(np.arange(lvl+1)*4)]
    args3 = [w,tuple(np.arange(lvl+1)*4+1)]
    args4 = [w,tuple(np.arange(lvl+1)*4+2)]
    args5 = [w,tuple(np.arange(lvl+1)*4+3)]

    args = args1+args2+args3+args4+args5
    
    Awwww = jnp.einsum(*args)

    args1 = []
    for i in range(lvl+1):
        args1 += [A,(i*3,i*3+1,i*3+2)]

    args2 = [b_lifted,tuple(np.arange(lvl+1)*3)]
    args3 = [w,tuple(np.arange(lvl+1)*3+1)]
    args4 = [w,tuple(np.arange(lvl+1)*3+2)]
    

    args = args1+args2+args3+args4

    Azzww = jnp.einsum(*args)

    Azzzz = jnp.linalg.norm(b_lifted)**2


    loss = 0.25*(Awwww+Azzzz-2*Azzww)

    return loss


def get_grad(w, A, b,lvl):
    #Get the gradient in closed form, without using auto-diff package
    #Does not take in flattened w
    
    if jit == True:
        lvl = jit_level
    
    A_topA = jnp.einsum(A,[0,1,2],A,[0,3,4])
    b_lifted_args = list(chain(*[(b, (i,)) for i in range(lvl+1)]))
    b_lifted = jnp.einsum(*b_lifted_args)

    ####################

    args1 = []
    for i in range(lvl+1):
        args1 += [A_topA,(i*4,i*4+1,i*4+2,i*4+3)]
    args2 = [w,tuple(np.arange(lvl+1)*4)]
    args3 = [w,tuple(np.arange(lvl+1)*4+1)]
    args4 = [w,tuple(np.arange(lvl+1)*4+2)]

    args = args1+args2+args3+args4
    
    Awww = jnp.einsum(*args)

    ####################

    args1 = []
    for i in range(lvl+1):
        args1 += [A,(i*3,i*3+1,i*3+2)]

    args2 = [b_lifted,tuple(np.arange(lvl+1)*3)]
    args3 = [w,tuple(np.arange(lvl+1)*3+1)]

    args = args1+args2+args3

    Azzw = jnp.einsum(*args)

    grad = Awww - Azzw

    return grad


def solve(grad_loss, problem_params, key, init_mag, lr=0.01, optimizer = None, w_0=None, plot_gradnorm = False, plot_loss = False, epochs=1000, loss_epsilon=1e-7, gradnorm_epsilon=1e-5):
    
    get_grad, get_loss = grad_loss
    A,b,level,n = problem_params

    lift_key, drop_key = jax.random.split(key)
    if w_0 is not None:
        if len(w_0.shape) < level+1:
            w = elevate_initialization(w_0, level)
        else:
            w = w_0
    else:
        w_0 = jax.random.normal(lift_key, shape=(n,))
        if level>0:
            w = elevate_initialization(w_0, level)
        else:
            w = w_0
    w = w/jnp.sqrt(jnp.prod(jnp.asarray(w.shape)))
    w = w*init_mag #changes to appropriate scale

    if optimizer is not None:
        opt_state = optimizer.init(w)

    gradnorms = np.zeros(epochs)
    losses = np.zeros(epochs)
    # the main training loop
    for epoch in range(epochs):
        grad = get_grad(w,A,b,level)
        loss = get_loss(w,A,b,level)

        grad = np.nan_to_num(grad)

        gradnorms[epoch] = jnp.linalg.norm(grad)
        losses[epoch] = loss

        if loss < loss_epsilon:
            break
        if optimizer is None:
            w = w - lr*grad
        else:
            updates, opt_state = optimizer.update(grad, opt_state)
            w = optax.apply_updates(w, updates)
    
    if plot_gradnorm:
        plt.figure()
        plt.yscale("log")
        plt.plot(np.arange(epochs),gradnorms)

    if plot_loss:
        plt.figure()
        plt.yscale("log")
        plt.plot(np.arange(epochs),losses)
        

    return loss, w, (losses,gradnorms)


def solve_highr(grad_loss, problem_params, key, init_mag, lr=0.01, optimizer = None, w_0=None, plot_gradnorm = False, plot_loss = False, epochs=1000, loss_epsilon=1e-7, gradnorm_epsilon=1e-5):
    
    get_grad, get_loss = grad_loss

    A,b,level,n,r = problem_params

    lift_key, drop_key = jax.random.split(key)
    if w_0 is not None:
        if len(w_0.shape) < level+1:
            w = elevate_initialization(w_0, level)
        else:
            w = w_0
    else:
        w_0 = jax.random.normal(lift_key, shape=(n*r,))
        if level>0:
            w = elevate_initialization(w_0, level)
        else:
            w = w_0
    w = w/jnp.sqrt(jnp.prod(jnp.asarray(w.shape)))
    w = w*init_mag #changes to appropriate scale

    if optimizer is not None:
        opt_state = optimizer.init(w)

    gradnorms = np.zeros(epochs)
    losses = np.zeros(epochs)
    # the main training loop
    for epoch in range(epochs):
        grad = get_grad(w,A,b,level)
        loss = get_loss(w,A,b,level)

        grad = np.nan_to_num(grad)

        gradnorms[epoch] = jnp.linalg.norm(grad)
        losses[epoch] = loss

        if loss < loss_epsilon:
            break
        if optimizer is None:
            w = w - lr*grad
        else:
            updates, opt_state = optimizer.update(grad, opt_state)
            w = optax.apply_updates(w, updates)
    
    if plot_gradnorm:
        plt.figure()
        plt.yscale("log")
        plt.plot(np.arange(epochs),gradnorms)

    if plot_loss:
        plt.figure()
        plt.yscale("log")
        plt.plot(np.arange(epochs),losses)
        

    return loss, w, (losses,gradnorms)
