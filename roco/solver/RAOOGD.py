import numpy as np
from .online_solver import OnlineSolver
from pymanopt.optimizers import *
from pymanopt import Problem,function
from pymanopt.manifolds import Product
import time
class RAOOGD(OnlineSolver):
    def __init__(self) -> None:
       self.solver_type = 'OGD'
 
    def __str__(self) -> str:
        return 'raoogd'

    def expert_train(self,t,problem,X_past,X_curr,past_norm,eta):
        mfd = problem.mfd
        value = problem.f_t(t,X_curr)
        self.value_histories[t] = value
        #print(value)
        grad_t = problem.g_t(t,X_curr) 
        X_next = mfd.exp(X_curr, -2 * eta * grad_t + eta* mfd.transp(X_past, past_norm, X_curr))
        return X_next,X_curr,grad_t
    
    def optimize(self,problem,X_0,beta,eta,N):
        mfd = problem.mfd
        T = problem.time
        track_list = []
        self.initial_with_problem(T,None,track_list)
        g_init = problem.g_t(0,X_0)
        self.X_curr = [X_0] * N
        self.X_past = [X_0] * N
        self.past_grad = [g_init] *N
        self.W_past = np.ones(N)/N
        self.W_curr = np.ones(N)/N
        self.eta = np.zeros(N)
        self.L = np.zeros(N)
        self.M = np.zeros(N)
        X_t = X_0
        for i in range(N):
            self.X_curr[i] = X_0
            self.X_past[i] = X_0
            self.past_grad[i] = g_init
            self.eta[i] = eta*(2)**(i)
        for t in range(T):
            #print(t)
            for i in range(N):
                (self.X_curr[i],self.X_past[i],self.past_grad[i]) = \
                    self.expert_train(t,problem,X_past=self.X_past[i],X_curr=self.X_curr[i],
                                      past_norm=self.past_grad[i],eta=self.eta[i])
            X_t = self.meta_train(t,problem,N,beta,X_t)
            value = problem.f_t(t,X_t)
            self.value_histories[t] = value
        return X_t
    
    def meta_train(self,t,problem,N,beta,X_t):
        mfd = problem.mfd
        bar_X_next = frechet_mean(mfd,self.X_curr,self.W_curr,X_t)
        self.W_past = self.W_curr
        if t==0:
            bar_g_t = problem.g_t(0,bar_X_next)
        else: 
            bar_g_t = problem.g_t(t-1,bar_X_next)
        
        for i in range(N):
            self.M[i] = mfd.inner_product(bar_X_next,bar_g_t,mfd.log(bar_X_next,self.X_curr[i]))
            self.W_curr[i] = np.exp(-beta*(self.L[i]+self.M[i]))

        self.W_curr /= np.sum(self.W_curr)
        d = mfd.zero_vector(bar_X_next)
        for i in range(N):
            d += (self.W_curr[i] - self.W_past[i]) * mfd.log(bar_X_next,self.X_curr[i])
        #X_next = mfd.exp( bar_X_next, d )
        X_next = frechet_mean(mfd,self.X_curr,self.W_curr,X_t) 
        g_t = problem.g_t(t,X_next)
        for i in range(N):
            self.L[i] += mfd.inner_product(bar_X_next,mfd.transp(X_next,g_t,bar_X_next),mfd.log(bar_X_next,self.X_curr[i]))
        return X_next

def frechet_mean(mfd,X,W,x):

    N = W.shape[0]
    @function.numpy(mfd)
    def func(u): 
        distance = 0
        for i in range(N):
            distance += W[i] * mfd.dist(u,X[i]) ** 2
        return distance
    
    @function.numpy(mfd)
    def grad(u): 
        grad = mfd.zero_vector(u)
        for i in range(N):
            grad -= W[i]*  mfd.log(u,X[i])
        return grad

    solver = SteepestDescent(min_gradient_norm = 1e-4,max_time = 1000)
    solver._verbosity = 0
    off_problem = Problem(manifold = mfd, cost=func, riemannian_gradient=grad)
    center = solver.run(off_problem,initial_point =x).point
    return	center
   