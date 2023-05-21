
import numpy as np
from .online_solver import OnlineSolver
from pymanopt.optimizers import *
from pymanopt import Problem,function
import time
class RADRv(OnlineSolver):
    def __init__(self) -> None:
       self.solver_type = 'OGD'
    
    def __str__(self) -> str:
        return 'radrv'

    def expert_train(self,t,problem,Y_curr,eta):
        mfd = problem.mfd
        if t==0:
            grad_yt = mfd.random_tangent_vector(Y_curr)
        else:
        #print(value)
            grad_yt = problem.g_t(t-1,Y_curr) 
        X_next = mfd.exp(Y_curr, - eta * grad_yt )
        value = problem.f_t(t,X_next)
        self.value_histories[t] = value
        grad_xt = problem.g_t(t,X_next) 
        Y_next = mfd.exp(X_next,-eta * grad_xt + mfd.log(X_next,Y_curr))
        return Y_next,X_next
    
    def optimize(self,problem,X_0,beta,eta,N):
        mfd = problem.mfd
        T = problem.time
        track_list = []
        self.initial_with_problem(T,None,track_list)
        shape = (N,)+X_0.shape
        self.X = np.zeros( shape )
        self.Y = np.zeros( shape )

        self.W = np.ones(N)/N
        self.eta = np.zeros(N)
        self.L = np.zeros(N)
        self.M = np.zeros(N)
        X_t = X_0
        for i in range(N):
            tmp = mfd.random_point()
            self.Y[i] = X_0
            self.eta[i] = eta*(2)**(i)
        for t in range(T):
            for i in range(N):
                (self.Y[i] , self.X[i]) = self.expert_train(t,problem,self.Y[i],eta = self.eta[i])
            X_t = self.meta_train(t,problem,N,beta,X_t)
            value = problem.f_t(t,X_t)
            self.value_histories[t] = value
        return X_t
    
    def meta_train(self,t,problem,N,beta,X_t):
        mfd = problem.mfd
        bar_X_next = frechet_mean(mfd,self.X,self.W,X_t)
        if t==0:
            bar_g_t = problem.g_t(0,bar_X_next)
        else: 
            bar_g_t = problem.g_t(t-1,bar_X_next)
        for i in range(N):
            self.M[i] = mfd.inner_product(bar_X_next,bar_g_t,mfd.log(bar_X_next,self.X[i]))
            self.W[i] = np.exp(-beta*(self.L[i]+self.M[i]))
        self.W /= np.sum(self.W)
        X_next = frechet_mean(mfd,self.X,self.W,X_t) 
        g_t = problem.g_t(t,X_next)
        for i in range(N):
            self.L[i] += mfd.inner_product(X_next,g_t, mfd.log( X_next,self.X[i]))
        return X_next

def frechet_mean(mfd,X,W,x):
    N = X.shape[0]
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

    solver = SteepestDescent(min_gradient_norm = 1e-4,max_time = 100)
    solver._verbosity = 0
    off_problem = Problem(manifold = mfd, cost=func, riemannian_gradient=grad)
    center = solver.run(off_problem,initial_point =x).point
    return center

        