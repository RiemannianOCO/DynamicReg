
import numpy as np
from .online_solver import OnlineSolver
from pymanopt.optimizers import *
from pymanopt import Problem,function
import time
class OnlineRCEG(OnlineSolver):
    def __init__(self) -> None:
       self.solver_type = 'OGD'
    
    def __str__(self) -> str:
        return 'roceg'

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
    
    def optimize(self,problem,X_0,eta= 0.01):
        T = problem.time
        track_list = []
        self.initial_with_problem(T,None,track_list)
        T = problem.time
        Y_curr = X_0
        for t in range(T):
            Y_curr, X_curr = self.expert_train(t,problem,Y_curr,eta=eta)
        return X_curr
