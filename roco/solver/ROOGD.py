import numpy as np
from .online_solver import OnlineSolver
from pymanopt.optimizers import *
class ROOGD(OnlineSolver):
    def __init__(self) -> None:
       self.solver_type = 'OGD'
    
    def __str__(self) -> str:
        return 'roogd'

    def expert_train(self,t,problem,X_past,X_curr,past_norm,eta):
        mfd = problem.mfd
        value = problem.f_t(t,X_curr)
        self.value_histories[t] = value
        #print(value)
        grad_t = problem.g_t(t,X_curr) 
        X_next = mfd.exp(X_curr, -2 * eta * grad_t + eta* mfd.transp(X_past, past_norm, X_curr))
        return X_next,X_curr,grad_t
    
    def optimize(self,problem,X_0,eta= 0.01):
        T = problem.time
        track_list = []
        self.initial_with_problem(T,None,track_list)
        T = problem.time
        X_curr = X_0
        X_past = X_0
        past_norm = problem.g_t(0,X_0)
        for t in range(T):
            (X_curr,X_past,past_norm) = self.expert_train(t,problem,X_past=X_past,X_curr=X_curr,past_norm=past_norm,eta=eta)
        return X_curr