import numpy as np
from .online_solver import OnlineSolver
import time
class OnlineGradientDescent(OnlineSolver):
    def __init__(self) -> None:
       self.solver_type = 'OGD'

    def __str__(self) -> str:
        return 'rogd' 

    def optimize(self,problem,X_0):
        T = problem.time
        track_list = []
        self.initial_with_problem(T,None,track_list)
        return self.gradient_solver(problem,X_0)
      

    def gradient_solver(self,problem,X_0):
        T = problem.time
        D = problem.param.D
        G = problem.param.G
        zeta = problem.param.zeta
        mfd = problem.mfd
        #center = mfd.center
        eta = D/(G* (zeta) ** (0.5))
        X_t = X_0
        for t in range(T):
            value = problem.f_t(t,X_t)
            self.value_histories[t] = value
            eta_t = eta / ((t+1)**0.5)
            grad_t = problem.g_t(t,X_t)   #gradient
            X_t_plus_1 = mfd.exp(X_t, -eta_t * grad_t)
            if np.isnan(X_t_plus_1).any():
                raise ValueError
            X_t =  X_t_plus_1
        return X_t
        