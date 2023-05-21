
import numpy as np

class RiemannianGDA:
    def __init__(self,problem,params) -> None:
        self.problem = problem
        self.params = params

    def train(self,max_iter):
        T =  max_iter
        self.loss = np.zeros(T)
        self.gnorm = np.zeros(T) 
        pass

    def optimize(self,x_0 = None, y_0 = None):
        p = self.params
        eta = p.eta

        prob = self.problem
        M = prob.M
        N = prob.N
        T = p.max_iter
        self.loss = np.zeros(T)
        self.gnorm = np.zeros(T)
        (x,y) = (x_0,y_0)
        for t in range(1,T):
            self.loss[t] = prob.f(x,y)
            (g_x,g_y) = prob.g(x,y)
            self.gnorm[t] =np.sqrt( M.inner_product(x,g_x,g_x) + N.inner_product(y,g_y,g_y) )
            x = M.exp(x,-eta * g_x)
            y = N.exp(y, p.nu*eta * g_y)
