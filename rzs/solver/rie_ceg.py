
import numpy as np

class RiemannianCEG:
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
        x = x_0
        y = y_0
        T = p.max_iter
        ( g_x , g_y ) = prob.g(x,y)
        w = M.exp(x , -eta * g_x)
        z = N.exp(y ,  eta * g_y)
        self.loss = np.zeros(T)
        self.gnorm = np.zeros(T) 
        avw = w
        avz = z
        for t in range(1,T):
           
            ( g_x , g_y ) = prob.g(x,y)
            w = M.exp(x , -eta * g_x)
            z = N.exp(y , p.nu* eta * g_y)

            ( g_w , g_z ) = prob.g(w,z)
            v_w = eta *  -g_w + M.log(w,x)
            v_z = eta *  g_z + N.log(z,y)

            x = M.exp(w, v_w)
            y = N.exp(z, p.nu* v_z)

            self.loss[t] = prob.f(x,y)
            self.gnorm[t] = np.sqrt(M.inner_product(x,g_x,g_x) + N.inner_product(y,g_y,g_y))