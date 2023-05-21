import numpy as np

class RiemannianOptimisticGDAvg:
    def __init__(self,problem,params) -> None:
        self.problem = problem
        self.params = params

    def train(self):
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
        (last_x, last_y) = (x,y)
        glast_x, glast_y = prob.g(x,y)
        avx = x
        avy = y
        
        for t in range(1,T):
            (g_avx,g_avy) = prob.g(avx,avy)
            self.loss[t] = prob.f(avx,avy)
            self.gnorm[t] = np.sqrt(M.inner_product(avx,g_avx,g_avx) + N.inner_product(avy,g_avy,g_avy) )
            (g_x,g_y) = prob.g(x,y)
            v_x = -2 * eta * g_x + eta * M.transp(last_x,glast_x,x)
            v_y = 2 * eta * g_y  - eta * N.transp(last_y,glast_y,y)
            last_x = x
            last_y = y
            x = M.exp(x,v_x)
            y = N.exp(y,p.nu * v_y)
            glast_x = g_x
            glast_y = g_y
            avx = M.exp(avx, (1/(t+1) * M.log(avx , x) )   )
            avy = N.exp(avy, (1/(t+1) * N.log(avy , y) )   )
