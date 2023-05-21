from rzs.problem.problem import Problem
from manifold.SPD_transp import psd_transport
from manifold.Sn_transp import Sn_transport
from pymanopt.manifolds import Sphere
from scipy.linalg import logm,sqrtm,inv,eigh
import numpy as np

def dlogm(x,h):
    t = 1e-5
    res = (logm(x+t*h) - logm(x)) / t
    return res

def dlogm1(X,Y,v):
    t = 1e-10
    res =( func(X+t*v,Y) - func(X,Y) )/ t
    return res
def symm(x):
    return (0.5) * (x+x.T)

def func(x,Y):
    c = inv(sqrtm(x))
    res =  -2 * c @ logm( c @ Y @ c ) @ c
    return res

class GeometryPCA(Problem):
    def __init__(self, d, alpha, m) -> None:
        M = psd_transport(d,k=1)
        N = Sn_transport(d)
        super().__init__(M, N)
        self.m = m
        self.alpha = alpha
        self.d = d

    def f(self,x,y):
        m = self.m
        y = y.flatten()
        M = self.M
        num_sample = m.shape[0]
        fun =0
        for i in range(num_sample):
            fun += M.dist(x,m[i]) ** 2
        fun = fun *self.alpha / num_sample
        fun += y.T @ x @ y     
        return fun 
        
    def g(self,x,y): 
        d = self.d
        m = self.m
        M = self.M
        num_sample = m.shape[0]
        g_x = np.zeros((d,d))

        for i in range(num_sample):
            g_x -=  M.log(x,m[i])
        g_x = 2 * g_x * self.alpha /  num_sample
        g_x += x @ np.outer(y , y.T) @ x
        #g_y = np.zeros(0)
        g_y = (np.eye(d) - np.outer(y , y.T)) @ (2 * x @ y)
        return(g_x ,g_y )


    def h(self,x,y,vx,vy):
        t = 1e-4
        M = self.M
        N = self.N
        (g_x,g_y) = self.g(x,y)
        x_1 = M.exp(x,t*vx)
        y_1 = N.exp(y,t*vy)
        (g_x_1,g_y_1) = self.g(x_1,y_1)
        h_x = (M.transp(x_1,g_x_1,x) - g_x) / t
        h_y = (N.transp(y_1,g_y_1,y) - g_y) / t
        return(symm(h_x), N.projection(y,h_y))


    def ady(self,x):
        (w,v) = eigh(x)
        return v[-1]
    
    def ady_2(self,x):
        (w,v) = eigh(x)
        return v[0]
'''
    def h(self,x,y,vx,vy):
        d = self.d
        m = self.m
        m_inv_sq = self.m_inv_sq
        num_sample = m.shape[0]
        M = self.M
        #d_yy = 2 * (np.eye(d) - np.outer(y,y.T)) @ x @ vy + 2 * ( y.T @ x @ y ) * vy
        #d_yx = (np.eye(d) - np.outer(y,y.T)) @ (2 * vx @ y)
        #h_y = d_yy + d_yx
        h_y =np.zeros(d)
        x_sq = sqrtm(x)
        d_xx = np.zeros((d,d))
        for i in range(num_sample):
            g_x  =  -2 * M.log(x,m[i])  
            ehess = dlogm1(x,m[i],vx)
            d_xx = x @ ehess @ x +symm(vx @g_x @x)
            #temp1 = x @ m_inv_sq[i] @ m_inv_sq[i]
            #temp2 = logm(temp1)
            #dxgx = -2 * temp2 @ vx - 2 * dlogm(temp1, vx @ m_inv_sq[i] @ m_inv_sq[i] ) @ x
            #gx = -  -symm (2 *temp2 @ x)
            #corr = vx @ inv(x) @ gx
            #d_xx += symm(dxgx) - symm(corr)
        d_xx = d_xx * self.alpha /  num_sample
        #corr2 = -vx @ np.outer(y,y.T) @ x
        #d_xx -= 2 * symm(vx @ np.outer(y,y.T) @ x) - symm(corr2)
        #d_xy = -2 * x @ symm( np.outer(vy,y.T) ) @ x
        h_x = d_xx 

        return(-h_x,h_y)
'''