from rzs.problem.problem import Problem
from manifold.SPD_transp import psd_transport
from numpy.linalg import det,inv
import numpy as np
from rzs.config import D
class Logdet(Problem):
    def __init__(self, d, c1, c2) -> None:
        M = psd_transport(d,k=1)
        N = psd_transport(d,k=1)
        super().__init__(M, N)
        self.c1 = c1
        self.c2 = c2
        self.d = d

    def f(self,x,y):
        assert not (np.isnan(x)).any()
        assert not (np.isnan(x)).any()

        logdetx = np.log(det(x))
        logdety = np.log(det(y))
        
        return  (self.c1 * logdetx **2 + self.c2 * logdetx * logdety - self.c1 * logdety **2) / self.d 
    
    def egrad(self,x,y):
        assert not (np.isnan(x)).any()
        assert not (np.isnan(x)).any()
        logdetx = np.log(det(x))
        logdety = np.log(det(y))

        egradx = ( self.c1 * 2 * logdetx + self.c2 * logdety ) * inv(x) / self.d
        egrady = ( -self.c1 * 2 * logdety + self.c2 * logdetx ) * inv(y) / self.d
        
        return(egradx ,  egrady)
    def g(self,x,y):
        (egradx ,  egrady) = self.egrad(x,y)
        gradx = self.M.euclidean_to_riemannian_gradient(x,egradx) 
        grady = self.N.euclidean_to_riemannian_gradient(y,egrady) 
        return (gradx,grady) 

  
    def h(self,x,y,vx,vy):
        (egradx ,  egrady) = self.egrad(x,y)
        logdetx = np.log(det(x))
        logdety = np.log(det(y))
        invx = inv(x)
        invy = inv(y)

        ehessx = ( 2 * self.c1 * np.trace(vx @ invx) ) * invx - (self.c1 * 2 * logdetx + self.c2 * logdety) * (invx @ vx @ invx) + (self.c2 * np.trace(vy @ invy)) * invx
        ehessx /= self.d
        ehessy = ( self.c2 * np.trace(vx @ invx) ) * invy + (self.c2 * logdetx - 2*self.c1 *logdety) * (invy @ vy @ invy) - (2 * self.c1 * np.trace( vy @ invy)) * invy
        ehessy /= self.d

        hessx = self.M.euclidean_to_riemannian_hessian(x,egradx,ehessx,vx)
        hessy = self.N.euclidean_to_riemannian_hessian(x,egrady,ehessy,vy)

        return(hessx,hessy)

    def ady(self,x):
        id = np.eye(self.d)
        logdetx = np.log(det(x))

        if logdetx >= 0:
            return np.exp(D  / self.d ) * id
        else:
            return np.exp(-D / self.d )  * id
            #return id