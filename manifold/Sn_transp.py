from scipy.linalg import sqrtm,expm,inv
from numpy.linalg import norm
import numpy as np
from pymanopt.manifolds import Sphere
class Sn_transport(Sphere):
    def __init__(self, n):
        super().__init__(n)
        self.center = np.eye(n)

    def transp(self,p,X,q):
        dist = self.dist(p,q)
        if dist < 1e-6:
            return self.projection(q, X)
       
        logpq = self.log(p,q)
        logqp = self.log(q,p)
       
        res  = X - self.inner_product(None,X,logpq) / (dist ** 2) * (logqp + logpq)
        return res
    
    transport = transp

'''
s = Sn_transport(3)
x = np.array([1,0,0])
y = np.array([0,1,0])
v = np.array([0,1,1])
print(s.transp(x,v,y))
'''