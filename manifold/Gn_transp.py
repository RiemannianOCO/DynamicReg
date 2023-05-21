from scipy.linalg import sqrtm,expm,inv
from numpy.linalg import norm
import numpy as np
from pymanopt.manifolds import Grassmann
class Gn_transport(Grassmann):
    def __init__(self, n,p):
        super().__init__(n,p)
        
    def transp(self,p,X,q):

        return self.projection(q, X)

    
    transport = transp
