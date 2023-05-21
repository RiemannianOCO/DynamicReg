from scipy.linalg import sqrtm,expm,inv
from numpy.linalg import norm
import numpy as np
from pymanopt.manifolds import SymmetricPositiveDefinite
class psd_transport(SymmetricPositiveDefinite):
    def __init__(self, n, k=1):
        super().__init__(n)
        self.center = np.eye(n)

    def transp(self,p,X,q):
        dist = self.dist(p,q)
        if dist < 1e-6:
            return X
        temp1 = self.log(p,q)
        p_sq = sqrtm(p)
        pinv_sq = inv(p_sq)
        temp2 = expm( 0.5 * pinv_sq @ temp1 @ pinv_sq)
        res = p_sq @ temp2 @ pinv_sq @ X @ pinv_sq @ temp2 @ p_sq
        return res
    
    