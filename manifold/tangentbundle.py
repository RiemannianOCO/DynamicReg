from scipy.linalg import sqrtm,expm,inv
from numpy.linalg import norm
import numpy as np
from pymanopt.manifolds import Product,Euclidean
from pymanopt.manifolds.product import _ProductTangentVector    
from pymanopt.manifolds.manifold import Manifold
class TangentBundle(Euclidean):
    def __init__(self, mfd: Manifold):
        p = mfd.random_point()
        g = mfd.random_tangent_vector(p)
        shape = list(g.shape)
        
        E = Euclidean(*shape)
        shape_1 = (2,)+g.shape
        super().__init__(*shape_1)
        self.base_mfd = mfd
        self.bundle = E

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        p,v = point
        tp_a,tv_a = tangent_vector_a
        tp_b,tv_b = tangent_vector_b
        
        return self.base_mfd.inner_product(p,tp_a,tp_b) + self.base_mfd.inner_product(p,tv_a,tv_b)
    
    def exp(self, point, tangent_vector):
        p,v = point
        tp,tv = tangent_vector
        exp_p = self.base_mfd.exp(p,tp)
        exp_v = self.base_mfd.transp(p,v+tv,exp_p)
        return np.array([exp_p,exp_v])
    
    def log(self, point_a, point_b):
        pa,va = point_a
        pb,vb = point_b
        log_p = self.base_mfd.log(pa,pb)
        log_v = self.base_mfd.transp(pb,vb,pa) - va
        return np.array([log_p,log_v])
    
    def dist(self, point_a, point_b):
        return np.sqrt(self.inner_product(point_a,self.log(point_a,point_b),self.log(point_a,point_b)))
    
    def transp(self, point_a, tangent_vector, point_b):
        pa,va = point_a
        pb,vb = point_b
        tp,tv = tangent_vector
        tran_tp = self.base_mfd.transp(pa,tp,pb)
        tran_tv = self.base_mfd.transp(pa,tv,pb)
        return np.array([tran_tp,tran_tv])
    
    def random_point(self):
        p = self.base_mfd.random_point()
        v = self.base_mfd.random_tangent_vector(p)
        return [p,v]
    
    def random_tangent_vector(self, point):
        p,v  = point
        tp = self.base_mfd.random_tangent_vector(p)
        tv = self.base_mfd.random_tangent_vector(p)
        return np.array([tp,tv])
    
    def norm(self, point, tangent_vector):
        return np.sqrt(self.inner_product(point,tangent_vector,tangent_vector))
    
    def zero_vector(self, point):
        p,v = point
        zp = self.base_mfd.zero_vector(p)
        zv = self.bundle.zero_vector(v)
        return np.array([zp,zv])