import numpy as np
from pymanopt.manifolds.product import _ProductTangentVector
from pymanopt.manifolds import Sphere,Product 
def func(data,mfd,X):
    p,v = X
    x,y =data
    return 0.5 * (mfd.dist(mfd.exp( p, x* v ),y)) ** 2

def grad(data,mfd,X):
    p,v = X
    x,y = data
    shape = v.shape
    y_hat = mfd.exp( p, x* v)
    e = mfd.log(y_hat,y)
    g_p = mfd.transp(y_hat, e , p)
    g_v = x * g_p
    return _ProductTangentVector([-g_p,-g_v])