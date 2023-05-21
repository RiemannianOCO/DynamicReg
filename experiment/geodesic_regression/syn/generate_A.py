import sys
import os
sys.path.append(os.getcwd())
import config
import numpy as np


n = config.n
T = config.T
S = config.S

np.random.seed(42)
X = np.zeros(T)
Y = np.zeros((T,n))
p = S.random_point()
if p[0]<0:
    p[0] = -p[0]
for i in range(T):
    if i % 1000 == 0:     
        v = S.random_tangent_vector(p)
    X[i] = (np.random.rand()-0.5)*2
    y_hat = S.exp(p,X[i]*v)
    e = S.random_tangent_vector(y_hat)
    e = 0.1*np.random.randn()*e /(S.norm(y_hat,e))
    Y[i] = S.exp(y_hat,e)

    

filename = config.foldname + 'data_X.npy'
print(filename)
np.save( filename , X )

filename = config.foldname + 'data_Y.npy'
print(filename)
np.save( filename , Y )