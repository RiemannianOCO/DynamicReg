import sys
import os
sys.path.append(os.getcwd())
import config
import numpy as np


n = config.n
T = config.T
block = config.block
Hn = config.mfd
center = Hn.center
np.random.seed(12)
A = np.zeros((T,block,n))
S = Hn.randn()
for i in range(T):
    for j in range(block):
        A[i,j] = Hn.exp(S,  1* np.random.rand()*Hn.random_tangent_vector(S) )
        if i % 100 == 0:
            A[i,j] = Hn.randn()
            S = Hn.randn()
        # S = Hn.exp(S,  0.01* np.random.rand()*Hn.random_tangent_vector(S) )

filename = config.foldname + 'data_A.npy'
print(filename)
np.save( filename , A )