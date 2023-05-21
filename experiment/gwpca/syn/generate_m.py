import numpy as np
from scipy.linalg import qr

mu = 0.2
L = 4.5
d = 10
n = 20

m = np.zeros((n,d,d))
m_inv_sq = np.zeros((n,d,d))
for i in range(n):
    temp = np.random.rand(d,d)
    (Q,R) = qr(temp)
    sigma = (L - mu) * np.random.rand(d) + mu
    m[i] = Q @ np.diag(sigma) @ Q.T
    m_inv_sq[i] = Q @ np.diag(1/ np.sqrt(sigma)) @ Q.T
np.save('.\experiment\gwpca\m.npy',m)
