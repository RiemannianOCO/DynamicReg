import numpy as np
def zeta(kappa,D):
    if kappa>=0:
        return 1
    else:
        return np.sqrt(kappa)* D / np.tanh( np.sqrt(kappa)* D )

def sigma(K,D):
    if K<=0:
        return 1
    else:
        return np.sqrt(K)* D / np.tan( np.sqrt(K)* D )

class Parameter:
    def __init__(self,max_iter = 100,eta=None,nu=1):
        '''
        assert D>0
        assert G>0
        assert mu>0
        assert L>0
        self.D = D
        self.G = G
        self.mu = mu
        self.L = L

        assert K>kappa
        self.K = K
        self.kappa = kappa
        self.zeta_0 =zeta(kappa,D)
        self.sigma_0 = sigma(K,D)
        '''
        
        self.max_iter = max_iter

        assert eta>0
        self.eta = eta

        self.nu = nu


