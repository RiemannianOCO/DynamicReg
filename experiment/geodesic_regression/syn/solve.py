import os
import sys

import numpy as np
import warnings
warnings.filterwarnings('error',category=RuntimeWarning)

sys.path.append(os.getcwd())
from roco.core.online_problem import OnlineProblem
import config
from roco.lib.function import geo_reg
from roco.solver import *
import matplotlib.pyplot as plt

n=config.n
T=config.T
mfd =config.mfd
fold_read = config.foldname
#fold_write = config.fold_strong

X = np.load(fold_read + 'data_X.npy')
Y = np.load(fold_read + 'data_Y.npy')


A = []
for i in range(T):
    A.append([X[i],Y[i]])
X_0 = np.array( [ Y[0], mfd.base_mfd.random_tangent_vector(Y[0])] )
loss= lambda data,X: geo_reg.func(data,mfd.base_mfd,X)
grad= lambda data,X: geo_reg.grad(data,mfd.base_mfd,X)
ol_georeg_prob = OnlineProblem(    mfd = mfd,
                                data = A,
                                time = T,
                                param = config.param,
                                loss = loss,
                                grad = grad
                                ) 
zeta = config.param.zeta
sigma = config.param.sigma
D = config.param.D
G = config.param.G
L = config.param.L
eta_aoogd_min = (sigma*D**2/(16*zeta**2*G**2*T)) **(0.5)
eta_aoogd_max = sigma/(4*zeta*L)
N_aoogd = int(np.log2(eta_aoogd_max/eta_aoogd_min)+1) + 1

eta_radrv_min = (D**2/(8*zeta*G**2*T)) **(0.5)
eta_radrv_max = (sigma)**0.5/(((2*zeta)**0.5)*L)
N_radrv = int(np.log2(eta_aoogd_max/eta_aoogd_min)+1) + 1
beta = (1/ (24 **0.5)) /(config.param.D **2 *config.param.zeta)
print(N_aoogd,N_radrv,eta_aoogd_min,eta_radrv_min)

solvers = [ROOGD(),RAOOGD(),OnlineRCEG(),RADRv(),OnlineGradientDescent()]
parameter = {}
parameter['roogd'] = {"eta":0.25}
parameter['raoogd'] = {"beta":beta,"eta":eta_aoogd_min,"N":N_aoogd}
parameter['roceg'] ={"eta":0.25}
parameter['radrv'] = {"beta":beta,"eta":eta_radrv_min,"N":N_radrv}
parameter['rogd'] = {}


for solver in solvers:
    str_solver = str(solver)
    solver.optimize(ol_georeg_prob,X_0, **(parameter[str_solver]) )
    solver.calculate_aver_value()
    plt.semilogy( solver.aver_value_histories)
    np.save(fold_read + 'sum_'+str_solver+'.npy',solver.sum_array(solver.value_histories))
    np.save(fold_read + 'aver'+str_solver+'.npy', solver.aver_value_histories)

plt.show()
