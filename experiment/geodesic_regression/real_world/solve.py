import os
import sys

import numpy as np
import warnings
warnings.filterwarnings('error',category=RuntimeWarning)

sys.path.append(
    os.getcwd()
)
from roco.core.online_problem import OnlineProblem
import config
from roco.lib.function import geo_reg
from roco.solver import *
import matplotlib.pyplot as plt
n=config.n
T=config.T
mfd =config.mfd

fold_read = config.foldname
print(fold_read)
np.random.seed(42)
X = np.load(fold_read + 'age_train.npy')
Y = np.load(fold_read + 'data_train.npy')

A = []
for i in range(T):
    for j in range(5):
        A.append( [X[i],Y[i]] )


base_start=mfd.base_mfd.random_point()
X_0 = np.array( [base_start, mfd.base_mfd.random_tangent_vector(base_start)] )
np.random.seed()
loss= lambda data,X: geo_reg.func(data,mfd.base_mfd,X)
grad= lambda data,X: geo_reg.grad(data,mfd.base_mfd,X)

ol_georeg_prob = OnlineProblem(    mfd = mfd,
                                data = A,
                                time = 320*5,
                                param = config.param,
                                loss = loss,
                                grad = grad
                                ) 


solvers = [ROOGD(),RAOOGD(),OnlineRCEG(),RADRv(),OnlineGradientDescent()]
#solvers = [ROOGD(),OnlineRCEG()]
parameter = {}
parameter['roogd'] = {"eta":0.2}
parameter['raoogd'] = {"beta":1e-1,"eta":0.1,"N":6}
parameter['roceg'] ={"eta":0.2}
parameter['radrv'] = {"beta":1e-1,"eta":0.1,"N":6}
parameter['rogd'] = {}


for solver in solvers:
    str_solver = str(solver)
    solver.optimize(ol_georeg_prob,X_0, **(parameter[str_solver]) )
    solver.calculate_aver_value()
    plt.semilogy( solver.aver_value_histories)
    np.save(fold_read + 'sum_'+str_solver+'.npy',solver.sum_array(solver.value_histories))
    np.save(fold_read + 'aver'+str_solver+'.npy', solver.aver_value_histories)

plt.show()

'''
idx = 25
solver = OnlineOpitimisticGradientDescent()
p[0],v[0] = solver.optimize(ol_fre_prob,X_0,eta= 0.2)
solver.calculate_aver_value()
plt.semilogy(solver.aver_value_histories)
np.save(fold_read + 'sum_oogd.npy',solver.sum_array(solver.value_histories))
np.save(fold_read + 'aver_oogd.npy', solver.aver_value_histories)

solver =  OnlineOpitimisticGradientDescent()
p[1] , v[1] =solver.optimize_meta(ol_fre_prob,X_0,beta=1e-1,eta=0.2,N=6)
solver.calculate_aver_value()
plt.semilogy( solver.aver_value_histories)
np.save(fold_read + 'sum_aoogd.npy',solver.sum_array(solver.value_histories))
np.save(fold_read + 'aver_aoogd.npy', solver.aver_value_histories)

solver = OnlineRCEG()
p[2] , v[2] =solver.optimize(ol_fre_prob,X_0,eta=0.2)
solver.calculate_aver_value()
plt.semilogy(  solver.aver_value_histories)
np.save(fold_read + 'sum_rceg.npy',solver.sum_array(solver.value_histories))
np.save(fold_read + 'aver_rceg.npy', solver.aver_value_histories)

solver = OnlineRCEG()
p[3] , v[3] =solver.optimize_meta(ol_fre_prob,X_0,beta=1e-1,eta=0.2,N=6)
solver.calculate_aver_value()
plt.semilogy( solver.aver_value_histories)
np.save(fold_read + 'sum_arceg.npy',solver.sum_array(solver.value_histories))
np.save(fold_read + 'aver_arceg.npy', solver.aver_value_histories)

solver = OnlineGradientDescent()
p[4] , v[4] =solver.optimize(ol_fre_prob,X_0,mu=0)
solver.calculate_aver_value()
plt.semilogy(solver.aver_value_histories)
np.save(fold_read + 'sum_ogd.npy',solver.sum_array(solver.value_histories))
np.save(fold_read + 'aver_ogd.npy', solver.aver_value_histories)

plt.show()
#solver.sum_time()
print(p[1][:10],v[1][:10])
for i in range(5):
    Shape[i] = mfd.base_mfd.exp(p[i],X[idx]*v[i])
    plt.scatter(Shape[i,:,1], Shape[i,:,0])
plt.scatter(Y[idx,:,1],Y[idx,:,0])
plt.show()
np.save( fold_read + 'p.npy',p)
np.save( fold_read + 'v.npy',v)
print('gradient solver completed')
'''