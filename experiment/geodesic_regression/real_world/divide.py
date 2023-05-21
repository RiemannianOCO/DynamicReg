import os
import sys

import numpy as np

sys.path.append(
    os.getcwd()
)
from core.online_problem import OnlineProblem
from experiment.geodesic_regression.real_world import config

n=config.n
T=400
mfd =config.mfd

fold_read = config.foldname
print(fold_read)
#fold_write = config.fold_strong
np.random.seed(42)
X = np.load(fold_read + 'age.npy')
Y = np.load(fold_read + 'data.npy')
print(Y[0])
'''
permutation = np.random.permutation(T)

X = X[permutation]
Y = Y[permutation , :]

X_train, Y_train = X[:320], Y[:320,:]
X_test, Y_test = X[320:], Y[320:,:]
'''
#np.save('''/home/appendix/code_rzs/online_learning/experiments/experiment/geodesic_regression/real_world/data/age_train''',
#        X_train) 

#np.save('''/home/appendix/code_rzs/online_learning/experiments/experiment/geodesic_regression/real_world/data/data_train''',
#        Y_train) 


#np.save('''/home/appendix/code_rzs/online_learning/experiments/experiment/geodesic_regression/real_world/data/age_test''',
#        X_test)

#np.save('''/home/appendix/code_rzs/online_learning/experiments/experiment/geodesic_regression/real_world/data/data_test''',
#        Y_test)  

