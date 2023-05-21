import numpy as np
from scipy import io
age = []
for str in ['ADinfoF','ADinfoM','NCinfoF','NCinfoM']:
    mat = io.loadmat('/home/appendix/code_rzs/online_learning/experiments/experiment/geodesic_regression/real_world/data/'+str+'.mat')
    mat_t = np.transpose(mat[str])
    age.append(mat_t[8])

raw = np.concatenate(age).astype(float)
data = (raw - 75)/(20)
# 再将其存为npy格式文件
np.save('/home/appendix/code_rzs/online_learning/experiments/experiment/geodesic_regression/real_world/data/age',data) 