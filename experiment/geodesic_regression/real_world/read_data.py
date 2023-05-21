import numpy as np
from scipy import io
from config import foldname
age = []
for str in ['ADLdataF','ADLdataM','NCLdataF','NCLdataM']:
    mat = io.loadmat(foldname+str+'.mat')
    mat_t = np.transpose(mat[str])
    age.append(mat_t)
 
raw = np.concatenate(age)
n = raw.shape[0]
data = np.zeros((n,50,2))
for i in range(n):
    raw_t = np.transpose(raw[i])
    U,s,v = np.linalg.svd(raw_t,full_matrices=False)
    data[i] = U
np.save(foldname+'data',data) 