import numpy as np



 
def func(A,mfd,x): 
    distance = 0
    N = A.shape[0]
    for i in range(N):
        distance += 1/N * mfd.dist(x,A[i]) ** 2
    return distance
    

def grad(A,mfd,x):
    N = A.shape[0] 
    grad = mfd.zero_vector(x)
    for i in range(N):
        grad -=  1/N *  mfd.log(x,A[i])
    return grad
