import os
import sys
sys.path.append(os.getcwd())
sys.path.append("/home/appendix/code_rzs/zero_sum_game/")
from rzs.config import Figsize,Linewidth,LabelFontdict,AxisFontdict,kw_dict
from rzs.parameter import Parameter
from rzs.problem.gwpca import GeometryPCA
from rzs.solver import *
import matplotlib.pyplot as plt
import numpy as np

m = np.load("/home/appendix/code_rzs/zero_sum_game/experiment/gwpca/bci.npy")
m /= 200
prob = GeometryPCA(d = m.shape[1],alpha=1 ,m = m)
x_0 = prob.M.random_point()
y_0 = prob.N.random_point()

p_ogda = Parameter(eta = 0.05,max_iter=500,nu=1)
p_hm = Parameter(eta = 0.05,max_iter=500,nu=1)
p_gda = Parameter(eta = 0.075,max_iter=500,nu=1)
p_ceg = Parameter(eta = 0.02,max_iter=500,nu=1)
alg_list = [RiemannianOptimisticGDA,RiemannianHM,RiemannianCEG,RiemannianGDA]
p_dict ={}
p_dict[RiemannianOptimisticGDA] = p_ogda
p_dict[RiemannianCEG] = p_ceg
p_dict[RiemannianHM] = p_hm
p_dict[RiemannianOptimisticGDAvg] = p_ogda
p_dict[RiemannianGDA] = p_gda

for alg in alg_list:
    m = alg(prob,p_dict[alg])
    m.optimize(x_0 = x_0, y_0 = y_0)
    plt.figure(1,figsize=Figsize)
    plt.semilogy(m.gnorm[1:],**kw_dict[alg])
    plt.legend(loc=1,prop={'size':LabelFontdict})
    plt.ylim(1e-10,10)
    plt.xlabel('Learning Rounds',fontdict={'size':AxisFontdict})
    plt.ylabel('Gradnorm',fontdict={'size':AxisFontdict})
    plt.xticks(size=AxisFontdict)
    plt.yticks(size=AxisFontdict)
    plt.gcf().set_facecolor(np.ones(3))
    plt.grid(True)
    plt.subplots_adjust(left=0.15)
plt.legend()
plt.show()