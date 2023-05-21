import os
import sys
sys.path.append(os.getcwd())
from rzs.config import Figsize,Linewidth,LabelFontdict,AxisFontdict,kw_dict
from rzs.problem.logdet import Logdet
from rzs.parameter import Parameter
from rzs.solver import *
import matplotlib.pyplot as plt
import numpy as np

d = 30
p = Parameter(eta = 0.2,max_iter=100)
p_gda = Parameter(eta = 0.5,max_iter=100)
prob = Logdet(d,1,1)
x_0 = prob.M.random_point()
y_0 = prob.N.random_point()
alg_list = [RiemannianOptimisticGDA,RiemannianOptimisticGDAvg,RiemannianCEG,RiemannianHM,RiemannianGDA]
p_dict ={}
p_dict[RiemannianOptimisticGDA] = p
p_dict[RiemannianCEG] = p
p_dict[RiemannianHM] = p
p_dict[RiemannianOptimisticGDAvg] = p
p_dict[RiemannianGDA] = p_gda

for alg in alg_list:
    m = alg(prob,p_dict[alg])
    m.optimize(x_0 = x_0, y_0 = y_0)
    plt.figure(1,figsize=Figsize)
    plt.semilogy(m.gnorm[1:],**kw_dict[alg])
    plt.legend(loc=1,prop={'size':LabelFontdict})
    plt.ylim(1e-10,100)
    plt.xlabel('Learning Rounds',fontdict={'size':AxisFontdict})
    plt.ylabel('Gradnorm',fontdict={'size':AxisFontdict})
    plt.xticks(size=AxisFontdict)
    plt.yticks(size=AxisFontdict)
    plt.gcf().set_facecolor(np.ones(3))
    plt.grid(True)
    plt.subplots_adjust(left=0.15)
plt.legend()
plt.show()