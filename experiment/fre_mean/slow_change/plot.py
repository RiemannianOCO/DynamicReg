import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import config

D=1
Figsize=(8,10)
Linewidth=5.0
LabelFontdict=14
AxisFontdict=14


str = ['raoogd','roceg','radrv','rogd','roogd']
kw_dict ={}
kw_dict['roogd'] = {'c': '#1772b2', 'ls' : '--' , 'linewidth': 3,'label':'R-OOGD'}
kw_dict['raoogd'] = {'c': '#249c24', 'ls': '--', 'linewidth': 3,'label':'R-AOOGD'}
kw_dict['roceg'] = {'c': '#ff7f0e', 'ls': '-', 'linewidth': 3,'label':'R-OCEG'}
kw_dict['radrv'] = {'c':'#d62425','ls': '--', 'label':'RADRv','linewidth':3}
kw_dict['rogd'] =  {'c':'#6a5acd','ls': '--', 'label':'R-OGD','linewidth':3}



foldname = config.foldname
for alg in str:
    sum = np.load(foldname+'sum_'+alg+'.npy')
    plt.figure(1,figsize=Figsize)
    plt.plot(sum,**kw_dict[alg])
    plt.legend(loc=1,prop={'size':LabelFontdict})
    plt.ylim(0,10000)
    plt.xlabel('Learning Rounds',fontdict={'size':AxisFontdict})
    plt.ylabel('Accu. Loss',fontdict={'size':AxisFontdict})
    plt.xticks(size=AxisFontdict)
    plt.yticks(size=AxisFontdict)
    plt.gcf().set_facecolor(np.ones(3))
    plt.grid(True)
    plt.subplots_adjust(left=0.15)
plt.legend()
plt.show()