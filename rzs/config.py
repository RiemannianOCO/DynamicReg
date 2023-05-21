D=10
Figsize=(8,10)
Linewidth=5.0
LabelFontdict=14
AxisFontdict=14

from rzs.solver import *
kw_dict ={}
kw_dict[RiemannianOptimisticGDA] = {'c': '#1772b2', 'ls' : '--' , 'linewidth': 3,'label':'ROGDA-last'}
kw_dict[RiemannianCEG] = {'c': '#249c24', 'ls': '--', 'linewidth': 3,'label':'RCEG'}
kw_dict[RiemannianHM] = {'c': '#ff7f0e', 'ls': '--', 'linewidth': 3,'label':'RHM'}
kw_dict[RiemannianOptimisticGDAvg] = {'c':'#d62425','ls': '--', 'label':'ROGDA-avg','linewidth':3}
kw_dict[RiemannianGDA] =  {'c':'#6a5acd','ls': '--', 'label':'RGDA','linewidth':3}


