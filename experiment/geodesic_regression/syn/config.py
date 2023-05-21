


import os
import sys

import numpy as np

sys.path.append(os.getcwd())
from roco.core.param import Param

# dim,rounds and blocks
n=5
T=10000
#block=10

# manifold 
from manifold.tangentbundle import TangentBundle
from manifold.Sn_transp import Sn_transport

S = Sn_transport(n)
mfd = TangentBundle(S)

# parameter
diameter = np.pi/ 4
curvature_below = 0
curvature_above = 1
lipschitz = diameter
bound =10
param = Param(diameter = diameter,
              curvature_above=curvature_above,
              curvature_below=curvature_below,
              lipschitz=lipschitz,
              smooth = 1
              )

# initial point
np.random.seed(42)
X_0 = mfd.random_point()
np.random.seed()

# save file
foldname = os.path.dirname(__file__)+ '/data/'