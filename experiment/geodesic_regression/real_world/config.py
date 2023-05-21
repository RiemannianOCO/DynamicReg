import os
import sys

import numpy as np
sys.path.append("/home/appendix/code_rzs/online_learning/experiments")
sys.path.append(os.getcwd())
from roco.core.param import Param

# dim,rounds and blocks
n = 50
p = 2
T= 320
#block=10

# manifold 
from manifold.tangentbundle import TangentBundle
from manifold.Gn_transp import Gn_transport

G = Gn_transport(n,p)
mfd = TangentBundle(G)

# parameter
diameter = 2**0.5 * np.pi
curvature_below = 0
curvature_above = 2
lipschitz = 1
param = Param(diameter = diameter,
              curvature_above=curvature_above,
              curvature_below=curvature_below,
              lipschitz=lipschitz,
              )

# initial point
np.random.seed(42)
X_0 = mfd.random_point()
np.random.seed()

# save file
foldname = os.path.dirname(__file__)+ '/data/'