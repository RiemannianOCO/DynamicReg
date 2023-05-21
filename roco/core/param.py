import sys
import os
import numpy as np
from roco.lib.function.std_func import sigma, zeta

class Param:
    def __init__(self, diameter = None, curvature_above  = None, curvature_below  = None, lipschitz  = None,  smooth = None) -> None:

        self.D = diameter
        self.G = lipschitz
        self.L = smooth
        self.K = curvature_above
        self.kappa = curvature_below

    @property
    def zeta(self):
        return zeta(self.kappa,self.D)
    
    @property
    def sigma(self):
        return sigma(self.kappa,self.D)