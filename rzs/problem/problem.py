from pymanopt.manifolds import manifold
class Problem:
    def __init__(self,M:manifold,N:manifold) -> None:
        self.M = M
        self.N = N
    def f(self):
        pass

    def g(self):
        pass
    
    def h(self):
        pass