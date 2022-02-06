
from abc import abstractclassmethod
from abc import ABC, abstractmethod
import numpy as np
from chaosformer.data.orbits import Orbit, CodifiableOrbit


class Map(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def iterate(x0, n):
        pass


class GenMap(Map):
    
    def __init__(self, f):
        super().__init__()
        self._f = f
    
    @property
    def f(self):
        return self._f
    
    def iterate(self, x0, n):
        generated_orbit = CodifiableOrbit(x0=x0)

        for _ in range(n):
            generated_orbit.evolve(f=self._f)
        
        return generated_orbit


def logistic(r, x):
    return r * x * (1 - x)


class LogisticMap(GenMap):

    def __init__(self, r):
        super().__init__(f=lambda x: logistic(r=r, x=x))


def tent(mu, x):
    if x < 0.5:
        f = mu * x
    elif x > 0.5:
        f = mu - mu * x
    return f


class TentMap(GenMap):

    def __init__(self, mu):
        super().__init__(f=lambda x: tent(mu=mu, x=x))


def henon(a, b, x0):
    x = x0[0]
    y = x0[1]

    new_x = 1 - a*x**2 + y
    new_y = b * x

    return np.array((new_x, new_y))


class HenonMap(GenMap):

    def __init__(self, a, b):
        super().__init__(f=lambda x: henon(a=a, b=b, x0=x))

