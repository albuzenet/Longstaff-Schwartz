import pandas as pd 
import numpy as np
from numpy.polynomial import Polynomial
from dataclasses import dataclass
from abc import ABC, abstractmethod


class StocasticProcess(ABC):
    """Represente a stocastic process (just for typing)"""
    @abstractmethod
    def simulate(self):
        ...

@dataclass
class GeometricBrownianMotion(StocasticProcess):
    """
    A classic geometric brownian motion which can be simulated.
    The closed form formula allow a fully vectorized calculation of the paths.
    """
    mu: float
    sigma: float
    s0: float

    def simulate(self, T: int, n: int, m: int) -> pd.DataFrame:                # n = number of path, m = number of discretization points

        dt = T/m
        np.random.seed(0)
        W = np.cumsum(np.sqrt(dt) * np.random.randn(m + 1, n), axis = 0)
        W[0] = 0

        T = np.ones(n).reshape(1, -1) * np.linspace(0, T, m + 1).reshape(-1, 1)
        
        s = self.s0 * np.exp((self.mu - 0.5 * self.sigma**2) * T + self.sigma * W)

        return s

@dataclass
class HestonProcess(StocasticProcess):
    """
    An Heston process which can be simulated using Milstein schema.
    """
    mu: float
    kappa: float
    theta: float
    eta: float
    rho: float
    s0: float
    v0: float

    def simulate(self, T: int, n: int, m: int) -> pd.DataFrame:   # n = number of path, m = number of discretization points

        dt = T/m
        z1 = np.random.randn(m, n)
        z2 = self.rho * z1 + np.sqrt(1-self.rho**2) * np.random.randn(m, n)

        s = np.zeros((m + 1, n))
        x = np.zeros((m + 1, n))
        v = np.zeros((m + 1, n))

        s[0] = self.s0
        v[0] = self.v0

        for i in range(m):

            v[i + 1] = v[i] + self.kappa * (self.theta - v[i]) * dt + self.eta * np.sqrt(v[i] * dt) * z1[i] + self.eta**2 / 4 * (z1[i]**2 - 1) * dt
            v = np.where(v > 0, v, -v)

            x[i + 1] = x[i] + (self.mu - v[i] / 2) * dt + np.sqrt(v[i] * dt) * z2[i]

            s[1:] = s[0] * np.exp(x[1:])
        
        return s