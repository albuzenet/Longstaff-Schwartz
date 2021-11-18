import pandas as pd 
import numpy as np
from numpy.polynomial import Polynomial
from scipy.stats import norm

from option import Option
from process import StocasticProcess


def monte_carlo_simulation(option: Option, process: StocasticProcess, n: int, m: int, alpha: float=0.05) -> float:            
    """
    Given an option and a process followed by the underlying, calculate the classic monte carlo price estimator
    """
    # n = number of path, m = number of discretization points
    s = process.simulate(T=option.T, n=n, m=m)
    st = s[-1]
    payoffs = option.payoff(s=st) 
    
    discount = np.exp(-process.mu * option.T)
    price = np.mean(payoffs) * discount

    quantile = norm.ppf(1 - alpha/2)
    confidence_interval = [
        np.round(price - quantile * np.std(payoffs * discount) / np.sqrt(n), 2),
        np.round(price + quantile * np.std(payoffs * discount) / np.sqrt(n), 2)
        ]

    print(f"The price of {option!r} = {price:.2f}")
    print(f'{(1-alpha)*100}% confidence interval = {confidence_interval}')

    return np.round(price, 2)


def monte_carlo_simulation_LS(option: Option, process: StocasticProcess, n: int, m: int, alpha: float=0.05) -> float:    
    """
    Given an option and a process followed by the underlying, calculate the option value using the Longstaff-Schwartz algorithme 
    """
    # n = number of path, m = number of discretization points
    np.random.seed(0)
    s = process.simulate(T=option.T, n=n, m=m)

    payoffs = option.payoff(s=s)

    v = np.zeros_like(payoffs)
    v[-1] = payoffs[-1]

    dt = option.T / m
    discount = np.exp(-process.mu * dt)

    for t in range(m - 1, 0, -1):
        polynome = Polynomial.fit(s[t], discount * v[t + 1], 5)
        c = polynome(s[t])
        v[t] = np.where(payoffs[t] > c, payoffs[t], discount * v[t + 1])

    price = discount * np.mean(v[1])

    print(f"The price of {option!r} = {round(price, 4)}")

def black_scholes_merton(s0, K, T, r, sigma, call=True):
    """
    Calculate the price of vanilla options using BSM formula 
    """
    d1 = (np.log(s0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = s0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    price = price if call else price - s0 + K * np.exp(-r * T)

    return np.round(price, 2)


def crr_pricing(s0=100, K=100, T=1, r=.1, sigma=.2, n = 25000):
    """
    Calculate the price of an americain option using a backward tree model
    """
    dt = T / n                            
    u = np.exp(sigma * np.sqrt(dt))                
    d = 1 / u                                   
    a = np.exp(r * dt)    
    p = (a - d)/ (u - d)  
    q = 1 - p         

    st = np.array( [s0 * u**i * d**(n - i) for i in range(n + 1)] ) 
    v = np.maximum(K - st, 0)

    for _ in range(n):
        v[:-1] = np.exp(-r * dt) * (p * v[1:] + q * v[:-1])   
        st = st * u                   
        v = np.maximum(v, K - st)
        
    return np.round(v[0], 3)