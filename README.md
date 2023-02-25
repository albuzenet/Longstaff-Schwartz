# Longstaff-Schwartz Americain option pricing

American option pricing using Longstaff Schwartz algorithm under the Heston model. <br>

The option file contains class representing derivative options <br>
The process file contains class representing stocastic models <br>
The pricing file contains functions for pricing derivatives using different techniques

## Classic Monte Carlo

First let's use the GeometricBrownianMotion class to simulation some GBM paths. This class use the available closed form formula of the GBM stocastic process
$$ 
\begin{cases}
dS_t = \mu S_t dt + \sigma S_t dW_t \\
\end{cases}
$$
Using the solution of this SDE, we can use numpy vectorization to simulate the paths in an efficient way. This allow us to simulate a good number of paths quite rapidly


```python
gmb = GeometricBrownianMotion(mu=0.05, sigma=0.2)
gmb
```




    GeometricBrownianMotion(mu=0.05, sigma=0.2)




```python
%time test = gmb.simulate(s0=60, T=1, n=200_000, m=252)
```

    CPU times: total: 2.91 s
    Wall time: 2.99 s
    

Only 5 sec to generate 200k paths

Now we can display the results of the simulation. Let's plot 10 simulated GBM paths


```python
S = gmb.simulate(s0=60, T=1, n=10, m=252)  # n = number of path, m = number of discretization points

plt.figure(figsize=(8, 6))
plt.plot(S)
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$S_t$', fontsize=20, labelpad=20).set_rotation(0)
plt.title('Simulation of 10 GBM paths', fontsize=15)
plt.xlim(-10, 280)
plt.show()
```


    
![png](demo_files/demo_8_0.png)
    


Now we use the Option class to represente the parameters of an option derivative 


```python
call = Option(s0=60, T=1, K=80, call=True)
call
```




    Option(s0=60, T=1, K=80, v0=None, call=True)



With both an option and a process representing the dynamic of the underlying, we can use the classic Monte Carlo method to approximate the price of the option. For this, we use the monte_carlo_simulation function from the custom princing module.


```python
%time price = monte_carlo_simulation(option=call, process=gmb, n=200_000, m=252)

put = Option(s0=60, T=1, K=80, call=False)
%time price = monte_carlo_simulation(option=put, process=gmb, n=200_000, m=252)
```

    The price of Option(s0=60, T=1, K=80, v0=None, call=True) = 0.77
    95.0% confidence interval = [0.76, 0.79]
    CPU times: total: 2.8 s
    Wall time: 2.92 s
    The price of Option(s0=60, T=1, K=80, v0=None, call=False) = 16.86
    95.0% confidence interval = [16.82, 16.91]
    CPU times: total: 2.91 s
    Wall time: 3.08 s
    

Now let's compare these results with the closed form formula of a vanilla call and put options when the underlying follows a GMB process. For this we use the black_scholes_merton function


```python
call_price = black_scholes_merton(r=0.05, sigma=0.2, option=call)
put_price = black_scholes_merton(r=0.05, sigma=0.2, option=put)

print(f'Theorical price of the call option = {call_price}')
print(f'Theorical price of the put option = {put_price}')
```

    Theorical price of the call option = 0.77
    Theorical price of the put option = 16.87
    

Very close !

## Longstaff-Schwartz applied to the Heston model 

We will now try to estimate the price of an american option in the Heston framework:<br><br>

$$ 
\begin{cases}
dS_t = r S_t dt + \sqrt{v_t} S_t dW^1_t \\
dv_t = \kappa (\theta - v_t) dt + \eta \sqrt{v_t} dW^2_t \\
dW^1_t dW^2_t = \rho dt
\end{cases}
$$

<br>
Like for the GBM process, we use the HestonProcess class to simulte an Heston process numerically. The paths are simulated using the Milstein schema of discretization. As opposite to the first process, it is not possible to fully vectorize the calculation of the paths. Thus, the simulation takes much longer than for the GeometricBrownianMotion class.


```python
heston = HestonProcess(mu=0.06, kappa=0.0005, theta=0.04, eta=0.1, rho=-0.5)
heston
```




    HestonProcess(mu=0.06, kappa=0.0005, theta=0.04, eta=0.1, rho=-0.5)




```python
%time s = heston.simulate(s0=60, v0=0.05, T=1, n=5_000, m=252)
```

    CPU times: total: 4.55 s
    Wall time: 4.59 s
    

More than 10 sec for 5k paths vs 5 sec for 200k with the GMB process

Then let's create two american put options


```python
put_1 = Option(s0=36, v0=0.05, T=1, K=40, call=False)
put_2 = Option(s0=60, v0=0.05, T=5/12, K=50, call=False)

put_1
```




    Option(s0=36, T=1, K=40, v0=0.05, call=False)



Now we use the Longstaff-Schwartz algorithm to evaluate the price of the americain put options.<br><br>
The algorithm estimate the price of an americain option in a backward induction manner, just like in a CRR tree. We take discretization nodes = possible dates of exercise for the sake of simplicity. At each node of discretization, we fit a polynome (using the Polynome class from numpy) between the silumated paths and the discounted continuation value at the next step in time. Then the current continuation value is estimated using the parametres of the fitted polynome. <br><br>
For this implementation we use all the paths even though Longstaff and Schwartz pointed that using only the in-the-money paths is probably more efficient. The algorithm is implemented in the monte_carlo_simulation_LS function


```python
%time monte_carlo_simulation_LS(option=put_1, process=heston, n=25_000, m=252)
%time monte_carlo_simulation_LS(option=put_2, process=heston, n=25_000, m=252)
```

    The price of Option(s0=36, T=1, K=40, v0=0.05, call=False) = 4.5816
    CPU times: total: 21.8 s
    Wall time: 21.9 s
    The price of Option(s0=60, T=0.4166666666666667, K=50, v0=0.05, call=False) = 0.2832
    CPU times: total: 23.1 s
    Wall time: 23.1 s
    

We can control the implementation of the monte_carlo_simulation_LS function by using an GBM process and comparing the results with a simple CRR tree. Like the Longstaff and Schwartz this algorithm this work in a backward manner. The CRR tree pricing method is implemented in the crr_pricing function.


```python
put_3 = Option(s0=36, T=1, K=40, call=False)
gmb = GeometricBrownianMotion(mu=.06, sigma=.2)

%time monte_carlo_simulation_LS(option=put_3, process=gmb, n=25_000, m=252)
```

    The price of Option(s0=36, T=1, K=40, v0=None, call=False) = 4.4415
    CPU times: total: 1.36 s
    Wall time: 1.41 s
    


```python
crr = crr_pricing(r=.06, sigma=.2, option=put_3, n=25000)
print(f'The price of the put_3 option estimated via a crr tree = {crr}')
```

    The price of the put_3 option estimated via a crr tree = 4.487
    


```python

```
