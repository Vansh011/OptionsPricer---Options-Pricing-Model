    # Underlying price (per share): S; 
    # Strike price of the option (per share): K;
    # Time to maturity (years): T;
    # Continuously compounding risk-free interest rate: r;
    # Volatility: sigma;
    # Number of binomial steps: N;

        # The factor by which the price rises (assuming it rises) = u ;
        # The factor by which the price falls (assuming it falls) = d ;
        # The probability of a price rise = pu ;
        # The probability of a price fall = pd ;
        # discount rate = disc ;

import math
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define Cox-Ross-Rubinstein binomial model
def Cox_Ross_Rubinstein_Tree(S, K, T, r, sigma, N, Option_type):
    u = math.exp(sigma * math.sqrt(T / N))
    d = 1 / u
    pu = (math.exp(r * T / N) - d) / (u - d)
    pd = 1 - pu
    disc = math.exp(-r * T / N)

    St = [0] * (N + 1)
    C = [0] * (N + 1)

    St[0] = S * d**N

    for j in range(1, N + 1):
        St[j] = St[j - 1] * u / d

    for j in range(1, N + 1):
        if Option_type == 'P':
            C[j] = max(K - St[j], 0)
        elif Option_type == 'C':
            C[j] = max(St[j] - K, 0)

    for i in range(N, 0, -1):
        for j in range(0, i):
            C[j] = disc * (pu * C[j + 1] + pd * C[j])

    return C[0]

# Define Jarrow-Rudd binomial model
def Jarrow_Rudd_Tree(S, K, T, r, sigma, N, Option_type):
    u = math.exp((r - (sigma**2) / 2) * T / N + sigma * math.sqrt(T / N))
    d = math.exp((r - (sigma**2) / 2) * T / N - sigma * math.sqrt(T / N))
    pu = 0.5
    pd = 1 - pu
    disc = math.exp(-r * T / N)

    St = [0] * (N + 1)
    C = [0] * (N + 1)

    St[0] = S * d**N

    for j in range(1, N + 1):
        St[j] = St[j - 1] * u / d

    for j in range(1, N + 1):
        if Option_type == 'P':
            C[j] = max(K - St[j], 0)
        elif Option_type == 'C':
            C[j] = max(St[j] - K, 0)

    for i in range(N, 0, -1):
        for j in range(0, i):
            C[j] = disc * (pu * C[j + 1] + pd * C[j])

    return C[0]

# Input the current stock price and check if it is a number.
S = input("What is the current stock price? ")
while True:
    try:
        S = float(S)
        break
    except:
        print("The current stock price has to be a NUMBER.")
        S = input("What is the current stock price? ")

# Input the strike price and check if it is a number.
K = input("What is the strike price? ")
while True:
    try:
        K = float(K)
        break
    except:
        print("The strike price has to be a NUMBER.")
        K = input("What is the strike price? ")

# Input the expiration_date and calculate the days between today and the expiration date.
while True:
    expiration_date = input("What is the expiration date of the options? (mm-dd-yyyy) ")
    try:
        expiration_date = datetime.strptime(expiration_date, "%m-%d-%Y")
    except ValueError as e:
        print("Error: %s\nTry again." % (e,))
    else:
        break

T = (expiration_date - datetime.utcnow()).days / 365

# Input the continuously compounding risk-free interest rate and check if it is a number.
r = input("What is the continuously compounding risk-free interest rate in percentage(%)? ")
while True:
    try:
        r = float(r)
        break
    except:
        print("The continuously compounding risk-free interest rate has to be a NUMBER.")
        r = input("What is the continuously compounding risk-free interest rate in percentage(%)? ")

# Input the volatility and check if it is a number.
sigma = input("What is the volatility in percentage(%)? ")
while True:
    try:
        sigma = float(sigma)
        if sigma > 100 or sigma < 0:
            print("The range of sigma has to be in [0, 100].")
            sigma = input("What is the volatility in percentage(%)? ")
        break
    except:
        print("The volatility has to be a NUMBER.")
        sigma = input("What is the volatility in percentage(%)? ")

# Normalize r and sigma
r = r / 100
sigma = sigma / 100

# Create a DataFrame for input values
data = {'Symbol': ['S', 'K', 'T', 'r', 'sigma'],
        'Input': [S, K, T, r, sigma]}
input_frame = pd.DataFrame(data, columns=['Symbol', 'Input'],
                            index=['Underlying price', 'Strike price', 'Time to maturity',
                                   'Risk-free interest rate', 'Volatility'])

# Calculate option prices using both models
binomial_model_pricing = {'Option': ['Call', 'Put', 'Call', 'Put'],
                          'Price': [Cox_Ross_Rubinstein_Tree(S, K, T, r, sigma, 1000, 'C'),
                                    Cox_Ross_Rubinstein_Tree(S, K, T, r, sigma, 1000, 'P'),
                                    Jarrow_Rudd_Tree(S, K, T, r, sigma, 1000, 'C'),
                                    Jarrow_Rudd_Tree(S, K, T, r, sigma, 1000, 'P')]}
binomial_model_pricing_frame = pd.DataFrame(binomial_model_pricing, columns=['Option', 'Price'],
                                            index=['Cox-Ross-Rubinstein', 'Cox-Ross-Rubinstein',
                                                   'Jarrow-Rudd', 'Jarrow-Rudd'])

# Plot call/put option prices with different steps
runs = list(range(50, 5000, 50))
CRR_call_prices = []
JR_call_prices = []
CRR_put_prices = []
JR_put_prices = []

for i in runs:
    CRR_call_prices.append(Cox_Ross_Rubinstein_Tree(S, K, T, r, sigma, i, 'C'))
    JR_call_prices.append(Jarrow_Rudd_Tree(S, K, T, r, sigma, i, 'C'))
    CRR_put_prices.append(Cox_Ross_Rubinstein_Tree(S, K, T, r, sigma, i, 'P'))
    JR_put_prices.append(Jarrow_Rudd_Tree(S, K, T, r, sigma, i, 'P'))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(runs, CRR_call_prices, label='Cox-Ross-Rubinstein (Call)')
plt.plot(runs, JR_call_prices, label='Jarrow-Rudd (Call)')
plt.xlabel('Number of Steps')
plt.ylabel('Option Price')
plt.title('Call Option Prices vs. Number of Steps')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(runs, CRR_put_prices, label='Cox-Ross-Rubinstein (Put)')
plt.plot(runs, JR_put_prices, label='Jarrow-Rudd (Put)')
plt.xlabel('Number of Steps')
plt.ylabel('Option Price')
plt.title('Put Option Prices vs. Number of Steps')
plt.legend()

plt.tight_layout()
plt.show()
