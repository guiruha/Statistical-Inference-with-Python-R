#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:53:42 2020

@author: guillem
"""

# DISTRIBUCIÓN BINOMIAL

from scipy.stats import binom

# para una B(10, 0.25)
binom.cdf(0, n = 10, p = 0.25)

binom.pmf(0, n = 10, p = 0.25)

binom.cdf(4, n = 10, p = 0.25)

binom.pmf(4, n = 10, p = 0.25)

binom.rvs(n = 10, p = 0.25, size = 100)

import numpy as np
import matplotlib.pyplot as plt

n, p = 10, 0.25
x = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p))
fig = plt.figure(figsize = (5, 2.7))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x, binom.pmf(x, n, p), 'bo', ms = 8, label = "binom pmf")
ax.vlines(x, 0, binom.pmf(x, n, p), color = "b", lw = 5, alpha = 0.5)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(5)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x, binom.cdf(x, n, p), 'bo', ms = 8, label = 'binom pmf')
ax.vlines(x, 0, binom.cdf(x, n, p), colors = 'b', lw = 5, alpha =0.5)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(5)
fig.suptitle("Distribución Binomial")
plt.show()

# DISTRIBUCIÓN GEOMETRICA

from scipy.stats import geom

# LA GEOMETRICA DE SCIPY EMPIEZA EN 1 NO EN 0!!! si queremos corregir podremos loc = -1

geom.pmf(0, p = 0.25, loc = -1)

geom.cdf(0, p= 0.25, loc = -1)

geom.cdf(4, p = 0.25, loc = -1)

geom.rvs(p=0.25, size = 20, loc = -1)


# Comprobamos lo del loc

geom.cdf(range(5), p = 0.3, loc = 0)
geom.cdf(range(5), p = 0.3, loc = -1)

geom.stats(p = 0.25, loc = 0, moments = "mv")

geom.stats(p = 0.25, loc = -1, moments = "mv")

n, p = 10, 0.25
x = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p))
fig = plt.figure(figsize = (5, 2.7))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x, geom.pmf(x, n, p), 'bo', ms = 5, label = "geom pmf")
ax.vlines(x, 0, geom.pmf(x, p), color = "b", lw = 5, alpha = 0.5)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(5)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x, geom.cdf(x, p), 'bo', ms = 5, label = 'geom pmf')
ax.vlines(x, 0, geom.cdf(x, p), colors = 'b', lw = 5, alpha =0.5)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(5)
fig.suptitle("Distribución Geometrica")
plt.show()

# DISTRIBUCIÓN BINOMIAL NEGATIVA

from scipy.stats import nbinom

nbinom.pmf(k = 5, n = 2, p = 0.1)

nbinom.pmf(k = 5, n = 2, p = 0.1, loc = 0)

nbinom.cdf(k = 4, n = 2, p = 0.1)

1 - nbinom.cdf(k = 4, n = 2, p = 0.1)

nbinom.rvs(n = 2, p = 0.1, size = 100)

params = nbinom.stats(n = 2, p = 0.1, moments = 'mv')

'E(X) = {} y Var(X) = {}'.format(params[0], params[1])

n, p = 10, 0.25
x = np.arange(nbinom.ppf(0.01, n, p), nbinom.ppf(0.99, n, p))
fig = plt.figure(figsize = (5, 2.7))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x, nbinom.pmf(x, n, p), 'bo', ms = 8, label = "nbinom pmf")
ax.vlines(x, 0, nbinom.pmf(x, n, p), color = "b", lw = 5, alpha = 0.5)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(5)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x, nbinom.cdf(x, n, p), 'bo', ms = 8, label = 'nbinom pmf')
ax.vlines(x, 0, nbinom.cdf(x, n, p), colors = 'b', lw = 5, alpha =0.5)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(5)
fig.suptitle("Distribución Binomial Negativa")
plt.show()

# DISTRIBUCIÓN POISSON

from scipy.stats import poisson

poisson.pmf(0, mu = 3)

poisson.cdf( 0, mu = 3)

sum(poisson. pmf(range(0, 10), mu = 3))

poisson.cdf(10, mu = 3)

poisson.stats(mu = 3, moments = 'mv') # Tonteria


mu = 10 # mu = lambda
x = np.arange(poisson.ppf(0.01, mu),poisson.ppf(0.99, mu))
fig =plt.figure(figsize=(5, 2.7))
ax = fig.add_subplot(1,2,1)
ax.plot(x, poisson.pmf(x, mu), 'bo', ms=5, label='poisson pmf')
ax.vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=2, alpha=0.5)
for tick in ax.xaxis.get_major_ticks():
  tick.label.set_fontsize(5)
for tick in ax.yaxis.get_major_ticks():
  tick.label.set_fontsize(5) 
ax = fig.add_subplot(1,2,2)
ax.plot(x, poisson.cdf(x, mu), 'bo', ms=5, label='poisson cdf')
ax.vlines(x, 0, poisson.cdf(x, mu), colors='b', lw=2, alpha=0.5)
for tick in ax.xaxis.get_major_ticks():
  tick.label.set_fontsize(5)
for tick in ax.yaxis.get_major_ticks():
  tick.label.set_fontsize(5)
fig.suptitle('Distribucion de Poisson')
plt.show()

# DISTRIBUCIÓN HIPERGEOMETRICA

from scipy.stats import hypergeom

hypergeom.pmf(1,M=15+10,n=15,N=3)
hypergeom.cdf(1,M=15+10,n=15,N=3)
1-hypergeom.cdf(1,M=15+10,n=15,N=3)

hypergeom.rvs(M=15+10,n=15,N=3,size=100)

[M, n, N] = [20, 7, 12]
x = np.arange(max(0, N-M+n),min(n, N))
fig =plt.figure(figsize=(5, 2.7))
ax = fig.add_subplot(1,2,1)
ax.plot(x, hypergeom.pmf(x, M, n, N), 'bo', ms=5, label='hypergeom pmf')
ax.vlines(x, 0, hypergeom.pmf(x, M, n, N), colors='b', lw=2, alpha=0.5)
ax.set_ylim([0, max(hypergeom.pmf(x, M, n, N))*1.1])
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(5) 
ax = fig.add_subplot(1,2,2)
ax.plot(x, hypergeom.cdf(x, M, n, N), 'bo', ms=5, label='hypergeom cdf')
ax.vlines(x, 0, hypergeom.cdf(x, M, n, N), colors='b', lw=2, alpha=0.5)
for tick in ax.xaxis.get_major_ticks():
  tick.label.set_fontsize(5)
for tick in ax.yaxis.get_major_ticks():
  tick.label.set_fontsize(5)
fig.suptitle('Distribucion Hipergeometrica')
plt.show()
