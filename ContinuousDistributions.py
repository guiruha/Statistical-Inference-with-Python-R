#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:52:16 2020

@author: guillem
"""


# DISTRIBUCIÓN UNIFORME

from scipy.stats import uniform

uniform.pdf(0.5, loc = -1, scale = 2)

uniform.ppf(0.5, loc = -1, scale = 2)

uniform.rvs(size = 30, loc = -1, scale = 2)

# DISTRIBUCIÓN EXPONENCIAL

from scipy.stats import expon
expon.pdf(0.0001, scale = 1./3)

expon.cdf(0.5, scale = 1./3)

expon.rvs(scale = 1./3, size = 10)

# DISTRIBUCIÓN NORMAL

from scipy.stats import norm

norm.pdf(2, loc = 1, scale = 2)

norm.cdf(2, loc = 1, scale = 2)

norm.ppf(0.95, loc = 1, scale = 2)

norm.rvs(loc = 1, scale = 2, size = 5)


