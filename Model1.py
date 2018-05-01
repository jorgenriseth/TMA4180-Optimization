import numpy as np
import matplotlib.pyplot as plt
from Util import *

# Calculate g for single point z
def gi(x, zi):
    A, c = from_x_to_matrix(x)
    return (zi-c).dot(A.dot(zi-c)) - 1

# Calculate residual r for single point z
def r(x, zi, wi):
    return np.maximum(wi * gi(x, zi), 0)

# Calculate residual vector
def R(x, Z, W):
    m, n = Z.shape
    R = np.zeros(m)
    for i in range(R.size):
        R[i] = r(x, Z[i], W[i])
    return R

# Calculate objective function
def f(x, Z, W):
    m, n = Z.shape
    return np.sum(R(x, Z, W)**2)

# Calculate gradient of g for single point z
def dgi(x, zi):
    n = zi.size
    dg = np.zeros(x.size)
    end = 0
    A, c = from_x_to_matrix(x)
    v = zi - c
    for j in range(n):
        start = end
        end = start + n - j
        dg[start] = v[j]**2
        dg[start+1:end] = 2 * v[j] * v[start+1:end]
    dg[-n:] = -2 * A.dot(v)
    return dg

# Calculate gradient of residual r for single point z
# h is the gi-value for the given point
def dri(x, zi, wi, ri = None):
    n = zi.size
    dr = np.zeros(x.size)
    if ri == None:
        ri = r(x, zi, wi)
    return (ri > 0) * dgi(x, zi) * wi

# Calculate jacobian of residual vector R
def jacobi(x, Z, W, g = None):
    m, n = Z.shape
    J = np.zeros((m, x.size))
    for i in range(m):
        J[i] = dri(x, Z[i], W[i], g)
    return J

# Calculate gradient of objective function
def df(x, Z, W, g = None):
    return 2 * (jacobi(x, Z, W, g).T).dot(R(x, Z, W))

# Evaluate hi(x) for a matrix of columnvectors
def G(X, A, c):
    return (((X - c)  @ A) * (X-c)).sum(axis = 1) - 1
