import numpy as np
import matplotlib.pyplot as plt
import time
import util


# Calculate h for single point z
def hi(x, zi):
    A, b = util.from_x_to_matrix(x)
    return zi.dot(A.dot(zi)) + b.dot(zi) - 1

# Calculate residual r for single point z
def r(x, zi, wi):
    return np.maximum(wi * hi(x, zi), 0)
    
# Evaluate hi(x) for a matrix of columnvectors
def H(X, A, b):
    return ((X @ A) * X).sum(axis = 1) + (b * X).sum(axis = 1) - 1

#Calculate residual vector
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

# Calculate gradient oh h for single point z
def dhi(x, zi):
    n = zi.size
    k = n*(n+1)//2
    dh = np.zeros(k+n)
    end = 0
    for j in range(n):
        start = end
        end = start + n-j
        dh[start] = zi[j]**2
        dh[start+1:end] = 2 * zi[j] * zi[start+1:end]
    dh[-n:] = zi
    return dh

# Calculate gradient of residual r for  single point z
# h is the hi value for the given point
def dri(x, zi, wi, ri = None):
    n = zi.size
    dr = np.zeros(x.size)
    if ri == None:
        ri = r(x, zi, wi)
    return (ri > 0) * dhi(x, zi) * wi

# Calculate jacobian of residual vector R
def jacobi(x, Z, W, h = None):
    m, n = Z.shape
    J = np.zeros((m, x.size))
    for i in range(m):
        J[i] = dri(x, Z[i], W[i], h)
    return J

# Calculate gradient of objective function
def df(x, Z, W, h = None):
    return 2 * (jacobi(x, Z, W, h).T).dot(R(x, Z, W))



################# RUN TEST ###################
# Finite difference test of gradient
def finite_difference_test(m = 10, n = 2):
    x, Z, W = generate_random(m, n)
    k = n*(n+1)//2
    p = np.random.randn(n+k)
    p = p/np.linalg.norm(p)
    f0 = f(x, Z, W)
    g = df(x, Z, W).dot(p)
    if g == 0:
        print("p: \n", p)
        print(df(x, Z, W))

    else:
        print("g = %e" %g)
        for ep in 10.0**np.arange(2, -9, -1):
            g_app = (f(x+ep*p, Z, W)-f0)/ep
            error = abs(g_app-g)/abs(g)
            print('ep = %e, error = %e, g_app = %e' % (ep,error, g_app))

