import numpy as np
import matplotlib.pyplot as plt
import time

def from_x_to_matrix(x):
    n = int((-3 + np.sqrt(9+8*x.size))//2)
    k = x.size - n
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    #Insert first k coefficients into matrix
    end = 0
    for j in range(n):
        start = end
        end = start + n-j
        A[j, j:] = x[start:end]
        A[j+1:,j] = A[j, j+1:]
    b = x[-n:] #Insert last n coefficients into vector
    return A, b

# Calculate h for single point z
def hi(x, zi):
    A, b = from_x_to_matrix(x)
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

##### ALGORITHMS #############################################
def backtracking_line_search(f, gradf, p, x, Z, W):
    ρ = 0.5
    c = 0.05
    α = 0.5
    
    ϕ_k = f(x + α * p, Z, W)
    dF = gradf(x, Z, W)
    it = 0
    while (ϕ_k >= f(x, Z, W) + c * α * dF.dot(p) and it < 200):
        α = ρ * α
        ϕ_k = f(x + α * p, Z, W)
        it += 1
#     print(it)
    return α


# Optimization algorithms
def steepest_descent(f, grad, x0, Z, W, tol = 1e-3):
    p = -df(x0, Z, W)
    x_k = x0
    it = 0
    
    while np.linalg.norm(p) > tol and it < 10000:
        
        α = backtracking_line_search(f, grad, p, x_k, Z, W)
        x_k = x_k + α * p
        p = -df(x_k, Z, W)
        it += 1
        
        if it % 500 == 0:
            print("\niter:", it)
            print("α =", α)
            print("f(x) =", f(x_k, Z, W))
            
    #print("f(x) =", f(x_k, Z, W), "df(x) =" , df(x_k, Z ,W))
    print("\n")
    return x_k, it


# Optimization algorithm
def bfgs_method(f, grad, x0, Z, W, tol = 1e-3):
    m, n = Z.shape
    k = n*(n+1)//2

    I = np.identity(n + k)
    H = I
    
    x_k = x0
    dF = grad(x_k, Z, W)
    
    it = 0
    while np.linalg.norm(dF) > tol and it < 10000:
        dF = grad(x_k, Z, W)
        
        p_k = - H.dot(dF)
        p_k = p_k/np.linalg.norm(p_k)
        
        α_k = backtracking_line_search(f, grad, p_k, x_k, Z, W)
        
        x_next = x_k + α_k * p_k
        dF_next = grad(x_next, Z, W)
        
        s_k = x_next - x_k
        y_k = dF_next - dF
        
        # Check if "reboot" is needed
        if s_k.dot(y_k) == 0:
            H = I
            continue
            
        # computing rho (6.14 in NW)
        ρ_k = 1/(np.dot(s_k,y_k))
        
        
        H = (I - ρ_k * s_k * y_k.T) @ H @ (I - ρ_k * y_k * s_k.T) + ρ_k * s_k * s_k.T
        
        it += 1
        
        x_k = x_next
        dF = dF_next
        
        # Print progress
        if it % 200 == 0:
            print("\niter:", it)
            print("α =", α_k)
            print("f(x) =", f(x_k, Z, W))
    print("f(x) =", f(x_k, Z, W), "df(x) =" , df(x_k, Z ,W))
        
    return x_k, k



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



# Generate a random set of points, weights, and unknowns
def generate_random(m, n):
    Z = np.random.randn(m*n).reshape(m, n)
    W = np.random.choice([-1.0, 1.0], m)
    x = np.array((1,0,1,0,0))
    return x, Z, W


# Given A, b/c, evaluate function values for contourplot.
def evaluate_function(func, A, b, xlim = (-5, 5), ylim = (-5, 5)):
    x = np.linspace(*xlim, 101)
    y = np.linspace(*ylim, 101)
    xx, yy = np.meshgrid(x, y)
    X = np.stack((xx.flatten(), yy.flatten())).T
    return xx, yy, H(X, A, b).reshape(xx.shape)


# Minimize and visualize objective function values for a random set of points
def optimize_random(m, n, method, func, output = False):
    x, Z, W = generate_random(m, n)
    a, it = method(f, df, x, Z, W)
    print("Iterations: ", it)
    A, b = from_x_to_matrix(a)
    xx, yy, C = evaluate_function(func, A, b)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.contourf(xx, yy, C, levels = [np.min(C), 0], cmap = "Wistia")
    ax1.plot(Z[W==1, 0], Z[W==1,1], 'o', color = 'b', label = "+")
    ax1.plot(Z[W==-1, 0], Z[W==-1,1], 'x', color = 'r', label = "-")
    ax1.legend()
    
    if output:
        print("W                Z                            W")
        Res = R(a, Z, W)
        for i in range(m):
            print(W[i], " ", Z[i], " "*10,  Res[i])

    return ax1

