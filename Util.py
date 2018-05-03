import numpy as np
import matplotlib.pyplot as plt

# Retrieve dimensions from matrix x
def get_n(x):
    return int((-3 + np.sqrt(9+8*x.size))//2)

# Insert x-values into A, b
def from_x_to_matrix(x):
    n = get_n(x)
    A = np.zeros((n, n))
    b = np.zeros(n)
    end = 0
    for j in range(n): # Insert first k coefficients into matrix
        start = end
        end = start + n-j
        A[j, j:] = x[start:end]
        A[j+1:,j] = A[j, j+1:]
    b = x[-n:] # Insert last n coefficients into vector
    return A, b

# For easier function calls, for given dataset f(x)
def set_function(F, dF, Z, W):
    return lambda x: F(x, Z, W), lambda x: dF(x, Z, W)

def is_descent(p, gradient_vector):
    return p.dot(gradient_vector) < 0

def is_feasible(cf, x):
    return (cf(x) > 0).all()

# Generate random points z, weights w
def generate_random(m, n, scale = 1):
    Z = scale * np.random.randn(m*n).reshape(m, n)
    W = np.random.choice([-1.0, 1.0], m)
    return Z, W

def pre_classify(x, Z, W, contour_func):
    A, b = from_x_to_matrix(x)
    H = contour_func(Z, A, b)
    W = (H > 0) * -1 + (H < 0) * 1
    return W

def get_feasible(eigs):
    x = np.zeros(5)
    x[0] = (eigs[0] + eigs[1])/2
    x[2] = (eigs[0] + eigs[1])/2
    x[1] = np.sqrt(x[0]*x[2] - eigs[0]*eigs[1])
    return x

def get_random_feasible(constraints, eigs):
    cf = lambda x: constraints(x, *eigs)
    x = np.random.randn(5)
    while not is_feasible(cf, x):
        x = np.random.randn(5)
    return x




def get_non_feasible(eigs):
    x = np.zeros(5)
    x[0] = (eigs[0] + eigs[1])/2
    x[2] = -(eigs[0] + eigs[1])/2
    x[1] = 0
    return x

# Generate test set fitted to start value
def generate_test_set(m, n, contourfunc, x_rand = False, misclass = False):
    k = n*(n+1)//2
    Z, W = generate_random(m, n)
    if x_rand:
        x = np.random.randn(n+k)
    else:
        assert n == 2
        x = np.array((1, 0, 1, 0, 0))

    A, b = from_x_to_matrix(x)
    H =  contourfunc(Z, A, b)
    W = (H > 0) * -1 + (H < 0) * 1
    if misclass:
        roll = np.random.rand(m)
        flips = roll > 0.85
        W = np.logical_and(np.logical_and(flips, W == 1), np.logical_and(np.logical_not(flips), W == -1)) * (-1) \
                + np.logical_and(np.logical_and(flips, W == -1), np.logical_and(np.logical_not(flips), W == 1)) * 1
    return x, Z, W

# Given A, b/c, evaluate function values for contourplot.
def evaluate_function(contourfunc, A, b, xlim = (-5, 5), ylim = (-5, 5)):
    x = np.linspace(*xlim, 101)
    y = np.linspace(*ylim, 101)
    xx, yy = np.meshgrid(x, y)
    X = np.stack((xx.flatten(), yy.flatten())).T
    return xx, yy, contourfunc(X, A, b).reshape(xx.shape)
 

# Minimize and visualize objective function values for a random set of points
def optimize_random(m, n, method, F, dF, contourfunc, output = False):
    x, Z, W = generate_test_set(m, n, contourfunc, x_rand = True)
    f = lambda x: F(x, Z, W)
    df = lambda x: dF(x, Z, W)
    x = np.random.randn(x.size)
    a, it, F = method(f, df, x)
    print("Iterations: ", it)
    A, b = from_x_to_matrix(a)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    visualize(ax1, A, b, Z, W, contourfunc)
    
    if output:
        print("W                Z                            W")
        Res = m2.R(a, Z, W)
        for i in range(m):
            print(W[i], " ", Z[i], " "*10,  Res[i])

    return fig

def visualize(ax, A, b, Z, W, contourfunc):
    xx, yy, C = evaluate_function(contourfunc, A, b)
    ax.contourf(xx, yy, C, levels = [np.min(C), 0], cmap = "Wistia")
    ax.plot(Z[W==1, 0], Z[W==1,1], 'o', color = 'b', label = "+")
    ax.plot(Z[W==-1, 0], Z[W==-1,1], 'x', color = 'r', label = "-")
    ax.legend()
    return ax

if __name__ == "__main__":
    import Model2 as m2
    import algorithms as alg
    m, n = 50, 2
    eig =  np.sort(np.abs(np.random.randn(2))) * 11

    #x = get_feasible_x(eig)
    x = get_non_feasible(eig)
    A, b = from_x_to_matrix(x)
    print(A)
    print(np.linalg.eigvals(A))
    Z, W = generate_random(m, n, scale = 1.4)
    W = pre_classify(x, Z, W, m2.H)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    visualize(ax, A, b, Z, W, m2.H)
    plt.show()