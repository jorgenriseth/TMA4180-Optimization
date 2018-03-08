import numpy as np
import matplotlib.pyplot as plt

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

# Generate random points z, weights w, and x0
def generate_random(m, n):
    Z = np.random.randn(m*n).reshape(m, n)
    W = np.random.choice([-1.0, 1.0], m)
    x = np.array((1,0,1,0,0))
    return x, Z, W   


# Given A, b/c, evaluate function values for contourplot.
def evaluate_function(contourfunc, A, b, xlim = (-5, 5), ylim = (-5, 5)):
    x = np.linspace(*xlim, 101)
    y = np.linspace(*ylim, 101)
    xx, yy = np.meshgrid(x, y)
    X = np.stack((xx.flatten(), yy.flatten())).T
    return xx, yy, contourfunc(X, A, b).reshape(xx.shape)
 

# Minimize and visualize objective function values for a random set of points
def optimize_random(m, n, method, f, df, contourfunc, output = False):
    x, Z, W = generate_random(m, n)
    a, it = method(f, df, x, Z, W)
    print("Iterations: ", it)
    A, b = from_x_to_matrix(a)
    xx, yy, C = evaluate_function(contourfunc, A, b)
    
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

    return fig



if __name__ == "__main__":
    import Model2 as m2
    import algorithms as alg
    m, n = (10, 2)
    optimize_random(m, n, alg.steepest_descent, m2.f, m2.df, m2.H)
    plt.show()    
