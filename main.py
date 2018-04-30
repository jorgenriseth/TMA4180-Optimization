import numpy as np
import matplotlib.pyplot as plt
import Model1 as m1
import Model2 as m2
import util as U
import algorithms as alg
import os
import time


# Finite differences verification of gradient in random direction.
def finite_differences(model, m = 10, n = 2):
    print("Finite differences verification of gradient, Model "+str(model))
    x, Z, W = U.generate_random(m, n)
    k = n*(n+1)//2
    p = np.random.randn(n+k)
    p = p/np.linalg.norm(p)
    
    if model == 1:
        f = m1.f
        df = m1.df
    elif model == 2:
        f = m2.f
        df = m2.df
    else:
        raise Exception("Invalid choice of model")
        
    f0 = f(x, Z, W)
    g = df(x, Z, W).dot(p)
    
    if g == 0:
        print("p: \n", p)
        print(df(x, Z, W))
    
    else:
        print("g = %e" %g)
        for ep in 10.0**np.arange(-1, -9, -1):
            g_app = ( f(x + ep*p, Z, W) - f0 )/ep
            error = abs(g_app - g)/abs(g)
            print('ep = %e, error = %e, g_app = %e' % (ep,error, g_app))
            
            
def plot_result(ax, A, b, Z, W, contour_function, xlim =(-5,5), ylim = (-5, 5)):
    xx, yy, C = U.evaluate_function(contour_function, A, b, xlim, ylim)
    
    ax.contourf(xx, yy, C, levels = [np.min(C), 0], cmap = "Wistia")
    ax.plot(Z[W==1, 0], Z[W==1,1], 'o', color = 'b', label = "+")
    ax.plot(Z[W==-1, 0], Z[W==-1,1], 'x', color = 'r', label = "-")
    ax.legend()
    plt.tight_layout()
    return ax   

def optimize(x, Z, W, method, f, df, backtrack = False, output = False):
    a, it, SSE = method(f, df, x, Z, W, backtrack) 
    if output:
        print("Iterations: ", it)
        print("f(a) = ", SSE)
    A, b = U.from_x_to_matrix(a)
    return A, b
    
def solve_and_plot(ax, x, Z, W, method, f, df, contour_function, backtrack = False, output = False):
    A, b = optimize(x, Z, W, method, f, df, backtrack, output)
    plot_result(ax, A, b, Z, W, contour_function)
    return ax
                

if __name__ == "__main__":
    # Make sure figure directory exists
    figdir = "./figures"
    if not os.path.exists(figdir):
            os.makedirs(figdir)

    # Finite differences verification for both models
    finite_differences(1)
    finite_differences(2)
    
    print("\n ######  SLEEP 2 Seconds #######")
    time.sleep(2)
    

    print("##### Start Optimization algorithms ######")    
   
    # Iterate through different data sizes.
    n = 2
    for m in [3, 5, 7, 10, 13, 15]:
        print("\n##### m = "+ str(m) + " #####\n")
        
        x, Z, W = U.generate_random(m, n)

        
        # Initiate figures
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        ax1.set_title("Model 1: Steepest Descent")
        ax2.set_title("Model 1: BFGS")
        ax3.set_title("Model 2: Steepest Descent")
        ax4.set_title("Model 2. BFGS")
        
        # Perform algorithms and save the figures.
        solve_and_plot(ax1, x, Z, W, alg.steepest_descent, m1.f, m1.df, m1.G, backtrack = True, output = True)
        solve_and_plot(ax2, x, Z, W, alg.bfgs_method, m1.f, m1.df, m1.G, backtrack = True, output = True)
        solve_and_plot(ax3, x, Z, W, alg.steepest_descent, m2.f, m2.df, m2.H, backtrack = True, output = True)
        solve_and_plot(ax4, x, Z, W, alg.bfgs_method, m2.f, m2.df, m2.H, backtrack = True, output = True)
        plt.savefig("./figures/" + str(m) + "points_backtrack.png")
        
    plt.show()

