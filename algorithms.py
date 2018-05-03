import numpy as np
import matplotlib.pyplot as plt
import Util

##### ALGORITHMS #############################################
# NW Algorithm 3.1, p.37
def backtracking_linesearch(f, gradf, p, x):
    rho = 0.5
    c = 0.05
    a = 1
    it_stop = 1000

    f0 = f(x)
    phi = f(x + a * p)
    dF = gradf(x)
    
    it = 0
    while (phi >= f0 + c * a * dF.dot(p) and it < it_stop):
        a = rho * a
        phi = f(x + a * p)
        it += 1

    convergence = it == it_stop
    return a, convergence
    

# NW Algorithm 3.5, Line Search, p.60
def linesearch(f, grad, p, x, c1 = 1e-4, c2= 0.9):
    a_max = 2
    a0 = 0     # Corresponding to alpha_i-1
    a1 = a_max/2    # Coresponding to alpha_i
    it_stop = 200

    phi0 = f(x, )
    Dphi0 = grad(x).dot(p)


    it = 0
    while it < it_stop:
        phi1 = f(x + a1 * p)

        if (phi1 > phi0 + c1 * a1 * Dphi0) or ( phi1 >= phi0 and it > 0 ):
            return zoom(a0, a1, f, grad, x, p, c1, c2)

        Dphi1 = grad(x + a1 * p).dot(p)

        if np.abs(Dphi1) <= -c2 * Dphi0:
            return a1, True

        if Dphi1 >= 0:
            return zoom(a1, a0, f, grad, x, p, c1, c2)

        a0 = a1

        if a_max == np.inf:
            a1 *= 2
        else:
            a1 = (a1 + a_max)/2
        it += 1
    
    convergence = it_stop == it
    return a1, convergence



# NW Algorithm 3.6 (zoom), p. 61: Use in linesearch
def zoom(a_lo, a_hi, f, grad, x, p, c1, c2):
    phi0 = f(x)
    Dphi0 = grad(x).dot(p)
    it_stop = 100
    
    it = 0
    while it < it_stop:
        a_j = (a_hi + a_lo)/2
        phi_j = f(x + a_j * p)
        phi_lo = f(x + a_lo * p)
        if (phi_j > phi0 + c1 * a_j * Dphi0) or (phi_j >= phi_lo):
            a_hi = a_j
        else:
            Dphi_j = grad(x + a_j * p).dot(p)
            if (np.abs(Dphi_j) <= -c2 * Dphi0):
                break
            if Dphi_j*(a_hi - a_lo) >= 0:
                a_hi = a_lo
            a_lo = a_j
        it += 1
    convergence = it_stop == it
    return a_j, convergence



# Optimization algorithms
# NW Steepest Descent, ~p.21
def steepest_descent(f, grad, x0, TOL = 1e-4, backtrack = False, output = False):
    p = -grad(x0)
    x1 = x0
    it_stop = 200

    it = 0
    while np.linalg.norm(p) > TOL and it < it_stop:
        p /= np.linalg.norm(p)
        
        if backtrack:
            a, iter_succ = backtracking_linesearch(f, grad, p, x1)
        else:
            a, iter_succ = linesearch(f, grad, p, x1, 1e-4, 0.5)
            
        x1 = x1 + a * p
        p = -grad(x1)
        it += 1

    return x1, it, f(x1)

def bfgs_constrained(f, grad, x0, constraint_func = None, TOL = 1e-4, backtrack = False):
    assert constraint_func
    if not Util.is_feasible(constraint_func, x0):
        print("Not feasible x0: {}, c(x):".format(x0))
        raise Exception("Not feasible startpoint")
    it_stop = 1000
    I = np.identity(x0.size)
    H = I
    x1 = x0
    dF1 = grad(x1)

    it = 0
    while np.linalg.norm(dF1) > TOL and it < it_stop:
        it += 1
        dF0 = dF1
        x0 = x1
        p = -H.dot(dF0)

        if p.dot(dF0) > 0: # Check if truly descent dir
            print("Reboot, not descent dir")
            H = I
            continue
        
        p = p/np.linalg.norm(p) #Unit direction vector

        a, iter_succ = linesearch(f, grad, p, x0)
        x1 = x0 + a*p
        inner = 0
        while not Util.is_feasible(constraint_func, x1):
            a = a * 0.1
            x1 = x0 + a*p
            inner += 1

        dF1 = grad(x1)
        s = x1 - x0
        y = dF1 - dF0


        if not y.dot(s) > 0: # Check curvature conditions
            print("Iter {}, Non-update, curvature".format(it))
            continue
        rho = 1/y.dot(s)
        H = (I - rho * np.outer(s, y)).dot(H).dot(I - rho * np.outer(s, y))\
        + rho * np.outer(s, s)
    
    print("BFGS_const, iter {}, f(x) = {}".format(it, f(x1)))
    return x1, it, f(x1)



# Optimization algorithm
def bfgs(f, grad, x0, TOL = 1e-4, backtrack = False):
    it_stop = 10000
    I = np.identity(x0.size)
    H = I
    x1 = x0
    dF1 = grad(x1)
    
    it = 0
    while np.linalg.norm(dF1) > TOL and it < it_stop:
        dF0 = dF1
        x0 = x1
        p = - H.dot(dF0)

        if p.dot(dF0) > 0:
            print("Reboot, Iter {}: Not Descent Dir".format(it))
            H = I
            continue
        p = p/np.linalg.norm(p) # Unit distance
        
        if backtrack:
            a, iter_succ = backtracking_linesearch(f, grad, p, x0)
        else:
            a, iter_succ = linesearch(f, grad, p, x0, 1e-4, 0.5)
        
        x1 = x0 + a * p
        dF1 = grad(x1)
        s = x1 - x0
        y = dF1 - dF0
        
        # Check if "reboot" is needed
        if not s.dot(y) > 0:
            it += 1
            print("Iter {}, Non-update, curvature".format(it))
            continue
        
        # computing rho (6.14 in NW)
        rho = 1/y.dot(s)
        H = (I - rho * np.outer(s, y)).dot(H).dot(I - rho * np.outer(s, y))\
             + rho * np.outer(s, s)

        it += 1

    print("BFGS, iter {}, f(x) = {}".format(it, f(x1)))
    return x1, it, f(x1)
    
    
if __name__ == "__main__":
    import Model2 as m2
    m, n = (15, 2)
    Util.optimize_random(m, n, bfgs, m2.f, m2.df, m2.H)
    plt.show()    
