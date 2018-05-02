import numpy as np
import matplotlib.pyplot as plt
import Util

##### ALGORITHMS #############################################
# NW Algorithm 3.1, p.37
def backtracking_linesearch(f, gradf, p, x):
    rho = 0.5
    c = 0.05
    a = 1
    it_stop = 200

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
def linesearch(f, grad, p, x, c1, c2):
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
    x_k = x0
    it_stop = 200

    it = 0
    while np.linalg.norm(p) > TOL and it < it_stop:
        p /= np.linalg.norm(p)
        
        if backtrack:
            a = backtracking_linesearch(f, grad, p, x_k)
        else:
            a = linesearch(f, grad, p, x_k, 1e-4, 0.5)
            
        x_k = x_k + a * p
        p = -grad(x_k)
        it += 1

    return x_k, it, f(x_k)


# Optimization algorithm
def bfgs(f, grad, x0, TOL = 1e-4, constraints = None, backtrack = False, output = False):
    it_stop = 10000
    I = np.identity(x0.size)
    H = np.identity(x0.size)
    x_k = x0
    dF = grad(x_k)
    
    it = 0
    while np.linalg.norm(dF) > TOL and it < it_stop:
        dF = grad(x_k)
        p_k = - H.dot(dF)

        if not Util.is_descent(p_k, dF):
            H = np.identity(x0.size)
            p_k = - H.dot(dF)

        p_k = p_k/np.linalg.norm(p_k) # Unit distance
        
        if backtrack:
            a_k, iter_succ = backtracking_linesearch(f, grad, p_k, x_k)
        else:
            a_k, iter_succ = linesearch(f, grad, p_k, x_k, 1e-4, 0.9)
        
        x_next = x_k + a_k * p_k
        dF_next = grad(x_next)
        
        s_k = x_next - x_k
        y_k = dF_next - dF
        
        # Check if "reboot" is needed
        if not s_k.dot(y_k) > 0:
            rho_k = 1/s_k.dot(y_k)
            it += 1
            continue
            
        # computing rho (6.14 in NW)
        rho_k = 1/s_k.dot(y_k)
        H = (I - rho_k * s_k * y_k.T) @ H @ (I - rho_k * y_k * s_k.T) + rho_k * s_k * s_k.T

        it += 1
        x_k = x_next
        dF = dF_next

    return x_k, it, f(x_k)
    
    
if __name__ == "__main__":
    import Model2 as m2

    m, n = (15, 2)
    Util.optimize_random(m, n, bfgs, m2.f, m2.df, m2.H)
    plt.show()    
