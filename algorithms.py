import numpy as np
from Util import *

##### ALGORITHMS #############################################
def backtracking_linesearch(f, gradf, p, x, Z, W):
    rho = 0.5
    c = 0.05
    a = 1
    
    phi_k = f(x + a * p, Z, W)
    dF = gradf(x, Z, W)
    it = 0
    while (phi_k >= f(x, Z, W) + c * a * dF.dot(p) and it < 200):
        a = rho * a
        phi_k = f(x + a * p, Z, W)
        it += 1

    return a     
    
    
def linesearch(f, grad, p, x, Z, W, c1, c2):
    a_max = 10000
    a0 = 0     # Corresponding to alpha_i-1
    a1 = a_max/2    # Coresponding to alpha_i 


    it = 0
    phi0 = f(x, Z, W)
    Dphi0 = grad(x, Z, W).dot(p)

    while it < 9999:
        phi1 = f(x + a1 * p, Z, W)

        if (phi1 > phi0 + c1 * a1 * Dphi0) or ( phi1 >= phi0 and it > 0 ):
            return zoom(a0, a1, f, grad, x, Z, W, p, c1, c2)

        Dphi1 = grad(x + a1 * p, Z, W).dot(p)

        if np.abs(Dphi1) <= -c2 * Dphi0:
            return a1

        if Dphi1 >= 0:
            return zoom(a1, a0, f, grad, x, Z, W, p, c1, c2)

        a0 = a1

        if a_max == np.inf:
            a1 *= 2
        else:
            a1 = (a1 + a_max)/2
        it += 1
        print(it, a1)
    
    print("Couldn't find alpha by linesearch, return last")
    return a1



# Zoom algorithm, p.61 in NW, for use in linesearch
def zoom(a_lo, a_hi, f, grad, x, Z, W, p, c1, c2):

    it = 0

    phi_0 = f(x, Z, W)

    Dphi_0 = grad(x, Z, W).dot(p)
    
    while it < 99:
        a_j = (a_hi + a_lo)/2
        phi_j = f(x + a_j * p, Z, W)
        phi_lo = f(x + a_lo * p, Z, W)
 
        if (phi_j > phi_0 + c1 * a_j * Dphi_0) or (phi_j >= phi_lo):
            a_hi = a_j
        
        else:
            Dphi_j = grad(x, Z, W).dot(p)

            if (np.abs(Dphi_j) <= -2 * Dphi_0):
                return a_j

            elif Dphi_j*(a_hi - a_lo) >= 0:
                a_hi = a_lo

            a_lo = a_j
        
        it += 1

    print("Counldnt find by zoom, ")
    return a_j


    
def interpolate(alpha2, alpha1, f, grad, p, x, z, w):
    #As described on p. 59 in NW. A cubic interpolation method to determine current step lenght within a interval    
    alpha_curr = alpha2
    alpha_prev = alpha1
    
    phi_curr = f(x + alpha_curr*p, z, w)
    phi_prev = f(x + alpha_prev*p, z, w)
    
    dPhi_curr = np.dot(grad(x + alpha_curr*p, z, w),p)
    dPhi_prev = np.dot(grad(x + alpha_prev*p, z, w),p)
    
    d1 = dPhi_prev + dPhi_curr - 3*(phi_prev - phi_curr)/(alpha_prev - alpha_curr)
    d2 = np.sign(alpha_curr - alpha_prev)*np.sqrt(d1**2 - dPhi_prev*dPhi_curr)
    
    
    return alpha_curr - (alpha_curr - alpha_prev)*(dPhi_curr + d2 - d1)/(dPhi_curr - dPhi_prev + 2*d2)

    

# Optimization algorithms
def steepest_descent(f, grad, x0, Z, W, backtrack = False, tol = 1e-3, output = False):
    p = -grad(x0, Z, W)
    x_k = x0
    it = 0
    
    while np.linalg.norm(p) > tol and it < 10000:
        
        if backtrack:
            a = backtracking_linesearch(f, grad, p, x_k, Z, W)
        else:
            a = linesearch(f, grad, p, x_k, Z, W, 1e-4, 0.9)
            
        x_k = x_k + a * p
        p = -grad(x_k, Z, W)
        it += 1
        
        if it % 500 == 0 and output:
            print("\niter:", it)
            print("α =", a)
            print("f(x) =", f(x_k, Z, W))
            print("\n")
    print("df: ", np.linalg.norm(grad(x_k, Z, W)))
    return x_k, it, f(x_k, Z, W)


# Optimization algorithm
def bfgs_method(f, grad, x0, Z, W, backtrack = True, tol = 1e-4, output = False):
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
        
        if backtrack:
            α_k = backtracking_linesearch(f, grad, p_k, x_k, Z, W)
        else:
            α_k = linesearch(f, grad, p_k, x_k, Z, W, 1e-3, 0.9)
        
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
        if it % 200 == 0  and output:
            print("\niter:", it)
            print("α =", α_k)
            print("f(x) =", f(x_k, Z, W))
            print("\n")
        
    print("df: ", np.linalg.norm(grad(x_k, Z, W)))
    return x_k, it, f(x_k, Z, W)
    
    
if __name__ == "__main__":
    import Model2 as m2

    m, n = (15, 2)
    optimize_random(m, n, steepest_descent, m2.f, m2.df, m2.H)
    plt.show()    
