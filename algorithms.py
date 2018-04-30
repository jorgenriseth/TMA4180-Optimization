import numpy as np
from util import *

##### ALGORITHMS #############################################
def backtracking_line_search(f, gradf, p, x, Z, W):
    ρ = 0.5
    c = 0.05
    α = 1
    
    ϕ_k = f(x + α * p, Z, W)
    dF = gradf(x, Z, W)
    it = 0
    while (ϕ_k >= f(x, Z, W) + c * α * dF.dot(p) and it < 200):
        α = ρ * α
        ϕ_k = f(x + α * p, Z, W)
        it += 1

    return α     
    
    
def line_search(f, grad,  p, x, z, w):
    
    #Line search algorithm satisfying strong Wolfe conditions. P. 60 in NW
    alpha_prev = 0
    alpha_max = 10000
    alpha_curr = 2
    
    #Picked some random numbers
    c1 = 1e-4
    c2 = 0.5
    
    i = 1
    while i < 1000:
        dPhi_0 = grad(x, z, w).dot(p)
        dPhi_i = grad(x + alpha_curr*p, z, w).dot(p)
        
        f_prev = f(x + alpha_prev*p, z, w)
        f_next = f(x + alpha_curr * p, z, w)
        f_curr = f(x, z, w)
        
        if (f_next > f_curr + c1 * alpha_curr * dPhi_0) or (f_next >= f_prev and i > 1):
            return zoom(f, grad, p, x, c1, c2, alpha_curr, alpha_prev, z, w)
        
        if np.abs(dPhi_i) <= -c2*dPhi_0: 
            return alpha_curr
        if dPhi_i >= 0:
            return zoom(f, grad, p, x, c1, c2, alpha_prev, alpha_curr, z, w)
        
        alpha_prev = alpha_curr
        alpha_curr = (alpha_curr + alpha_max)/2
        
        i += 1
        
    return alpha_curr
        
    
    
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

    
def zoom(f, grad, p, x, c1, c2, Alpha_high, Alpha_low, z, w):
    
    #Zoom algorithm as described on p. 61 in NW. Used for finding optimal step lenght
    alpha_low = Alpha_low
    alpha_high = Alpha_high
    
    i = 0
    tol = 10**-4
    alpha_curr = alpha_high
    alpha_prev = alpha_low
    while i < 1000:
        #Interpolate, choose alpha_j between hi & low
        #alpha_j = (alpha_high + alpha_low)/2
        
        if (np.abs(alpha_high - alpha_low) < tol):
            return alpha_high
        alpha_j = interpolate(alpha_high, alpha_low, f, grad, p, x, z, w)
        #print(alpha_j)
        phi_j = f(x + alpha_j*p, z, w)
        phi_low = f(x + alpha_low*p, z, w)
        
        dPhi_0 = np.dot(grad(x, z, w),p) # Bør være grad i x
        
        
        if (phi_j > f(x, z, w) + c1*alpha_j*dPhi_0) or (phi_j >= phi_low):
            alpha_high = alpha_j
        else:
            dPhi_i = np.dot(grad(x + alpha_j*p, z, w),p)
            if np.abs(dPhi_i) <= -c2*dPhi_0:
                return alpha_j
            if dPhi_i*(alpha_high - alpha_low) >= 0:
                alpha_high = alpha_low
            alpha_low = alpha_j
      
    return alpha_j


# Optimization algorithms
def steepest_descent(f, grad, x0, Z, W, backtrack = True, tol = 1e-3, output = False):
    p = -grad(x0, Z, W)
    x_k = x0
    it = 0
    
    while np.linalg.norm(p) > tol and it < 10000:
        
        if backtrack:
            α = backtracking_line_search(f, grad, p, x_k, Z, W)
        else:
            α = line_search(f, grad, p, x_k, Z, W)
            
        x_k = x_k + α * p
        p = -grad(x_k, Z, W)
        it += 1
        
        if it % 500 == 0 and output:
            print("\niter:", it)
            print("α =", α)
            print("f(x) =", f(x_k, Z, W))
            print("\n")
    print("df: ", np.linalg.norm(grad(x_k, Z, W)))
    return x_k, it, f(x_k, Z, W)


# Optimization algorithm
def bfgs_method(f, grad, x0, Z, W, backtrack = True, tol = 1e-5, output = False):
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
            α_k = backtracking_line_search(f, grad, p_k, x_k, Z, W)
        else:
            α_k = line_search(f, grad, p_k, x_k, Z, W)
        
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
    A = np.array((1, 0, 0, 1)).reshape(2,2)
    f = lambda x, Z, W: 0.5 * x.T.dot(A).dot(x)
    df = lambda x, Z, W: A.dot(x)
    x0 = np.array((10, 7))
    Z = np.zeros(2)
    W = np.zeros(2)
    p = np.random.randn(2)
    print(backtracking_line_search(f, df, p, x0, Z, W))
    print(steepest_descent(f, df, x0, Z, W))
    
    
    
    

