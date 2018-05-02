import numpy as np
import matplotlib.pyplot as plt

import Util as u
import Model2 as m2
import algorithms as alg


m, n = 30, 2

def grad_constraints(x, eig_lo, eig_hi):
    dc1 = lambda s: np.array((1, 0, 0, 0, 0))
    dc2 = lambda s: np.array((-1, 0, 0, 0, 0))
    dc3 = lambda s: np.array((0, 0, 1, 0, 0))
    dc4 = lambda s: np.array((0, 0, -1, 0, 0))
    dc5 = lambda s: np.array(( np.sqrt( s[2]/(4*s[0]) )         , \
                               np.sqrt( s[1]**2/(eig_lo**2+s[1]**2) ) , \
                               np.sqrt( s[0]/(4*s[2]) ), 0, 0))
    return np.array( (dc1(x), dc2(x), dc3(x), dc4(x), dc5(x)) )

def constraints(x, eig_lo, eig_hi):
    c1 = lambda s: s[0] - eig_lo
    c2 = lambda s: -s[0] + eig_hi
    c3 = lambda s: s[2] - eig_lo
    c4 = lambda s: -s[2] + eig_hi
    c5 = lambda s: np.sqrt(s[0]*s[2]) - np.sqrt(eig_lo**2 + s[1]**2)
    return np.array((c1(x), c2(x), c3(x), c4(x), c5(x)))

# C is constraints
def P(x, Z, W, mu, C,  f):
    if not (C > 0).all():
        return np.inf
    
    return f(x, Z, W) - mu * np.sum(np.log(C))

def dP(x, Z, W, mu, C, dC, df):
    return df(x, Z, W) - mu * np.sum(dC.T/C, axis = 1)
    
def compute_lagrange(mu, C):
    return mu/C

def lagrange(x, Z, W, f, L, C):
    return f(x, Z, W) - np.sum(C*L)

def grad_lagrange(x, Z, W, df, L, dC):
    return df(x, Z, W) - np.sum(dC.dot(L))

def check_KKT(x, Z, W, df, L, C, dC, TOL):
    if np.linalg.norm(grad_lagrange(x, Z, W, df, L, dC)) < TOL:
        return False
    
    elif (C < 0).any():
        return False
    
    elif (L < 0).any():
        return False
    
    elif (L * C > TOL).any():
        return False
    return True

def barrier(x0, Z, W, f, df, eigs, mu0, TOL):
    it = 1
    mu = 2 * mu0
    x = x0
    C = constraints(x0, *eigs)
    dC = grad_constraints(x0, *eigs)
    p = lambda x, Z, W: P(x, Z, W, mu, C, f)
    dp = lambda x, Z, W: dP(x, Z, W, mu, C, dC, df)
    
    while it < 99:
        print("Iter:", it)
        print(C)
        print(dC)
        print(f(x0, Z, W))

        x, it, fk = alg.bfgs(p, dp, x, TOL = 1/it**2, backtrack = False, output = True)
        print("bfgs_done")
        C = constraints(x, *eigs)
        dC = grad_constraints(x, *eigs)
        p = lambda x, Z, W: P(x, Z, W, mu, C, f)
        dp = lambda x, Z, W: dP(x, Z, W, mu, C, dC, df)
        
        L = compute_lagrange(mu, C)
        
        import pdb; pdb.set_trace()
        if check_KKT(x, Z, W, df, L, C, dC, TOL):
            return x, it, fk
        mu /= 2
        
    print("Couldnt find by barrier")
    return x, it, fk



if __name__ == "__main__":
    m, n = 30, 2
    x, Z, W = u.generate_test_set(m, n, m2.H, x_rand = True, misclass = False)
    A, b = u.from_x_to_matrix(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    u.visualize(ax, A, b, Z, W, m2.H)
    plt.show()

    x0 = np.array((4, 1, 3, 1, 1))
    A, b = u.from_x_to_matrix(x0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    u.visualize(ax, A, b, Z, W, m2.H)
    plt.show()

    f, df = u.set_function(m2.f, m2.df, Z, W)

    x, it, f = alg.bfgs(f, df, x0, backtrack = False)
    A, b = u.from_x_to_matrix(x)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    u.visualize(ax, A, b, Z, W, m2.H)
    plt.show()
    print(it)
