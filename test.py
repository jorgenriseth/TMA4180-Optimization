import numpy as np
import matplotlib.pyplot as plt

import Util
import Model2 as m2
import algorithms as alg

def grad_constraints(x, eig_lo, eig_hi):
    dc1 = lambda s: np.array((1, 0, 0, 0, 0))
    dc2 = lambda s: np.array((-1, 0, 0, 0, 0))
    dc3 = lambda s: np.array((0, 0, 1, 0, 0))
    dc4 = lambda s: np.array((0, 0, -1, 0, 0))
    dc5 = lambda s: np.array(( np.sqrt( s[2]/(4*s[0]) )         , \
                               -np.sqrt( s[1]**2/(eig_lo**2 + s[1]**2) ) , \
                               np.sqrt( s[0]/(4*s[2]) ), 0, 0))
    return np.array( (dc1(x), dc2(x), dc3(x), dc4(x), dc5(x)) )

def constraints(x, eig_lo, eig_hi):
    c1 = lambda s: s[0] - eig_lo
    c2 = lambda s: -s[0] + eig_hi
    c3 = lambda s: s[2] - eig_lo
    c4 = lambda s: -s[2] + eig_hi
    c5 = lambda s: np.sqrt(s[0]*s[2]) - np.sqrt(eig_lo**2 + s[1]**2)
    return np.array((c1(x), c2(x), c3(x), c4(x), c5(x)))

def is_feasible(cf, x):
    return (cf(x) > 0).all()

# C is constraints
def P(x, mu, C,  f):
    if not (C > 0).all():
        return np.inf
    
    return f(x) - mu * np.sum(np.log(C))

def dP(x, mu, C, dC, df):
    return df(x) - mu * np.sum(dC.T/C, axis = 1)
    
def compute_lagrange(mu, C):
    return mu/C

def lagrange(x, f, L, C):
    return f(x) - np.sum(C*L)

def grad_lagrange(x,df, L, dC):
    return df(x) - np.sum(dC.dot(L))

def check_KKT(x, df, L, C, dC, TOL):
    if np.linalg.norm(grad_lagrange(x, df, L, dC)) > TOL:
        return False
    
    elif (C < 0).any():
        return False
    
    elif (L < 0).any():
        return False
    
    elif (L * C > TOL).any():
        return False
    return True

def barrier(x0, f, df, eigs, mu0, TOL):
    it = 1
    it_stop = 1000
    mu = 2 * mu0
    x = x0
    C = constraints(x0, *eigs)
    dC = grad_constraints(x0, *eigs)
    p = lambda x: P(x, mu, C, f)
    dp = lambda x: dP(x, mu, C, dC, df)
    
    while it < it_stop:
        print("Iter:", it)
        print(C)
        print(dC)
        print(f(x0))
        print(dp(x0))


        x1, linesearch_iter, fk = alg.bfgs(p, dp, x, TOL = 1/it**2, backtrack = True)
        print("bfgs_done")
        C = constraints(x, *eigs)
        dC = grad_constraints(x, *eigs)
        p = lambda x: P(x, mu, C, f)
        dp = lambda x: dP(x, mu, C, dC, df)
        
        L = compute_lagrange(mu, C)
        
        if check_KKT(x, df, L, C, dC, TOL):
            print("KKT")
            return x, it, fk
        mu /= 2
        it += 1
        
    print("Couldnt find by barrier")
    return x, it, fk

def bfgs_constrained(f, grad, x0, cf, dcf, TOL = 1e-4):
    it_stop = 10000
    I = np.identity(x0.size)
    H = I
    dF1 = grad(x0)
    x1 = x0
    assert is_feasible(cf, x0)

    it = 0
    while np.linalg.norm(dF1) > TOL and it < it_stop:
        dF0 = dF1
        x0 = x1
        p = -H.dot(dF0)

        if not Util.is_descent(p, dF0):
            print("Reboot, descent")
            H = I
            continue
        p = p/np.linalg.norm(p)

        a, iter_succ = alg.linesearch(f, grad, p, x0, 1e-4, 0.9)

        if not iter_succ:
            print("Iter {}, non convergent step-size". format(it))
        
        x1 = x0 + a * p
        while not is_feasible(cf, x1):
            print("{} not feasible".format(cf(x1)) )
            a /= 2
            x1 = x0 + a * p
            if a == 0:
                print("Couldn't find min by BFGS")
                break

        dF1 = grad(x1)
        s = x1 - x0
        y = dF1 - dF0

        if not s.dot(y) > 0:

            it += 1
            continue
        rho = 1/y.dot(s)
        H = (I - rho * np.outer(s, y)).dot(H).dot(I - rho * np.outer(s, y))\
             + rho * np.outer(s, s)
        it += 1

    print("BFGS iter: {}, f(x) = {}".format(it, f(x1)))
    return x1, it, f(x1)

# Unconsttrained
def test_method(method, constraints = None, feasible = True):
    m, n = 30, 2
    eig =  np.abs(np.array((1, 16)) + np.random.randn(2))

    if constraints:
        cf = lambda x: constraints(x, *eig )

        # Create and generate
    if feasible:
        x = Util.get_random_feasible(constraints, eig)
    else:
        x = Util.get_non_feasible(eig)

    print("\nFOUND x\n")
    A, b = Util.from_x_to_matrix(x)
    Z, W = Util.generate_random(m, n, scale = 0.9)
    W = Util.pre_classify(x, Z, W, m2.H)
    f, df = Util.set_function(m2.f, m2.df, Z, W)

    # Visualize goal
    fig = plt.figure()
    ax = fig.add_subplot(111)
    Util.visualize(ax, A, b, Z, W, m2.H)
    plt.show()

    # Reset x
    x0 = Util.get_feasible(eig*np.array((1.2, 0.8)))
    A0, b0 = Util.from_x_to_matrix(x0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    Util.visualize(ax, A0, b0, Z, W, m2.H)
    plt.show()

    # Find and visualize x anew
    if constraints:
        x1, it1, f1 = method(f, df, x0)
    else:
        x1, it1, f1 = method(f, df, x0, backtrack = False)
    A1, b1 = Util.from_x_to_matrix(x1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    Util.visualize(ax, A1, b1, Z, W, m2.H)
    plt.show()

if __name__ == "__main__":
    test_method(alg.bfgs, constraints, feasible = False)