import numpy as np
import matplotlib.pyplot as plt

import Util
import algorithms as alg
import Model2 as m2

def constraints(x, eig):
    c1 = x[0] - eig[0]
    c2 = -x[0] + eig[1]
    c3 = x[2] - eig[0]
    c4 = -x[2] + eig[1]
    c5 = x[0]*x[2] - (eig[0]**2 + x[1]**2)
    #c5 = np.sqrt(x[0] * x[2]) - np.sqrt((eig[0]**2 + x[1]**2))
    return np.array((c1, c2, c3, c4, c5))

def grad_constraints(x, eig):
    dc = np.zeros((5, 5))
    dc[0, 0] = 1
    dc[1, 0] = -1
    dc[2, 2] = 1
    dc[3, 2] = -1
    dc[4,:3] = [ x[2], -2*x[1], x[0] ]
    #dc[4,:3] = [np.sqrt(x[2]/(4*x[0])), -x[1]/np.sqrt(eig[0]**2 + x[1]**2), np.sqrt(x[0]/(4*x[2]))]
    return dc

def set_constraints(eig):
    return lambda x: constraints(x, eig),\
             lambda x: grad_constraints(x, eig)

def lagrange(x, f, cf, lambstar):
    return f(x) - cf(x).dot(lambstar)

def grad_lagrange(x, df, dcf, lambstar):
    return df(x) - (dcf(x).T).dot(lambstar)

def compute_lagrange(x, cf, mu):
    return mu/cf(x)

def log_barrier(x, f, cf, mu):
    C = cf(x)
    if not (C > 0).all():
        return np.inf
    return f(x) - mu * np.sum(np.log(C))

def grad_log_barrier(x, df, c, dcf, mu):
    return df(x) - mu *np.sum(dcf(x).T/c(x), axis = 1)

def set_barrier(f, df, cf, dcf, mu):
    return lambda x: log_barrier(x, f, cf, mu),\
            lambda x: grad_log_barrier(x, df, cf, dcf, mu)

def KKT(x, f, df, cf, dcf, lambstar, mu, TOL = 1e-2):
    C = cf(x)
    GL = grad_lagrange(x, df, dcf, lambstar)
    if np.linalg.norm(GL) > TOL:
        print("Grad_Lagrange : {}, norm = {}".format(GL, np.linalg.norm(GL)))
        return False
    if C.any() < 0:
        print("Not feasible")
        return False
    if lambstar.any() < 0:
        print("Lagrange: {}".format(lambstar))
        return False
    if mu > TOL:
        print("Prod not equal")
        return False
    return True

def barrier(f, df, x0, cf, dcf, TOL = 1e-4):
    R = []
    mu = 1
    primal_TOL = 1e-4
    it_stop = 15

    assert Util.is_feasible(cf, x0)

    it = 0
    x = x0
    while it < it_stop:
        it += 1
        print("Iter: {}".format(it))
        P, dP = set_barrier(f, df, cf, dcf, mu)
        x, it1, f1  = alg.bfgs_constrained(P, dP, x, cf, primal_TOL)


        lagmult = compute_lagrange(x, cf, mu)
        
        if KKT(x, P, dP, cf, dcf, lagmult, mu, TOL):
            return x, it, f1, R
            
        """
        if mu < TOL:
            break
        """

        mu *= 0.3
        R.append(np.linalg.norm(grad_lagrange(x, dP, dcf, lagmult)))
        

    print("Barrier, couldn't find minimum. f(x) = {}".format(f1))
    return x, it, f1, R





if __name__ == "__main__":
    m, n = 20, 2
    eig =  np.abs(np.array((1, 2))) #+ np.random.randn(2))
    cf, dcf = set_constraints(eig)
    #x = Util.get_random_feasible(constraints, eig)
    x = Util.get_random_non_feasible(constraints, eig)

    print("\nFOUND x\n")

    A, b = Util.from_x_to_matrix(x)
    Z, W = Util.generate_random(m, n, scale = 0.9)
    W = Util.pre_classify(x, Z, W, m2.H)
    f, df = Util.set_function(m2.f, m2.df, Z, W)
    P, dP = set_barrier(f, df, cf, dcf, 1)

    # Visualize goal
    fig = plt.figure()
    ax = fig.add_subplot(111)
    Util.visualize(ax, A, b, Z, W, m2.H)

    # Reset x
    x0 = Util.get_feasible(eig*np.array((1.2, 0.8)))
    A0, b0 = Util.from_x_to_matrix(x0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    Util.visualize(ax, A0, b0, Z, W, m2.H)

    # Find and visualize x anew
    #x1, it1, f1, R = alg.bfgs(f, df, x0)
    #x1, it1, f1, R = alg.bfgs_constrained(f, df, x0, cf, backtrack= False)
    x1, it1, f1, R = barrier(P, dP, x0, cf, dcf)

    A1, b1 = Util.from_x_to_matrix(x1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    Util.visualize(ax, A1, b1, Z, W, m2.H)

    plt.figure()
    plt.plot(R)
    plt.semilogy()
    plt.show()
    print("Resulting eigenvals: {}".format(np.linalg.eigvals(A1)))
    print("Constrained eigenvals: {}".format(eig))
    print(Util.is_feasible(cf, x1))
    print(cf(x1))