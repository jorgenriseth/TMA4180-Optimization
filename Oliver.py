import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def makeZ(num, dim):
    z = 1.5 * (2 * np.random.rand(num, dim) - np.ones((dim,)))
    w = np.random.choice([-1, 1], size=num)
    return z, w

def classify_by_ellipse(m,n,area,model, missclasification):
    randomVar = 0
    if missclasification:
        randomVar = np.random.randint(0,1000)/1000
    A = [[1, 0.4], [0.4, 0.8]]  #symmetric, positive definite A
    c = np.random.uniform(-1, 1, n) #random vector
    z = np.zeros((m,n))
    w = np.zeros(n)
    for i in range(m):
        z[i]=np.random.uniform(-area,area,n)
    w = np.zeros(m)
    for i in range(m):
        f_value=eval_func(z[i],A,c,model)
        if f_value>=1:  #if outside the ellipse, the weight should be -1
            if randomVar <= 0.9:
                w[i] = -1
            else:
                w[i] = 1
        else:
            if randomVar <= 0.9:
                w[i] = 1
            else:
                w[i] = -1
    return z,w

def classify_by_rectangle(m,n,area,rec, missclasification):
    z = np.zeros((m, n ))
    w = np.zeros(m)
    randomVar = 0
    if missclasification:
        randomVar = np.random.randint(0,1000)/1000
    for i in range(m):
        z[i] = np.random.uniform(-area, area, n)
        x = z[i][0]
        y = z[i][1]
        if rec[0] < x < rec[1] and rec[2]< y < rec[3]:
            if randomVar <= 0.9:
                w[i] = 1
            else:
                w[i] = -1
        else:
            if randomVar <= 0.9:
                w[i] = -1
            else:
                w[i] =1
    return z,w

def eval_func(zi,A,c,model):
    if model == 1:
        f = np.dot(zi-c,np.dot(A,zi-c))
        return f
    else:
        f = np.dot(zi,np.dot(A,zi))+np.dot(c,zi)
        return f


def construct_A_and_c(x, n):
    A = np.zeros((n, n))
    k = int(n * (n + 1) / 2)
    index = 0
    for i in range(n):
        for j in range(i, n):
            A[i, j] = x[index]
            A[j, i] = x[index]
            index += 1
    c = x[k:]
    return A, c


def residual(z, w, A, c, model):
    if model == 1:
        y = z - c
        #print(np.dot(y, np.dot(A, y)))
        r = (np.dot(y, np.dot(A, y))-1)* w
    elif model == 2:
        r = (np.dot(z, np.dot(A, z)) + np.dot(c, z) - 1) * w
    if r < 0:
        r = 0
    return r


def gradr(zi, w, x, n, model):
    A, c = construct_A_and_c(x, n)
    gradientVector = np.ones((len(x),))
    lengthOfA = int(n * (n + 1) / 2)
    lengthOfT = int(n * (n + 1) / 2 + n)
    k = 0
    l = 0
    if model == 1:
        for i in range(lengthOfA):
            if k == l:
                gradientVector[i] = (zi[k] - c[k]) ** 2
            else:
                gradientVector[i] = 2 * (zi[k] - c[k]) * (zi[l] - c[l])
            l += 1
            if l == n:
                k += 1
                l = k
        c_i = 0
        for i in range(lengthOfA, lengthOfT):
            gradientVector[i] = -2 * (zi - c).T @ A[c_i, :]
            c_i += 1
        return gradientVector

    elif model == 2:
        for i in range(lengthOfA):
            if k == l:
                gradientVector[i] = (zi[k]) ** 2
            else:
                gradientVector[i] = 2 * (zi[k]) * (zi[l])
            i += 1
            l += 1
            if l == n:
                k += 1
                l = k
        c_i = 0
        for i in range(lengthOfA, lengthOfT):
            gradientVector[i] = zi[c_i]
            c_i += 1
        return gradientVector


def jacobi(z, w, x, n, model):
    """Compute jacobi determinant as in chapter 10.2"""
    m = len(z)
    A, c = construct_A_and_c(x, n)
    b = int(n * (n + 1) / 2)
    jac = np.zeros((m, b + n))

    for i in range(m):
        if model == 1:
            inside = (z[i] - c).T @ A @ (z[i] - c) <= 1
        elif model == 2:
            inside = np.dot(z[i], np.dot(A, z[i])) + np.dot(c, z[i]) <= 1
        if w[i] > 0:
            if inside:
                jac[i, :] = np.zeros((b + n,))
            else:
                jac[i, :] = gradr(z[i], w, x, n, model)
        else:
            if inside:
                jac[i, :] = - gradr(z[i], w, x, n, model)
            else:
                jac[i, :] = np.zeros((b + n,))
    return jac


def residuals(z, w, x, n, model):
    res = np.zeros((len(z),))
    A, c = construct_A_and_c(x, n)
    for i in range(len(z)):
        res[i] = residual(z[i], w[i], A, c, model)
    return res


def function(z, w, x, n, model):
    res = residuals(z, w, x, n, model)
    return np.sum(res ** 2) ###SKALAR ?


def gradient(z, w, x, n, model):
    res = residuals(z, w, x, n, model)
    jac = jacobi(z, w, x, n, model)
    grad = 2 * jac.T @ res
    return grad


def setmodelzw(z, w, x, n, model):
    f = lambda x: function(z, w, x, n, model)
    g = lambda x: gradient(z, w, x, n, model)
    return f, g


def test_grad():
    N = 2
    z, w = makeZ(10, 2)
    x = np.random.rand(5)
    f, g = setmodelzw(z, w, x, N, 2)
    eps = [10 ** k for k in range(-1, -12, -1)]
    fi = f(x)
    gi = g(x)
    #
    #    p = np.identity(int(2*(2+1)/2) + 2)
    #    permute = np.random.rand(int(2*(2+1)/2) + 2, int(2*(2+1)/2) + 2)
    #    p = permute@p

    p = np.random.rand((int(N * (N + 1) / 2 + N)))
    print("p = {}".format(p))
    for e in eps:
        print("ep = {:e}, error = {:e}".format(e, gi.dot(p) - (f(x + e * p) - fi) / e))

def backtrackingLineSearch(z,w,x,p,n,model,grad):
    alp = 0.5
    ro = 0.5
    c1 = 0.05
    while function(z, w, x + alp*p, n, model) > function(z, w, x, n, model)+c1*alp*np.dot(grad,p):
        alp = ro*alp
        
    return alp

def steepesDescent(x0,z,w,n, model):
    x = x0
    k = 0
    grad = gradient(z, w, x, n, model)
    p = -grad
    while True:
        a_k = backtrackingLineSearch(z,w,x,p,n,model,grad)
        x += a_k*p
        A,c = construct_A_and_c(x,N)
        grad = gradient(z,w,x,n,model)
        p = -grad
        k+=1
        #print("f(x) =", function(z,w,x,n,model))
        if k%1000 == 0:
            print(k)
        if k == 10000 or np.linalg.norm(grad) < 10**(-3) :
            break
    return x,k



def BFGS(x0,z,w,n, model):
    alpha = 0.01
    x = x0
    grad = gradient(z, w, x, n, model)
    H0 = np.linalg.norm(grad)/alpha *np.identity(n*(n+1)/2+n)
    H = H0
    k = 0
    while np.linalg.norm(grad)>10**(-3):
        p = -np.dot(H,grad)/np.linalg.norm(-np.dot(H,grad))
        a_k = backtrackingLineSearch(z, w, x, p, n, model, grad)
        oldX = x
        x += a_k*p
        oldGrad = grad
        grad = gradient(z,w,x,n,model)
        s = x-oldX
        y = grad-oldGrad
        if np.dot(s,y) == 0:
            H = np.identity(int(n*(n+1)/2+n))
            continue
        rho = 1/np.dot(s,y)
        beta =rho * np.outer(y,s)
        H = np.dot(np.dot((np.identity(int(n*(n+1)/2+n))-beta),H),np.identity(int(n*(n+1)/2+n))-beta)+rho * np.outer(s,s)
        # alpha = np.dot((x - x0), (grad - oldGrad)) - np.linalg.norm(grad - oldGrad) ** 2
        k+=1
        if k%100 == 0:
            print(k)
        if k == 1000:
            break
    return x,k

def plotElipse(x,n ,z, model, figurename):
    A,c = construct_A_and_c(x,n)
    points = 200
    xVector = np.linspace(-3.5,3.5,points)
    yVector = np.linspace(-3.5,3.5,points)
    V = np.ones((points,points))
    X,Y = np.meshgrid(xVector,yVector)

    for i,xval in enumerate(xVector):
        for j,yval in enumerate(yVector):
            z_i = np.array([X[i,j],Y[i,j]])
            if model == 1:
                V[i,j] = np.dot(np.dot(z_i-c,A),z_i-c)
            elif model == 2:
                V[i,j]=np.dot(z_i,np.dot(A,z_i))+np.dot(c,z_i)
    plt.figure(figurename)
    plt.contour(X,Y,V,levels=[1])
    plt.scatter(z[:, 0], z[:, 1], c=w)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(figurename)
    plt.show()



if __name__ == "__main__":
    N = 2
    M = 20
    model =1
    missclasification = True
    # z, w = classify_by_ellipse(M,N,1,model)
    z,w = classify_by_ellipse(M,N,2,model,missclasification)
    x = np.random.rand(5)
    A,c=construct_A_and_c(x,N)

    #test_grad()
    #
    f, g = setmodelzw(z, w, x, N,model)
    a,k =steepesDescent(x,z,w,N,model)
    print(w)
    print(residuals(z,w,a,N,model))
    plotElipse(a, N, z, model, "classify_by_ellipse_steepesDescent_model2")
    print(a)
