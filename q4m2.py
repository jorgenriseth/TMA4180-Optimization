import numpy as np

n = 2
m = 4
x_dim = int(((n*(n+1)/2)+n))
z = np.array([[1,1,1], [1,1,1], [0,0,1], [0,1,-1]])

def from_x_to_Ab(x):
    A = np.zeros([n, n])
    b = np.zeros([n])
    row_index = 0
    column_index = 0
    x_index = 0
    for j in range(n):
        while (row_index < n):
            A[row_index, j]=x[x_index]
            A[j, row_index] = x[x_index]
            x_index = x_index + 1
            row_index = row_index + 1
        column_index = column_index + 1
        row_index = column_index
    for i in range(n):
        b[i]=x[x_index]
        x_index = x_index + 1
    return(A,b)

def f(x,z):
    A, b = from_x_to_Ab(x)
    sum = 0
    for i in range(m):
        verdi = np.dot(np.dot(np.transpose((z[i, 0:n])),A), (z[i, 0:n])) + np.dot(np.transpose(b),z[i, 0:n])
        if (z[i][n] == 1):
            sum = sum + (max(verdi-1, 0))**2
        elif (z[i][n] == -1):
            sum = sum + (max(1-verdi, 0))**2
    return(sum)

def dxi(x, z_i):
    A, b = from_x_to_Ab(x)
    diff = np.zeros([x_dim])
    row_index = 0
    column_index = 0
    id = 0
    for i in range(n):
        while row_index<n:
            if row_index == column_index:
                diff[id] = (z_i[row_index])**2
                id = id +1
                row_index = row_index + 1
            else:
                diff[id] = 2*(z_i[row_index])*(z_i[column_index])
                id = id + 1
                row_index = row_index + 1
        column_index = column_index + 1
        row_index = column_index

    for i in range(n):
        diff[id] = z_i[i]
        id = id + 1

    return(diff)

def df(x,z):
    A, b = from_x_to_Ab(x)
    sum = np.zeros_like(x)
    for i in range(m):
        verdi = np.dot(np.dot(np.transpose((z[i, 0:n])),A), (z[i, 0:n])) + np.dot(np.transpose(b),z[i, 0:n])
        if (z[i][n] == 1):
            sum = sum + (max(verdi - 1, 0))*dxi(x,z[i,0:n])
        elif (z[i][n] == -1):
            sum = sum + -(max(1 - verdi, 0))*dxi(x,z[i,0:n])

    return(2*sum)

if __name__=='__main__':
    np.random.seed(1)
    N = 5
    # generate random point and direction
    x = np.random.randn(N)
    p = np.random.randn(N)
    f0= f(x,z)
    g = df(x,z).dot(p)
    # compare directional derivative with finite differences
    for ep in 10.0**np.arange(-1,-13,-1):
        g_app = (f(x+ep*p,z)-f0)/ep
        error = abs(g_app-g)/abs(g)
        print('ep = %e, error = %e' % (ep,error))
