import numpy as np

n = 2
m = 4
x_dim = int(((n*(n+1)/2)+n))
x = np.zeros([x_dim])
z = np.array([[1,1,1], [1,1,1], [0,0,1], [0,1,-1]])

A = np.array([[4, 2], [2,7]])
c = np.array([0,0])

#Lage x-vektor fra A og c
row_index = 0
column_index = 0
x_index = 0
for j in range(n):
    while (row_index < n):
        x[x_index]=A[row_index, j]
        x_index = x_index + 1
        row_index = row_index + 1
    column_index = column_index + 1
    row_index = column_index
for i in range(n):
    x[x_index] = c[i]
    x_index = x_index + 1


def from_x_to_Ac(x):
    A = np.zeros([n, n])
    c = np.zeros([n])
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
        c[i]=x[x_index]
        x_index = x_index + 1
    return(A,c)

'''
x = np.array([1,2,3,4,5])
A, c = from_x_to_Ac(x)
print(c)
'''

#Funksjonsverdi
def f(x,z):
    A, c = from_x_to_Ac(x)
    sum = 0
    for i in range(m):
        verdi = np.dot(np.dot(np.transpose((z[i, 0:n] - c)),A), (z[i, 0:n] - c))
        if (z[i][n] == 1):
            sum = sum + (max(verdi-1, 0))**2
        elif (z[i][n] == -1):
            sum = sum + (max(1-verdi, 0))**2
    return(sum)

def dxi(x, z_i):
    A, c = from_x_to_Ac(x)
    diff = np.zeros([x_dim])
    row_index = 0
    column_index = 0
    id = 0
    for i in range(n):
        while row_index<n:
            if row_index == column_index:
                diff[id] = (z_i[row_index]-c[row_index])**2
                id = id +1
                row_index = row_index + 1
            else:
                diff[id] = 2*(z_i[row_index]-c[row_index])*(z_i[column_index]-c[column_index])
                id = id + 1
                row_index = row_index + 1
        column_index = column_index + 1
        row_index = column_index

    for i in range(n):
        diff[id] = (-2)*np.dot((z_i-c),A[i,:])
        id = id + 1

    return(diff)

def df(x,z):
    A, c = from_x_to_Ac(x)
    sum = np.zeros_like(x)
    for i in range(m):
        verdi = np.dot(np.dot(z[i,0:n] - c, A), z[i,0:n] - c)
        if (z[i][n] == 1):
            sum = sum + (max(verdi - 1, 0))*dxi(x,z[i,0:n])
        elif (z[i][n] == -1):
            sum = sum + -(max(1 - verdi, 0))*dxi(x,z[i,0:n])

    return(2*sum)

if __name__=='__main__':
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






