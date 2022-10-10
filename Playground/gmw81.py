
from this import d
from tkinter import dialog
import numpy as np
from sympy import I


def gmw81_p(mat, p):
    n = mat.shape[0]
    arr = np.empty(n)

    #r = 0.0 * mat

    gamma = 0.0
    xi = 0.0
    beta = 0.0
    for i in range(n):
        gamma = max(abs(mat[i,i]), gamma)
        for j in range(i+1, n):
            xi = max(abs(mat[i,j]), xi)

    delta = 4e-7 * max(gamma + xi, 1)
    if n == 1:
        mat[0,0] = max(mat[0,0], 4e-7 * max(abs(mat[0,0]),1))
        return
    else:
        beta = np.sqrt(max(gamma, xi / np.sqrt(n**2 -1.0), 4e-7))

    for j in range(n):
        q = j
        for i in range(j+1,n):
            if abs(mat[i,i]) >= abs(mat[q,q]):
                q = i
        
        if q != j:
            index = p[q]
            p[q] = p[j]
            p[j] = index

            for i in range(n):
                temp = mat[q,i]
                mat[q,i] = mat[j,i]
                mat[j,i] = temp

            for i in range(n):
                temp = mat[i,q]
                mat[i,q] = mat[i,j]
                mat[i,j] = temp

        theta_j = 0.0
        if j < n-1:
            for i in range(j+1, n):
                theta_j = max(theta_j, abs(mat[j,i]))
        
        dj = max(abs(mat[j,j]), (theta_j/beta)**2, delta)

        mat[j,j] = dj
        for i in range(i+1, n):
            arr[i] = mat[i,j]
            mat[j,i] /= dj

        for i in range(i+1, n):
            for k in range(i+1,n):
                mat[k,i] -= arr[i] * mat[k,j]
                #mat[i,k] = mat[k,i]


    #for i in range(n):
    #    for j in range(i):
    #        mat[j,i] = 0

def to_permute(p):
    P = np.eye(p.shape[0])
    O = np.eye(p.shape[0])
    for i in range(p.shape[0]):
        O[i,:] = P[p[i],:]
    return O

n = 4
A = np.random.rand(n,n)
A = A.transpose() @ A + np.random.rand(1) * np.eye(n)

M = A.copy()

p = np.arange(4)

gmw81_p(M,p)
P = to_permute(p)

L = 0.0 * M
D = 0.0 * M

for i in range(n):
    D[i,i] = M[i,i]

for i in range(n):
    for j in range(i):
        L[i,j] = M[i,j]
for i in range(n):
    L[i,i] = 1.0

print(A)
print(L)
print(D)

print(L @ D @ L.transpose())

v = np.random.rand(n, 1)
print(p)
print(v)
print(P @ v)
print(P.transpose() @ v)

