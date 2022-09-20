import numpy as np


def mul_transpose_mat_add(mat, n, m):
    omat = np.zeros(m*m)

    for i in range(0, m):
        for j in range(0, i+1):
            entry = 0.0
            for k in range(0, n):
                entry += mat[k*m + i] * mat[k*m + j]
            omat[i*m + j] += entry
            if i != j:
                omat[j*m + i] = entry

    return omat

arr = np.ones(4*3)
for j in range(0,3):
    for i in range(0,4):
        arr[i*3+j] = i*3+j

print(arr)

print(mul_transpose_mat_add(arr, 4, 3))

m1 = np.random.rand(4,3)
for j in range(0,3):
    for i in range(0,4):
        m1[i,j] = i*3+j

print(m1)

print(m1.transpose() @ m1)