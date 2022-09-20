import numpy as np

def adc_model(b, S0, adc):
    return S0*np.exp(-b*adc)

def mul_transpose_mat_add(mat):
    n = mat.shape[0]
    m = mat.shape[1]

    omat = np.array(m*m)

    for i in range(0, m):
        for j in range(0, i+1):
            entry = 0.0
            for k in range(0, n):
                entry += mat[i*m + k] * mat[j*m + k]
            omat[i*m + j] += entry
            if i != j:
                omat[j*m + i] += entry

            


b = np.array([0, 100, 400, 800])
S0 = 100
adc = 0.002

data = adc_model(b, S0, adc)

print(data)


