# -*- coding: utf-8 -*-
# Jordi Armengol, Bruno Tamborero


########################################################

import numpy as np
import matplotlib.pyplot as plt


"""
Implementar una funcion H_WH(N) que devuelva la matriz NxN asociada a la transformación de Walsh-Hadamard

H_WH(4)=
      [[ 0.5,  0.5,  0.5,  0.5],
       [ 0.5,  0.5, -0.5, -0.5],
       [ 0.5, -0.5, -0.5,  0.5],
       [ 0.5, -0.5,  0.5, -0.5]]
"""
from math import sqrt

'''
def H_WH(N):
    if N == 0:
        return 1
    if N == 1:
        return (1/sqrt(2)) * np.array([[1, 1], [1, -1]])
    return np.kron(H_WH(1), H_WH(N-1))
'''



from math import log2
def H_WH(N):
    def h(k):
        if k == 0:
            return np.array([1])
        if k == 1:
            return np.array([[1, 1], [1, -1]])
        return np.kron(h(1), h(k - 1))
    def arrange(M):
        def count_sign_changes(r):
            current_sign = 1
            counter = 0
            for x in r:
                if x != current_sign:
                    counter += 1
                    current_sign = x
            return counter
        R = np.zeros(M.shape)
        for row in M:
            i = count_sign_changes(row)
            R[i] = row
        return R
    H = h(log2(N))
    W = arrange(H) * 1/sqrt(N)
    return W


print('H_WH(4)=\n', H_WH(4))


"""
Implementar la DWHT (Discrete Walsh-Hadamard Transform) y su inversa
para bloques NxN

dwht_bloque(p) 
idwht_bloque(p) 

p bloque NxN

dwht_bloque(
            [[217,   8, 248, 199],
             [215, 189, 242,  10],
             [200,  65, 191,  92],
             [174, 239, 237, 118]]
            )=
            [[ 661,   -7.5, -48.5, 201],
             [   3,  -27.5,  25.5,  57],
             [  59,  -74.5,  36.5, -45],
             [ -51, -112.5, 146.5,  45]]

"""
def dwht_bloque(p):
    p = np.array(p)
    N, _ = p.shape
    H = H_WH(N)
    H_T = np.transpose(H)  # simétrica, no sería necesario
    return np.tensordot(np.tensordot(H, p, axes=1), H_T, axes=1)

print('''dwht_bloque(\n
[[217,   8, 248, 199],
[215, 189, 242,  10],
[200,  65, 191,  92],
[174, 239, 237, 118]]
\n)=\n''', dwht_bloque(
[[217,   8, 248, 199],
[215, 189, 242,  10],
[200,  65, 191,  92],
[174, 239, 237, 118]]
))


def idwht_bloque(p):
    # de hecho, la inversa es ella misma
    p = np.array(p)
    N, _ = p.shape
    H_inv = np.linalg.inv(H_WH(N))  # realmente la inversa es ella misma
    H_inv_T = np.transpose(H_inv)  # simétrica, no sería necesario
    return np.tensordot(np.tensordot(H_inv, p, axes=1), H_inv_T, axes=1)

print('''idwht_bloque(dwht_bloque(\n
[[217,   8, 248, 199],
[215, 189, 242,  10],
[200,  65, 191,  92],
[174, 239, 237, 118]]
\n))=\n''', idwht_bloque(dwht_bloque(
[[217,   8, 248, 199],
[215, 189, 242,  10],
[200,  65, 191,  92],
[174, 239, 237, 118]]
)))

"""
Reproducir los bloques base de la transformación para los casos N=4,8,16
Ver imágenes adjuntas
"""

from matplotlib.patches import Rectangle


def reproducir_bloques():
    for N in [4, 8, 16]:
        plt.axis('off')
        fig = plt.figure(figsize=(N*N+N+1, N*N+N+1))
        for i in range(0, N):
            for j in range(0, N):
                M = np.zeros((N, N))
                M[i, j] = 1
                M_transform = idwht_bloque(M)
                fig.add_subplot(N, N, i*N + j + 1)
                plt.imshow(M_transform, cmap=plt.cm.get_cmap('bwr'))
                plt.axis('off')
        plt.show()
        # plt.savefig(str(N) + '.png')
        if N == 8:  # para 16 nos peta por memoria
            break

reproducir_bloques()
