# Jordi Armengol, Bruno Tamborero
# -*- coding: utf-8 -*-

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

import scipy.ndimage
from scipy.cluster.vq import vq, kmeans

#%%
imagen = misc.ascent()#Leo la imagen
(n,m)=imagen.shape # filas y columnas de la imagen
#plt.imshow(imagen, cmap=plt.cm.gray)
#plt.xticks([])
#plt.yticks([])
#plt.show()
        
"""
Mostrar la imagen habiendo cuantizado los valores de los píxeles en
2**k niveles, k=1..8

Para cada cuantización dar la ratio de compresión y Sigma

Sigma=np.sqrt(sum(sum((imagenOriginal-imagenCuantizada)**2)))/(n*m)

"""

'''
def cuantizar(im, k):
    new_im = im.copy()
    for index_row, row in enumerate(new_im):
        for index_pixel, pixel in enumerate(row):
            bits = '{0:08b}'.format(pixel)
            msb = bits[:k]
            integer = int(msb, 2)
            new_im[index_row][index_pixel] = integer
    return new_im
'''


def nivelar(k):
    i = 0
    niveles = []
    resid = 256 % k
    while i < 256:
        niveles.append(i)
        i += 256//k
    for index, nivel in enumerate(niveles):
        if index == 0:
            continue
        if resid <= 0:
            break
        niveles[index] += 1
        resid -= 1
    if 256 % k == 0:
        niveles.append(255)
    elif 256 % k > 1:
        niveles[-1] = 255
    return niveles


def asignar_nivel(niveles, pixel):
    i = 0
    while i < len(niveles) and pixel >= niveles[i]:
        i += 1
    if pixel == 255:
        i -= 1
    res = niveles[i-1] + (niveles[i]-niveles[i-1])//2
    if i == len(niveles) - 1:
        res += 1
    return res


def cuantizar(im, k):
    niveles = nivelar(k+1)
    new_im = im.copy()
    for index_row, row in enumerate(new_im):
        for index_pixel, pixel in enumerate(row):
            new_im[index_row][index_pixel] = asignar_nivel(niveles, pixel)
    return new_im


for k in range(1, 9):
    imagenCuantizada = cuantizar(imagen, k=k)
    plt.imshow(imagenCuantizada, cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    # plt.savefig(str(k)+'.png')
    imagenOriginal = imagen
    ratio_com = (n*m*8)/(n*m*k)
    Sigma = np.sqrt(sum(sum((imagenOriginal - imagenCuantizada) ** 2))) / (n * m)
    print('k =', k, 'ratio compresión =', ratio_com, 'Sigma =', Sigma)
#%%
"""
Mostrar la imagen cuantizando los valores de los pixeles de cada bloque
n_bloque x n_bloque en 2^k niveles, siendo n_bloque=8 y k=2

Calcular Sigma y la ratio de compresión (para cada bloque 
es necesario guardar 16 bits extra para los valores máximos 
y mínimos del bloque, esto supone 16/n_bloque**2 bits más por pixel).
"""


def dividir(array, n_rows_blocks=8, n_cols_blocks=8):
    r, c = array.shape
    b = (array.reshape(c // n_rows_blocks, n_rows_blocks, -1, n_cols_blocks).swapaxes(1, 2).reshape(-1, n_rows_blocks, n_cols_blocks).swapaxes(1, 2))
    return b


def reconstruir(bloques):
    a, b, c = bloques.shape
    n_bloques = b*c
    new_im = np.concatenate(bloques[0:n_bloques])
    i = n_bloques
    while i < len(bloques):
        new_im = np.concatenate((new_im, np.concatenate(bloques[i:i+n_bloques])), axis=1)
        i += n_bloques
    return np.transpose(new_im)


def cuantizar_minimo_maximo(bloque, k):
    minimo = np.min(bloque)
    maximo = np.max(bloque)
    delta = (maximo - minimo)/(2**k)
    bloque = np.round(((bloque - minimo)//delta + 1/2)*delta + minimo) if delta != 0 else bloque
    return bloque

def cuantizar_bloques(im, n_bloque, k):
    bloques = dividir(im, n_bloque, n_bloque)
    new_bloques = bloques.copy()
    for index, bloque in enumerate(bloques):
        new_bloques[index] = cuantizar_minimo_maximo(bloque, k)
    new_im = reconstruir(new_bloques)
    return new_im

n_bloque = 8
k = 2
imagenCuantizada = cuantizar_bloques(imagen, n_bloque=n_bloque, k=k)
plt.imshow(imagenCuantizada, cmap=plt.cm.gray)
plt.xticks([])
plt.yticks([])
plt.show()
#plt.savefig(str(k)+'.png')
imagenOriginal = imagen
ratio_com = (n*m*8)/(n*m*(16/n_bloque)**2)
Sigma = np.sqrt(sum(sum((imagenOriginal - imagenCuantizada) ** 2))) / (n * m)
print('k =', k, 'ratio compresión =', ratio_com, 'Sigma =', Sigma)

