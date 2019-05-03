# -*- coding: utf-8 -*-
#  Jordi Armengol, Bruno Tamborero
"""

"""

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.cluster.vq import vq, kmeans

#%%
# Nuestra versión de ScyPy ya no tiene imread: https://stackoverflow.com/questions/15345790/scipy-misc-module-has-no-attribute-imread
#imagen=scipy.misc.imread('../standard_test_images/house.png')
import imageio
imagen=imageio.imread('../standard_test_images/house.png')


(n,m)=imagen.shape # filas y columnas de la imagen
plt.figure()    
plt.imshow(imagen, cmap=plt.cm.gray)
plt.show()
 

#%%
   
"""
Usando K-means http://docs.scipy.org/doc/scipy/reference/cluster.vq.html
crear un diccionario cuyas palabras sean bloques 8x8 con 512 entradas 
para la imagen house.png.

Dibujar el resultado de codificar house.png con dicho diccionario.

Calcular el error, la ratio de compresión y el número de bits por píxel
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


def cuantizar(imagen, n_clusters):
    bloques = dividir(imagen)
    a, b, c = bloques.shape
    bloques = bloques.reshape(a, b*c).astype(float)
    codebook, distortion = kmeans(bloques, n_clusters)
    codebook = codebook.astype(int)
    code, dist = vq(bloques, codebook)
    return code, codebook


n_entradas = 512
shape_bloques = (8, 8)
code, codebook = cuantizar(imagen, n_entradas)
code = np.array([np.reshape(codebook[e], shape_bloques) for e in code])
imagenCuantizada = reconstruir(code)
plt.figure()
plt.imshow(imagenCuantizada, cmap=plt.cm.gray)
plt.show()

n, m = imagen.shape
bitsPorPixelOriginal = 8
tam_dic = n_entradas * shape_bloques[0] * shape_bloques[1] * bitsPorPixelOriginal
n_bloques = len(dividir(imagen))
from math import log2
ratio_com = (n*m*bitsPorPixelOriginal)/(tam_dic + n_bloques*log2(n_entradas))  # log2(n_entradas) -> para cada bloque, bits necesarios para saber a qué entrada corresponde el bloque
Sigma = np.sqrt(sum(sum((imagen - imagenCuantizada) ** 2))) / (n * m)
bitsPorPixel = (tam_dic + n_bloques*log2(n_entradas))/(n*m)
print('House:', 'ratio compresión =', ratio_com, 'Sigma =', Sigma, 'bitsPorPixel', bitsPorPixel)

"""
Hacer lo mismo con la imagen cameraman.png

https://atenea.upc.edu/mod/folder/view.php?id=1833385
http://www.imageprocessingplace.com/downloads_V3/root_downloads/image_databases/standard_test_images.zip
"""
#imagen=scipy.misc.imread('../standard_test_images/cameraman.png')
cameraman = imageio.imread('../standard_test_images/cameraman.png')
n_entradas = 512
shape_bloques = (8, 8)
code_cameraman, codebook_cameraman = cuantizar(cameraman, n_entradas)
code_cameraman = np.array([np.reshape(codebook_cameraman[e], shape_bloques) for e in code_cameraman])
imagenCuantizadaCameraman = reconstruir(code_cameraman)
plt.figure()
plt.imshow(imagenCuantizadaCameraman, cmap=plt.cm.gray)
plt.show()

n, m = cameraman.shape
bitsPorPixelOriginal = 8
tam_dic = n_entradas * shape_bloques[0] * shape_bloques[1] * bitsPorPixelOriginal
n_bloques = len(dividir(cameraman))
from math import log2
ratio_com = (n*m*bitsPorPixelOriginal)/(tam_dic + n_bloques*log2(n_entradas))
bitsPorPixel = (tam_dic + n_bloques*log2(n_entradas))/(n*m)
Sigma = np.sqrt(sum(sum((cameraman - imagenCuantizadaCameraman) ** 2))) / (n * m)
print('Cameraman:', 'ratio compresión =', ratio_com, 'Sigma =', Sigma, 'bitsPorPixel', bitsPorPixel)

"""
Dibujar el resultado de codificar cameraman.png con el diccionarios obtenido
con la imagen house.png

Calcular el error.
"""

n_entradas = 512
shape_bloques = (8, 8)
code_house, codebook_house = cuantizar(imagen, n_entradas)
code_house = np.array([np.reshape(codebook_cameraman[e], shape_bloques) for e in code_house])
imagenCuantizadaCameraman = reconstruir(code_house)
plt.figure()
plt.imshow(imagenCuantizadaCameraman, cmap=plt.cm.gray)
plt.show()

ratio_com = (n*m*8)/(512*8*8)
Sigma = np.sqrt(sum(sum((cameraman - imagenCuantizadaCameraman) ** 2))) / (n * m)
print('Cameraman con el diccionario de House:', 'Sigma =', Sigma)

