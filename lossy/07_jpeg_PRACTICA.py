# -*- coding: utf-8 -*-
"""

Jordi Armengol, Bruno Tamborero

"""

import numpy as np
import scipy
import scipy.ndimage
import math 
pi=math.pi




import matplotlib.pyplot as plt




        
"""
Matrices de cuantización, estándares y otras
"""

    
Q_Luminance=np.array([
[16 ,11, 10, 16,  24,  40,  51,  61],
[12, 12, 14, 19,  26,  58,  60,  55],
[14, 13, 16, 24,  40,  57,  69,  56],
[14, 17, 22, 29,  51,  87,  80,  62],
[18, 22, 37, 56,  68, 109, 103,  77],
[24, 35, 55, 64,  81, 104, 113,  92],
[49, 64, 78, 87, 103, 121, 120, 101],
[72, 92, 95, 98, 112, 100, 103, 99]])

Q_Chrominance=np.array([
[17, 18, 24, 47, 99, 99, 99, 99],
[18, 21, 26, 66, 99, 99, 99, 99],
[24, 26, 56, 99, 99, 99, 99, 99],
[47, 66, 99, 99, 99, 99, 99, 99],
[99, 99, 99, 99, 99, 99, 99, 99],
[99, 99, 99, 99, 99, 99, 99, 99],
[99, 99, 99, 99, 99, 99, 99, 99],
[99, 99, 99, 99, 99, 99, 99, 99]])

def Q_matrix(r=1):
    m=np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            m[i,j]=(1+i+j)*r
    return m

"""
Implementar la DCT (Discrete Cosine Transform) 
y su inversa para bloques NxN

dct_bloque(p,N)
idct_bloque(p,N)

p bloque NxN

"""


''''
def dct(F, yuv, N):
    def C(k):
        if k == 0:
            return 1 / math.sqrt(2)
        return 1

    for u in range(0, N):
        for v in range(0, N):
            S_y = 0
            S_cb = 0
            S_cr = 0
            for x in range(0, N):
                s_y = 0
                s_cb = 0
                s_cr = 0
                for y in range(0, N):
                    s_y += yuv[x, y][0]*math.cos((pi*(2*x + 1)*u)/(2*N))*cos((pi*(2*y+1)*v)/(2*N))
                    s_cb += yuv[x, y][1] * math.cos((pi * (2 * x + 1) * u) / (2 * N)) * cos(
                        (pi * (2 * y + 1) * v) / (2 * N))
                    s_cr += yuv[x, y][2] * math.cos((pi * (2 * x + 1) * u) / (2 * N)) * cos(
                        (pi * (2 * y + 1) * v) / (2 * N))
                S_y += s_y
                S_cb += s_cb
                S_cr += s_cr
            F[u, v][0] = (2/N) * C(u) * C(v) * S_y
            F[u, v][1] = (2 / N) * C(u) * C(v) * S_cb
            F[u, v][2] = (2 / N) * C(u) * C(v) * S_cr
    return F
'''

from scipy.fftpack import dct, idct # más eficiente

def dct_bloque(p):
    # 2 llamadas porque es DCT 2D
    aux = dct(p, norm='ortho', axis=0)
    return dct(aux, norm='ortho', axis=1)

def idct_bloque(p):
    aux = idct(p, norm='ortho', axis=0)
    return idct(aux, norm='ortho', axis=1)

"""
Reproducir los bloques base de la transformación para los casos N=4,8
Ver imágenes adjuntas.
"""

# https://users.cs.cf.ac.uk/Dave.Marshall/Multimedia/PDF/10_DCT.pdf
def reproducir_bloques():
    for N in [4, 8]:
        # para generar los bloques base:
        # M = DCT a una matriz diagonal (y transponerla)
        # para cada elemento (i,j), aplicamos tensordot
        # de la respectiva fila
        M = scipy.fftpack.dct(np.diag([1]*N), norm='ortho').T
        plt.axis('off')
        fig = plt.figure(figsize=(N*N+N+1, N*N+N+1))
        for i in range(0, N):
            for j in range(0, N):
                # M = np.zeros((N, N))
                # M[i, j] = 1
                # M_transform = dct_bloque(M.astype(np.uint8))
                M_transform = np.tensordot(M[i], M[j], axes=0)
                fig.add_subplot(N, N, i*N + j + 1)
                plt.imshow(M_transform, cmap=plt.cm.gray)
                plt.axis('off')
        plt.show()

reproducir_bloques()
"""
Implementar la función jpeg_gris(imagen_gray) que: 
1. dibuje el resultado de aplicar la DCT y la cuantización 
(y sus inversas) a la imagen de grises 'imagen_gray' 

2. haga una estimación de la ratio de compresión
según los coeficientes nulos de la transformación: 
(#coeficientes/#coeficientes no nulos).

3. haga una estimación del error
Sigma=np.sqrt(sum(sum((imagen_gray-imagen_jpeg)**2)))/np.sqrt(sum(sum((imagen_gray)**2)))


"""
# Con Q_Matrix también funcionaría, pero la compresión no sería la misma.

def quant(p):
    if len(p.shape) == 2:
        N, _ = p.shape
        for u in range(N):
            for v in range(N):
                p[u, v] = round(p[u, v]/Q_Luminance[u, v])
        return p
    N, _, channels = p.shape
    for u in range(N):
        for v in range(N):
            p[u, v, 0] = round(p[u, v, 0]/Q_Luminance[u, v])
            p[u, v, 1] = round(p[u, v, 1]/Q_Chrominance[u, v])
            p[u, v, 2] = round(p[u, v, 2]/Q_Chrominance[u, v])
    return p


def dequant(p):
    if len(p.shape) == 2:
        N, _ = p.shape
        for u in range(N):
            for v in range(N):
                p[u, v] = round(p[u, v]*Q_Luminance[u, v])
        return p
    N, _, channels = p.shape
    for u in range(N):
        for v in range(N):
            p[u, v, 0] = round(p[u, v, 0] * Q_Luminance[u, v])
            p[u, v, 1] = round(p[u, v, 1] * Q_Chrominance[u, v])
            p[u, v, 2] = round(p[u, v, 2] * Q_Chrominance[u, v])
    return p


def dividir(array, n_rows_blocks=8, n_cols_blocks=8):
    r, c = array.shape
    b = (array.reshape(c // n_rows_blocks, n_rows_blocks, -1, n_cols_blocks).swapaxes(1, 2).reshape(-1, n_rows_blocks,
                                                                                                    n_cols_blocks).swapaxes(
        1, 2))
    return b


def reconstruir(bloques):
    a, b, c = bloques.shape
    n_bloques = b * c
    new_im = np.concatenate(bloques[0:n_bloques])
    i = n_bloques
    while i < len(bloques):
        new_im = np.concatenate((new_im, np.concatenate(bloques[i:i + n_bloques])), axis=1)
        i += n_bloques
    return np.transpose(new_im)

# https://arxiv.org/pdf/1405.6147.pdf
def aplicar_jpeg_gris(imagen_gray):
    bloques = dividir(imagen_gray)
    bloques_trans = np.zeros(bloques.shape)
    for index, bloque in enumerate(bloques):
        bloque_shifted = bloque - 128
        # DCT
        bloque_trans = dct_bloque(bloque_shifted)
        # Cuantizaciçom
        bloques_quant = quant(bloque_trans)
        bloques_trans[index] = bloques_quant
        if index % 1000 == 0:
            print(index + 1, 'pixel of', len(bloques))
    imagen_gray_jpeg = reconstruir(bloques_trans)
    return imagen_gray_jpeg


def recuperar_jpeg_gris(imagen_gray_jpeg):
    # Deshacemos los pasos anteriores
    bloques = dividir(imagen_gray_jpeg)
    bloques_recuperados = np.zeros(bloques.shape)
    for index, bloque in enumerate(bloques):
        bloque_dequant = dequant(bloque)
        bloque_recuperado = idct_bloque(bloque_dequant) + 128
        bloques_recuperados[index] = bloque_recuperado
        if index % 1000 == 0:
            print(index + 1, 'pixel of', len(bloques))
    imagen_gray = reconstruir(bloques_recuperados)
    return imagen_gray


def jpeg_gris(imagen_gray):
    fila=len(imagen_gray)
    columna=len(imagen_gray[0])
    comprimida = aplicar_jpeg_gris(imagen_gray.copy())
    plt.figure()
    plt.imshow(comprimida, cmap=plt.cm.gray)
    plt.show()
    descomprimida = recuperar_jpeg_gris(comprimida)
    plt.figure()
    plt.imshow(descomprimida, cmap=plt.cm.gray)
    plt.show()
    #haga una estimación de la ratio de compresión#según los coeficientes nulos de la transformación: (#coeficientes/#coeficientes no nulos).
    coeficientesNulos = (comprimida== 0.).sum()
    ratioCompresion = (fila*columna) / ((fila*columna) - coeficientesNulos)
    #Haga una estimación del error para cada una de las componentes RGB
    print('Ratio Compresión = '+ str(ratioCompresion))
    Sigma=np.sqrt(sum(sum((imagen_gray-comprimida)**2)))/np.sqrt(sum(sum((imagen_gray)**2)))
    print("Estimación del error imagen gris:",Sigma)





"""
Implementar la función jpeg_color(imagen_color) que: 
1. dibuje el resultado de aplicar la DCT y la cuantización 
(y sus inversas) a la imagen RGB 'imagen_color' 

2. haga una estimación de la ratio de compresión
según los coeficientes nulos de la transformación: 
(#coeficientes/#coeficientes no nulos).

3. haga una estimación del error para cada una de las componentes RGB
Sigma=np.sqrt(sum(sum((imagen_color-imagen_jpeg)**2)))/np.sqrt(sum(sum((imagen_color)**2)))

"""

def rgb2yuv(pixel):
    R, G, B = pixel
    Y = int(0.299 * R + 0.587 * G + 0.114 * B + 0)
    Cb = int(-0.169 * R + -0.334 * G + 0.500 * B + 128)
    Cr = int(0.500 * R + -0.419 * G + -0.081 * B + 128)
    return np.array([Y, Cb, Cr])

def yuv2rgb(pixel):
    Y, U, V = pixel
    R = int(1 * Y + 0 * (U - 128) + 1.402 * (V - 128))
    G = int(1 * Y + -0.34414 * (U - 128) - 0.71414 * (V - 128))
    B = int(1 * Y + 1.772 * (U - 128) + 0 * (V - 128))
    return np.array([R, G, B])

def dividir_3d(array, n_rows_blocks=8, n_cols_blocks=8):
    r, c, channels = array.shape
    new_array = array.reshape((r * c // (n_rows_blocks * n_cols_blocks), n_rows_blocks, n_cols_blocks, channels))
    return new_array


def reconstruir_3d(bloques):
    a, b, c, d = bloques.shape
    return bloques.reshape(a // b, a // c, d)

def aplicar_jpeg_color(imagen_color):
    bloques = dividir_3d(imagen_color)
    bloques_trans = np.zeros(bloques.shape)
    for index, bloque in enumerate(bloques):
        # 1: YUV -> RGB
        bloque_yuv = np.array(list(map(lambda row: [*map(lambda pixel: rgb2yuv(pixel), row)], bloque)))
        bloque_trans = np.zeros((bloque_yuv.shape))
        # 2: DCT componente a componente (Y, Cb, Cr)
        bloque_trans[:, :, 0] = dct_bloque(bloque_yuv[:, :, 0])
        bloque_trans[:, :, 1] = dct_bloque(bloque_yuv[:, :, 1])
        bloque_trans[:, :, 2] = dct_bloque(bloque_yuv[:, :, 2])
        # 3: Cuantización
        bloque_quant = quant(bloque_trans)
        bloques_trans[index] = bloque_quant
        if index % 1000 == 0:
            print(index + 1, 'pixel of', len(bloques))
    imagen_color_jpeg = reconstruir_3d(bloques_trans)
    return imagen_color_jpeg


def recuperar_jpeg_color(imagen_color_jpeg):
    # Deshacemos los pasos anteriores
    bloques = dividir_3d(imagen_color_jpeg)
    bloques_recuperados = np.zeros(bloques.shape)
    for index, bloque in enumerate(bloques):
        bloque_dequant = dequant(bloque)
        bloque_recuperado = np.zeros((bloque_dequant.shape))
        bloque_recuperado[:, :, 0] = idct_bloque(bloque_dequant[:, :, 0])
        bloque_recuperado[:, :, 1] = idct_bloque(bloque_dequant[:, :, 1])
        bloque_recuperado[:, :, 2] = idct_bloque(bloque_dequant[:, :, 2])
        bloque_rgb = np.array(list(map(lambda row: [*map(lambda pixel: yuv2rgb(pixel), row)], bloque_recuperado)))
        bloques_recuperados[index] = bloque_rgb
        if index % 1000 == 0:
            print(index + 1, 'pixel of', len(bloques))
    imagen_color = reconstruir_3d(bloques_recuperados)
    return imagen_color


def jpeg_color(imagen_color):
    fila = len(imagen_color)
    columna = len(imagen_color[0])
    comprimida = aplicar_jpeg_color(imagen_color)
    plt.figure()
    plt.imshow(comprimida)
    plt.show()
    descomprimida = recuperar_jpeg_color(comprimida)
    plt.figure()
    plt.imshow(descomprimida.astype(np.uint8))
    plt.show()

    # haga una estimación de la ratio de compresión#según los coeficientes nulos de la transformación: (#coeficientes/#coeficientes no nulos).
    coeficientesNulos = (comprimida == 0.0).sum()
    ratioCompresion = (fila * columna * 3) / ((fila * columna * 3) - coeficientesNulos)
    # Haga una estimación del error para cada una de las componentes RGB
    print('Ratio Compresión = ' + str(ratioCompresion))
    imagen_jpeg = comprimida.astype(np.int64)
    sigmaR = np.sqrt(sum(sum((imagen_color[:, :, 0] - imagen_jpeg[:, :, 0]) ** 2))) / np.sqrt(
        sum(sum((imagen_color[:, :, 0]) ** 2)))
    sigmaG = np.sqrt(sum(sum((imagen_color[:, :, 1] - imagen_jpeg[:, :, 1]) ** 2))) / np.sqrt(
        sum(sum((imagen_color[:, :, 1]) ** 2)))
    sigmaB = np.sqrt(sum(sum((imagen_color[:, :, 2] - imagen_jpeg[:, :, 2]) ** 2))) / np.sqrt(
        sum(sum((imagen_color[:, :, 2]) ** 2)))
    print("Estimación del error para cada una de las componentes RGB:", sigmaR, sigmaG, sigmaB)
    return comprimida

"""
#--------------------------------------------------------------------------
Imagen de GRISES

#--------------------------------------------------------------------------
"""


### .astype es para que lo lea como enteros de 32 bits, si no se
### pone lo lee como entero positivo sin signo de 8 bits uint8 y por ejemplo al 
### restar 128 puede devolver un valor positivo mayor que 128
mandril_gray=scipy.ndimage.imread('../standard_test_images/mandril_gray.png').astype(np.int32)
import time
start= time.clock()
mandril_jpeg=jpeg_gris(mandril_gray)
end= time.clock()
print("tiempo",(end-start))

"""
#--------------------------------------------------------------------------
Imagen COLOR
#--------------------------------------------------------------------------
"""
## Aplico.astype pero después lo convertiré a 
## uint8 para dibujar y a int64 para calcular el error

mandril_color=scipy.misc.imread('../standard_test_images/mandril_color.png').astype(np.int32)



start= time.clock()
mandril_jpeg=jpeg_color(mandril_color)     
end= time.clock()
print("tiempo",(end-start))
     
       









