# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import scipy
import scipy.ndimage
import math
import time
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

'''
#https://gist.github.com/bombless/4286560
def RGB2YUV(input):
  (R, G, B) = input
	Y = int(0.299 * R + 0.587 * G + 0.114 * B)
	U = int(-0.147 * R + -0.289 * G + 0.436 * B)
	V = int(0.615 * R + -0.515 * G + -0.100 * B)
	
	return (Y, U, V)

def YUV2RGB(input):
	(Y, U, V) = input
	R = int(Y + 1.14 * V)
	G = int(Y - 0.39 * U - 0.58 * V)
	B = int(Y + 2.03 * U)
	
	return (R, G, B)

'''
import scipy.fftpack
from math import cos
def rgb2yuv(pixel):
    R, G, B = pixel
    Y = int(0.299 * R + 0.587 * G + 0.114 * B + 0)
    Cb = int(-0.169 * R + -0.334 * G + 0.500 * B + 128)
    Cr = int(0.500 * R + -0.419 * G + -0.081 * B + 128)
    return np.array([Y, Cb, Cr])
# https://www.chegg.com/homework-help/questions-and-answers/6-consider-yuv-rgb-color-format-conversion-1402-gl-l-1-034414-071414-u-128-v-128-11772-req-q37231191
def yuv2rgb(pixel):
    Y, U, V = pixel
    R = int(1 * Y + 0 * (U-128) + 1.402 * (V-128))
    G = int(1 * Y + -0.34414 * (U-128) -0.71414 * (V - 128))
    B = int(1 * Y + 1.772 * (U - 128) + 0 * (V -128))
    return np.array([R, G, B])

def dct(F, yuv, N):
    def C(k):
        if k == 0:
            return 1/math.sqrt(2)
        return 1
    for u in range(0, N):
        for v in range(0, N):
            '''
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
            '''
            # versión eficiente. Dejamos comentamos nuestro código porque lo entendeos mejor
            F[u,v] = scipy.fftpack.dct(yuv[u,v], norm='ortho')
    return F

def quant(F, N):
    FQ = np.zeros((N, N, 3))
    for u in range(0, N):
        for v in range(0, N):
            FQ[u,v][0] = np.round(F[u,v,0]/Q_Luminance[u,v])*Q_Luminance[u,v]
            FQ[u, v][1] = np.round(F[u,v,1] / Q_Chrominance[u, v]) * Q_Chrominance[u, v]
            FQ[u, v][2] = np.round(F[u,v,2] / Q_Chrominance[u, v]) * Q_Chrominance[u, v]
    return FQ

def dct_bloque(p):
    Q = Q_matrix()
    N, _, channels = p.shape
    # 1: color. rgb -> yuv
    yuv = np.array(list(map(lambda row: [*map(lambda pixel: rgb2yuv(pixel), row)], p)))
    # 2: DCT
    F = np.zeros(p.shape)
    F = dct(F, yuv, N)
    # 3: quantization
    FQ = quant(F, N)
    return F
#dct_bloque(np.zeros((8,8)))
#exit()
# https://arxiv.org/pdf/1405.6147.pdf

def idct(F, yuv, N):
    def C(k):
        if k == 0:
            return 1/math.sqrt(2)
        return 1
    F_inv = np.zeros(F.shape)
    for u in range(0, N):
        for v in range(0, N):
            F_inv[u, v] = scipy.fftpack.idct(yuv[u, v], norm='ortho')
    return F_inv
def idct_bloque(p):
    N, _, channels = p.shape
    # 1: color. rgb -> yuv
    #rgb = np.array(list(map(lambda row: [*map(lambda pixel: yuv2rgb(pixel), row)], p)))
    # 2: DCT
    F = np.zeros(p.shape)
    F = idct(F, p, N)
    rgb = np.array(list(map(lambda row: [*map(lambda pixel: yuv2rgb(pixel), row)], F)))
    # 3: quantization
    #FQ = quant(F, N)
    return rgb


"""
Reproducir los bloques base de la transformación para los casos N=4,8
Ver imágenes adjuntas.
"""


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
def dividir(array, n_rows_blocks=8, n_cols_blocks=8):
    r, c, chanels = array.shape
    b = (array.reshape(c // n_rows_blocks, n_rows_blocks, -1, n_cols_blocks).swapaxes(1, 2).reshape(-1, n_rows_blocks, n_cols_blocks).swapaxes(1, 2))
    return b
def jpeg_gris(imagen_gray):
    bloques = dividir(imagen_gray)
    for bloque in bloques:
        bloque_trans = dct_bloque(bloque)
    pass


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
def dividir_3d(array, n_rows_blocks=8, n_cols_blocks=8):
    r, c, channels = array.shape
    new_array = array.reshape((r*c//(n_rows_blocks*n_cols_blocks), n_rows_blocks, n_cols_blocks, channels))
    return new_array

def reconstruir_3d(bloques):
    a, b, c, d = bloques.shape
    return bloques.reshape(a//b, a//c, d)
    n_bloques = a
    new_im = np.concatenate(bloques[0:b*c])
    #print(new_im.shape)
    #return
    i = n_bloques
    while i < len(bloques):
        new_im = np.concatenate((new_im, np.concatenate(bloques[i:i+n_bloques])), axis=3)
        i += n_bloques
    return new_im

def aplicar_jpeg_color(imagen_color):
    print(imagen_color.shape)
    bloques = dividir_3d(imagen_color)
    print(bloques.shape)
    #exit()
    bloques_trans = np.zeros(bloques.shape)
    for index, bloque in enumerate(bloques):
        bloque_trans = dct_bloque(bloque)
        bloques_trans[index] = bloque_trans
        print(index + 1, 'of', len(bloques))
    x = reconstruir_3d(bloques_trans)
    imagen_color = reconstruir_3d(bloques_trans)
    print(imagen_color.shape)
    #exit()
    return imagen_color
    # print(bloques.shape)

def deaplicar_jpeg_color(comprimida):
    bloques = dividir_3d(comprimida)
    bloques_trans = np.zeros(bloques.shape)
    for index, bloque in enumerate(bloques):
        bloque_trans = idct_bloque(bloque)
        bloques_trans[index] = bloque_trans
        print(index + 1, 'of', len(bloques))
    x = reconstruir_3d(bloques_trans)
    imagen_color = reconstruir_3d(bloques_trans)
    return imagen_color
def jpeg_color(imagen_color):
    comprimida = aplicar_jpeg_color(imagen_color)
    plt.figure()
    plt.imshow(comprimida)
    plt.show()
    descomprimida = deaplicar_jpeg_color(comprimida)
    plt.figure()
    plt.imshow(descomprimida.astype(np.uint8))
    plt.show()
    return comprimida

"""
#--------------------------------------------------------------------------
Imagen de GRISES

#--------------------------------------------------------------------------
"""


### .astype es para que lo lea como enteros de 32 bits, si no se
### pone lo lee como entero positivo sin signo de 8 bits uint8 y por ejemplo al 
### restar 128 puede devolver un valor positivo mayor que 128
'''
mandril_gray=scipy.ndimage.imread('mandril_gray.png').astype(np.int32)

start= time.clock()
mandril_jpeg=jpeg_gris(mandril_gray)
end= time.clock()
print("tiempo",(end-start))
'''

"""
#--------------------------------------------------------------------------
Imagen COLOR
#--------------------------------------------------------------------------
"""
## Aplico.astype pero después lo convertiré a 
## uint8 para dibujar y a int64 para calcular el error

mandril_color=scipy.misc.imread('../standard_test_images/mandril_color.png').astype(np.int32)



#start= time.clock()
mandril_jpeg=jpeg_color(mandril_color)
#end= time.clock()
#print("tiempo",(end-start))
     
       









