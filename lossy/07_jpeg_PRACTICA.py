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

def rgb2yuv(pixel):
    R, G, B = pixel
    Y = int(0.299 * R + 0.587 * G + 0.114 * B + 0)
	Cb = int(-0.169 * R + -0.334 * G + 0.500 * B + 128)
	Cr = int(0.500 * R + -0.419 * G + -0.081 * B + 128)
    return (Y, Cb, Cr)

def dct(F):
    def C(k):
        if k == 0:
            return 1/math.sqrt(2)
        return 1
    for u in range(0, N):
        for v in range(0, N):
            S = 0
            for x in range(0, N):
                s = 0
                for y in range(0, N):
                    s += yuv[x,y]*math.cos((pi*(2*x + 1)*u)/(2*N))*cos((pi*(2*y+1)*v)/(2*N))
                S += s
            F[u,v] = (2/N) * C(u) * C(v) * S
    return F

def quant(F):
    Q = Q_matrix()
    N, _ = F.shape
    F = np.zeros(p.shape)
    for u in range(0, N):
        for v in range(0, N):
            F[u,v] = round(F(u,v)/)

def dct_bloque(p):
    Q = Q_matrix()
    N, _ = p.shape
    # 1: color. rgb -> yuv
    yuv = np.array(list(map(lambda row: map(lambda pixel: rgb2yuv(pixel), row), p)))
    # 2: DCT
   
    F = np.zeros(p.shape)
    F = dct(F)
    # 3: quantization
    F = quant(F)
    return F
            
def idct_bloque(p):
    pass


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

def jpeg_gris(imagen_gray):
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

def jpeg_color(imagen_color):
    pass

"""
#--------------------------------------------------------------------------
Imagen de GRISES

#--------------------------------------------------------------------------
"""


### .astype es para que lo lea como enteros de 32 bits, si no se
### pone lo lee como entero positivo sin signo de 8 bits uint8 y por ejemplo al 
### restar 128 puede devolver un valor positivo mayor que 128

mandril_gray=scipy.ndimage.imread('mandril_gray.png').astype(np.int32)

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

mandril_color=scipy.misc.imread('./mandril_color.png').astype(np.int32)



start= time.clock()
mandril_jpeg=jpeg_color(mandril_color)     
end= time.clock()
print("tiempo",(end-start))
     
       









