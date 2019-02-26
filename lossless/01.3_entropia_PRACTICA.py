# -*- coding: utf-8 -*-
"""

"""
import math
import numpy as np
import matplotlib.pyplot as plt


'''
Dada una lista p, decidir si es una distribución de probabilidad (ddp)
0<=p[i]<=1, sum(p[i])=1.
'''
from functools import reduce
def es_ddp(p,tolerancia=10**(-5)):
    assert tolerancia >= 0
    for prob in p:
        if prob < 0 or prob > 1:
            return False
    return abs(reduce(lambda x, y: x+y,p,0)) - 1 < tolerancia


'''
Dado un código C y una ddp p, hallar la longitud media del código.
'''

def LongitudMedia(C,p):
    assert es_ddp(p)
    return reduce(lambda x,y: x+len(y[0])*y[1],zip(C,p),0)  
'''
Dada una ddp p, hallar su entropía.
'''
def H1(p):
    assert es_ddp(p)
    return -1*reduce(lambda x,y: x+y*(lambda z: 0 if z == 0 else math.log2(z))(y),p,0) 

'''
Dada una lista de frecuencias n, hallar su entropía.
'''
def H2(n):
    s = sum(n)
    p = list(map(lambda x: x/s,n))
    return H1(p)



'''
Ejemplos
'''
C=['001','101','11','0001','000000001','0001','0000000000']
p=[0.5,0.1,0.1,0.1,0.1,0.1,0]
n=[5,2,1,1,1]

print(H1(p))
print(H2(n))
print(LongitudMedia(C,p))



'''
Dibujar H([p,1-p])
'''
p = list(np.arange(0,1,0.01))
inverse_p = list(map(lambda x: 1 - x,p))
pairs = list(zip(p,inverse_p))
hs = list(map(lambda x: H1([x[0],x[1]]), pairs))
plt.plot(hs)
plt.show()



