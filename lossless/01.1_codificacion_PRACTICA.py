# -*- coding: utf-8 -*-

import random

'''
0. Dada una codificación R, construir un diccionario para codificar m2c y otro para decodificar c2m
'''
R = [('a','0'), ('b','11'), ('c','100'), ('d','1010'), ('e','1011')]

# encoding dictionary
m2c = dict(R)

# decoding dictionary
c2m = dict([(c,m) for m, c in R])


'''
1. Definir una función Encode(M, m2c) que, dado un mensaje M y un diccionario 
de codificación m2c, devuelva el mensaje codificado C.
'''

def Encode(M, m2c):
    C = [m2c[m] for m in M]
    return C
    
    
''' 
2. Definir una función Decode(C, c2m) que, dado un mensaje codificado C y un diccionario 
de decodificación c2m, devuelva el mensaje original M.
'''
def Decode(C,c2m):
    M = [c2m[c] for c in C]
    return M
  

#------------------------------------------------------------------------
# Ejemplo 1
#------------------------------------------------------------------------

R = [('a','0'), ('b','11'), ('c','100'), ('d','1010'), ('e','1011')]

# encoding dictionary
m2c = dict(R)

# decoding dictionary
c2m = dict([(c,m) for m, c in R])







#------------------------------------------------------------------------
# Ejemplo 2
#------------------------------------------------------------------------
R = [('a','0'), ('b','10'), ('c','110'), ('d','1110'), ('e','1111')]

# encoding dictionary
m2c = dict(R)

# decoding dictionary
c2m = dict([(c,m) for m, c in R])

''' 
3.
Codificar y decodificar 20 mensajes aleatorios de longitudes también aleatorias.  
Comprobar si los mensajes decodificados coinciden con los originales.
'''




#------------------------------------------------------------------------
# Ejemplo 3 
#------------------------------------------------------------------------
R = [('a','0'), ('b','01'), ('c','011'), ('d','0111'), ('e','1111')]

# encoding dictionary
m2c = dict(R)

# decoding dictionary
c2m = dict([(c,m) for m, c in R])




alphabet = ['a','b','c','d','e']
random.seed(1234)
for i in range(0,20):
    l = random.randint(1,1000)
    m = random.choices(alphabet,k=l) # python 3.6
    e = Encode(m,m2c)
    d = Decode(e,c2m)
    print('Message',(i+1),':',m == d)



''' 
4. Codificar y decodificar los mensajes  'ae' y 'be'. 
Comprobar si los mensajes decodificados coinciden con los originales.
'''

m1 = ['a','e']
e1 = Encode(m1,m2c)
d1 = Decode(e1,c2m)
print(m1,'decodificado coincide con el original?',m1 == d1)
m2 = ['b','e']
e2 = Encode(m2,m2c)
d2 = Decode(e2,c2m)
print(m2,'decodificado coincide con el original?',m2 == d2)

'''
¿Por qué da error?

(No es necesario volver a implementar Decode(C, m2c) para que no dé error)
'''


# Daría error porque el código no es prefijo y la decodificación es ambigua, pero con nuestra implementación basada en listas
# no da error.




