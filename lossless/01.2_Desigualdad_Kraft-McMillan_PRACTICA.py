# -*- coding: utf-8 -*-
"""

"""

'''
Dada la lista L de longitudes de las palabras de un código 
q-ario, decidir si pueden definir un código.

'''
from functools import reduce
def  kraft1(L, q=2):
    return reduce(lambda x, y: x+pow(q,-y), L,0) <= 1


'''
Dada la lista L de longitudes de las palabras de un código 
q-ario, calcular el máximo número de palabras de longitud 
máxima, max(L), que se pueden añadir y seguir siendo un código.

'''

def kraft2(L, q=2):
    nwords = 0
    while kraft1(L,q):
        L.append(max(L))
        nwords += 1
    return nwords

'''
Dada la lista L de longitudes de las palabras de un  
código q-ario, calcular el máximo número de palabras 
de longitud Ln, que se pueden añadir y seguir siendo 
un código.
'''

def kraft3(L, Ln, q=2):
	nwords = 0
    while kraft1(L,q):
        L.append(Ln)
        nwords += 1
    return nwords

'''
Dada la lista L de longitudes de las palabras de un  
código q-ario, hallar un código prefijo con palabras 
con dichas longitudes
'''
from collections import Counter
def Code(L,q=2):
    # Huffman canonical code, RFC 1951
    assert kraft1(L,q)
    def Code(L,q=2):
    bl_count = Counter(L)
    code = 0
    bl_count[0] = 0
    L_sorted = sorted(L)
    next_code = {}
    for l in range(0,L_sorted[-1]+1):
        code = (code + bl_count[l-1])*q
        next_code[l] = code
    def_code = []
    lengths = {}
    for l in L_sorted:
        length = l
        def_code.append(next_code[length])
        lengths[next_code[length]] = length
        next_code[length] += 1
    def_code = list(map(lambda x: format(x,'0'+str(lengths[x])+'b'),def_code))
    return def_code   
#%%

'''
Ejemplos
'''
#%%

L=[2,3,3,3,4,4,4,6]
q=2

print("\n",sorted(L),' codigo final:',Code(L,q))
print(kraft1(L,q))
print(kraft2(L,q))
print(kraft3(L,max(L)+1,q))

#%%
q=3
L=[1,3,5,5,3,5,7,2,2,2]
print(sorted(L),' codigo final:',Code(L,q))
print(kraft1(L,q))
