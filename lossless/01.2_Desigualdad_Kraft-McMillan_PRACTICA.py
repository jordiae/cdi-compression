# -*- coding: utf-8 -*-
"""

"""

'''
Dada la lista L de longitudes de las palabras de un código 
q-ario, decidir si pueden definir un código.

'''

def  kraft1(L, q=2):



'''
Dada la lista L de longitudes de las palabras de un código 
q-ario, calcular el máximo número de palabras de longitud 
máxima, max(L), que se pueden añadir y seguir siendo un código.

'''

def  kraft2(L, q=2):


'''
Dada la lista L de longitudes de las palabras de un  
código q-ario, calcular el máximo número de palabras 
de longitud Ln, que se pueden añadir y seguir siendo 
un código.
'''

def  kraft3(L, Ln, q=2):


'''
Dada la lista L de longitudes de las palabras de un  
código q-ario, hallar un código prefijo con palabras 
con dichas longitudes
'''
def Code(L,q=2):

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
