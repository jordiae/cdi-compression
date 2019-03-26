# -*- coding: utf-8 -*-
"""
@author: martinez
"""

import math
import random




#%%
"""
Dado un mensaje y su alfabeto con sus frecuencias dar el código 
que representa el mensaje utilizando precisión infinita (reescalado)

El intervalo de trabajo será: [0,R), R=2**k, k menor entero tal que R>4T

T: suma total de frecuencias

"""

from itertools import accumulate
from math import log2
def IntegerArithmeticCode(mensaje,alfabeto,frecuencias):
    T = sum(frecuencias)
    R = 1
    while R <= 4*T:
        R *= 2
    indices = dict(zip(alfabeto,sorted(range(len(frecuencias)), key=lambda k: frecuencias[k], reverse = False)))
    print(indices)
    acumuladas = [0] + list(accumulate(frecuencias))
    #acumuladas = list(map(lambda x: x*R,acumuladas))
    acumuladas = list(map(lambda x: int(((R-0)*(x-acumuladas[0]))/(acumuladas[len(acumuladas)-1]-acumuladas[0]) + 0),acumuladas))
    codigo = '0'
    esperados = 1
    for c in mensaje:
        print(acumuladas)
        #R = acumuladas[-1]
        indice = indices[c]
        lower_bound = acumuladas[indice]
        upper_bound = acumuladas[indice+1]
        print(indice,lower_bound,upper_bound)
        if upper_bound < R/2:
            acumuladas = list(map(lambda x: x*2, acumuladas))
            #lower_bound *= 2
            #upper_bound *= 2
            codigo += '0'*esperados
            esperados = 1
        elif lower_bound >= R/2 and upper_bound > (R/4)*3:
            acumuladas = list(map(lambda x: x*2-R, acumuladas))
            #lower_bound = 2*lower_bound - R
            #upper_bound = 2*upper_bound - R
            codigo += '1'*esperados
            esperados = 1
        elif lower_bound >= R/4 and upper_bound<= (R/4)*3:
            acumuladas = list(map(lambda x: x*2-R/2, acumuladas))
            #lower_bound = 2*lower_bound - R/2
            #upper_bound = 2*upper_bound - R/2
            esperados += 1
        else:
            acumuladas = list(map(lambda x: int(((R-lower_bound)*(x-acumuladas[0]))/(acumuladas[len(acumuladas)-1]-acumuladas[0]) + lower_bound),acumuladas))
    return codigo
        #acumuladas = list(map(lambda x: ((upper_bound-lower_bound)*(x-acumuladas[0]))/(acumuladas[len(acumuladas)-1]-acumuladas[0]) + lower_bound,acumuladas))
    m = acumuladas[indices[mensaje[-1]]]
    M = acumuladas[indices[mensaje[-1]]+1]
    t = int(-log2(M-m))
    x_lower = int(2**t * m)
    x_upper = int(2**t * M)
    if x_lower != x_upper:
        if x_lower % 2 == 0:
            return '0' + "{0:b}".format(x_lower)
    return '0' + "{0:b}".format(x_upper)

    
#%%
            
            
"""
Dada la representación binaria del número que representa un mensaje, la
longitud del mensaje y el alfabeto con sus frecuencias 
dar el mensaje original
"""
           
def IntegerArithmeticDecode(codigo,tamanyo_mensaje,alfabeto,frecuencias):
    total = sum(frecuencias)
    probs = list(map(lambda x: x/total,frecuencias))
    acumuladas = [0] + list(accumulate(probs))
    def str_to_float(s):
        xx = 0
        cs = codigo[1:]
        i = 1
        for c in cs:
            xx += int(c) / (2**i)
            i += 1
        return xx
    x = str_to_float(codigo)
    mensaje = ''
    k = 0
    while k < tamanyo_mensaje:
        #print("Iteracio",k)
        #print('Comencem amb x=',x,'i acumuladas =',acumuladas)
        for i in range(0,len(acumuladas)):
            if i == 0:
                continue
            if x < acumuladas[i]:
                mensaje += alfabeto[i-1]
                x = (x - acumuladas[i-1])/(acumuladas[i]-acumuladas[i-1])
                #print('x sense reescalar',x)
                break
        #lower_bound = acumuladas[i-1]
        #upper_bound = acumuladas[i]
        #x = (lambda y: ((upper_bound-lower_bound)*(y-acumuladas[0]))/(acumuladas[len(acumuladas)-1]-acumuladas[0]) + lower_bound)(x)
        #print('x reescalada',x)
        
        #acumuladas = list(map(lambda x: ((upper_bound-lower_bound)*(x-acumuladas[0]))/(acumuladas[len(acumuladas)-1]-acumuladas[0]) + lower_bound,acumuladas))
        #print(acumuladas)
        #x = (x - acumuladas[i-1])/(acumuladas[i]-acumuladas[i-1])
        k += 1
        input()
    return mensaje


             
            
#%%
       




#%%
'''
Definir una función que codifique un mensaje utilizando codificación aritmética con precisión infinita
obtenido a partir de las frecuencias de los caracteres del mensaje.

Definir otra función que decodifique los mensajes codificados con la función 
anterior.
'''


def EncodeArithmetic(mensaje_a_codificar):
    fuente = {}
    for c in mensaje_a_codificar:
        if c not in fuente:
            fuente[c] = 1
        else:
            fuente[c] += 1
    alfabeto, frecuencias = keys, values = fuente.keys(), fuente.values()
    m, M = IntegerArithmeticCode(mensaje_a_codificar, frecuencias, alfabeto)
    return mensaje_codificado,alfabeto,frecuencias
    
def DecodeArithmetic(mensaje_codificado,tamanyo_mensaje,alfabeto,frecuencias):
    mensaje_decodificado = IntegerArithmeticDecode(mensaje_codificado,tamanyo_mensaje,alfabeto,frecuencias)
    return mensaje_decodificado
        
#%%
'''
Ejemplo (!El mismo mensaje se puede codificar con varios códigos¡)

'''

lista_C=['010001110110000000001000000111111000000100010000000000001100000010001111001100001000000',
         '01000111011000000000100000011111100000010001000000000000110000001000111100110000100000000']
alfabeto=['a','b','c','d']
frecuencias=[1,10,20,300]
mensaje='dddcabccacabadac'
tamanyo_mensaje=len(mensaje)  

for C in lista_C:
    mensaje_recuperado=DecodeArithmetic(C,tamanyo_mensaje,alfabeto,frecuencias)
    print(mensaje==mensaje_recuperado)



#%%

'''
Ejemplo

'''

mensaje='La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte. En las calles no había más ruido que el rumor estridente de los remolinos de polvo, trapos, pajas y papeles que iban de arroyo en arroyo, de acera en acera, de esquina en esquina revolando y persiguiéndose, como mariposas que se buscan y huyen y que el aire envuelve en sus pliegues invisibles. Cual turbas de pilluelos, aquellas migajas de la basura, aquellas sobras de todo se juntaban en un montón, parábanse como dormidas un momento y brincaban de nuevo sobresaltadas, dispersándose, trepando unas por las paredes hasta los cristales temblorosos de los faroles, otras hasta los carteles de papel mal pegado a las esquinas, y había pluma que llegaba a un tercer piso, y arenilla que se incrustaba para días, o para años, en la vidriera de un escaparate, agarrada a un plomo. Vetusta, la muy noble y leal ciudad, corte en lejano siglo, hacía la digestión del cocido y de la olla podrida, y descansaba oyendo entre sueños el monótono y familiar zumbido de la campana de coro, que retumbaba allá en lo alto de la esbeltatorre en la Santa Basílica. La torre de la catedral, poema romántico de piedra,delicado himno, de dulces líneas de belleza muda y perenne, era obra del siglo diez y seis, aunque antes comenzada, de estilo gótico, pero, cabe decir, moderado por uninstinto de prudencia y armonía que modificaba las vulgares exageraciones de estaarquitectura. La vista no se fatigaba contemplando horas y horas aquel índice depiedra que señalaba al cielo; no era una de esas torres cuya aguja se quiebra desutil, más flacas que esbeltas, amaneradas, como señoritas cursis que aprietandemasiado el corsé; era maciza sin perder nada de su espiritual grandeza, y hasta sussegundos corredores, elegante balaustrada, subía como fuerte castillo, lanzándosedesde allí en pirámide de ángulo gracioso, inimitable en sus medidas y proporciones.Como haz de músculos y nervios la piedra enroscándose en la piedra trepaba a la altura, haciendo equilibrios de acróbata en el aire; y como prodigio de juegosmalabares, en una punta de caliza se mantenía, cual imantada, una bola grande debronce dorado, y encima otra más pequenya, y sobre ésta una cruz de hierro que acababaen pararrayos.'
mensaje_codificado,alfabeto,frecuencias=EncodeArithmetic(mensaje)
mensaje_recuperado=DecodeArithmetic(mensaje_codificado,len(mensaje),alfabeto,frecuencias)

ratio_compresion=8*len(mensaje)/len(mensaje_codificado)
print(ratio_compresion)

if (mensaje!=mensaje_recuperado):
        print('!!!!!!!!!!!!!!  ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        
        
