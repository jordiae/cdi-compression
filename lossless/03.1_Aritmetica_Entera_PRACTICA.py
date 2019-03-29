# -*- coding: utf-8 -*-
"""
@author: martinez

Jordi Armengol, Bruno Tamborero.
"""

import math
# ¿Se tenía que usar random? Estaba importado en el enunciado
# import random

from itertools import accumulate



#%%
"""
Dado un mensaje y su alfabeto con sus frecuencias dar el código 
que representa el mensaje utilizando precisión infinita (reescalado)

El intervalo de trabajo será: [0,R), R=2**k, k menor entero tal que R>4T

T: suma total de frecuencias

"""

# Basado en https://people.cs.nctu.edu.tw/~cmliu/Courses/Compression/chap4.pdf

def bitfield(n):
    return [int(digit) for digit in bin(n)[2:]]

def bitfield_to_dec(bit_array):
    dec = 0
    for i in range(len(bit_array)):
        dec += bit_array[-(i+1)] * 2**i
    return dec

# Most significant bit equal
def equal_msb(a, b):
    return a[0] == b[0]

def get_bitfields_lower_upper(lower_bound, upper_bound, nbits):
    lower = bitfield(lower_bound)
    upper = bitfield(upper_bound)
    return (nbits-len(lower))*[0] + lower, (nbits-len(upper))*[0] + upper # para que tengan nbits

def shift_left_and_set_lsb(bfield, bit):
    length = len(bfield)
    for i in range(1, len(bfield)):
        bfield[i-1] = bfield[i]
    bfield[length-1] = bit
    return bfield

def e3(lower, upper):
    return lower[0:2] == [0, 1] and upper[0:2] == [1, 0]

def IntegerArithmeticCode(mensaje,alfabeto,frecuencias):
    codigo = ''
    T = sum(frecuencias)
    acumuladas = [0] + list(accumulate(frecuencias))
    indices = dict(zip(alfabeto,range(len(frecuencias))))
    k = int(math.log2(4*T)) + 1
    R = 2**k
    lower, upper = get_bitfields_lower_upper(0, R-1, k)
    e3_counter  = 0
    for c in mensaje:
        decimal_lower_bound = bitfield_to_dec(lower)
        decimal_upper_bound = bitfield_to_dec(upper)
        indice = indices[c]
        lower_c = acumuladas[indice]
        upper_c = acumuladas[indice+1]
        new_lower = decimal_lower_bound + ((decimal_upper_bound-decimal_lower_bound+1)*lower_c)//T
        new_upper = decimal_lower_bound + ((decimal_upper_bound-decimal_lower_bound+1)*upper_c)//T - 1
        lower, upper = get_bitfields_lower_upper(new_lower, new_upper, k)
        while equal_msb(lower, upper) or e3(lower, upper):
            if equal_msb(lower, upper): # escalado e1, e2
                b = lower[0]
                codigo += str(b) + e3_counter * str(1-b) # send b (+ los esperados por e3)
                lower = shift_left_and_set_lsb(lower, 0)
                upper = shift_left_and_set_lsb(upper, 1)
                e3_counter = 0
            if e3(lower, upper): # escalado e3
                lower = shift_left_and_set_lsb(lower, 0)
                upper = shift_left_and_set_lsb(upper, 1)
                # complement
                lower[0] = 1-lower[0]
                upper[0] = 1-upper[0]
                e3_counter += 1
    for e in lower:
        codigo += str(e) + e3_counter * str(1-e)
        if e3_counter > 0:
            e3_counter = 0
    return codigo

    
    
#%%
            
            
"""
Dada la representación binaria del número que representa un mensaje, la
longitud del mensaje y el alfabeto con sus frecuencias 
dar el mensaje original
"""
           
def IntegerArithmeticDecode(codigo,tamanyo_mensaje,alfabeto,frecuencias):
    mensaje = ''
    T = sum(frecuencias)
    k = int(math.log2(4*T)) + 1
    R = 2**k
    acumuladas = list(accumulate(frecuencias))
    lower = [0]*k
    upper = [1]*k
    c_k = k
    t = []
    for i in range(c_k):
        t.append(int(codigo[i]))
    while len(mensaje) < tamanyo_mensaje:
        decimal_t = bitfield_to_dec(t)
        decimal_lower_bound = bitfield_to_dec(lower)
        decimal_upper_bound = bitfield_to_dec(upper)
        j = 0
        frec_acum = int(((decimal_t-decimal_lower_bound+1)*T-1)/(decimal_upper_bound-decimal_lower_bound+1))
        while acumuladas[j] <= frec_acum:
            j += 1
        mensaje += alfabeto[j]
        lower_c = 0 if j <= 0 else acumuladas[j-1]
        upper_c = acumuladas[j]
        new_lower = decimal_lower_bound + ((decimal_upper_bound-decimal_lower_bound+1)*lower_c)//T
        new_upper = decimal_lower_bound + ((decimal_upper_bound-decimal_lower_bound+1)*upper_c)//T - 1
        lower, upper = get_bitfields_lower_upper(new_lower, new_upper, k)
        while equal_msb(lower, upper) or e3(lower, upper):
            if equal_msb(lower, upper):
                lower = shift_left_and_set_lsb(lower, 0)
                upper = shift_left_and_set_lsb(upper, 1)
                t = shift_left_and_set_lsb(t, int(codigo[c_k]))
                c_k += 1
            if e3(lower, upper):
                lower = shift_left_and_set_lsb(lower, 0)
                upper = shift_left_and_set_lsb(upper, 1)
                t = shift_left_and_set_lsb(t, int(codigo[c_k]))
                c_k += 1
                # complement
                lower[0] = 1-lower[0]
                upper[0] = 1-upper[0]
                t[0] = 1-t[0]
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
    alfabeto, frecuencias = list(fuente.keys()), list(fuente.values())
    mensaje_codificado = IntegerArithmeticCode(mensaje_a_codificar, alfabeto, frecuencias)
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
        
        
