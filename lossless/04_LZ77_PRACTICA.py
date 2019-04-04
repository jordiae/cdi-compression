# -*- coding: utf-8 -*-

# Jordi Armengol, Bruno Tamborero

"""
Dado un mensaje, el tamaño de la ventana de trabajo L, y el tamaño
del buffer de búsqueda S dar la codificación del mensaje usando el
algoritmo LZ77


mensaje='cabracadabrarrarr'
LZ77Code(mensaje,12,6)=['c', 0, 0], ['a', 0, 0],  ['b', 0, 0],
 ['r', 0, 0],  ['c', 1, 3],  ['d', 1, 2], ['r', 4, 7],  ['EOF', 4, 3]]
  
  
[símbolo, longitud_máxima_cadena, posición]    
"""


def LZ77Code(mensaje,S=12,L=6):
	output = []
	search_buffer = ''
	dic = ''
	i = 0
	while i < len(mensaje):
		if i == 0:
			output.append([mensaje[0],0,0])
			search_buffer = mensaje[1:1+L]
			dic = mensaje[0]
			i += 1
			continue
		posicion = 0
		longitud_maxima_cadena = 0
		simbolo = search_buffer[0]
		search = dic + search_buffer
		j = len(dic) - 1
		while j >= 0:
			if dic[j] == search_buffer[0]:
				l = 1
				while len(dic)+l < len(search):
					if search[j+l] == search[len(dic)+l]:
						l +=1
					else:
						break
				if l > longitud_maxima_cadena:
					posicion = len(dic) - j
					longitud_maxima_cadena = l
					if len(dic)+l < len(search):
						simbolo = search[len(dic)+l]
					elif i+longitud_maxima_cadena < len(search):
						simbolo = search[i+longitud_maxima_cadena]
					else:
						simbolo = search[len(dic)+l-1]
						longitud_maxima_cadena -= 1
						if longitud_maxima_cadena == 0:
							posicion = 0
			j -= 1	
		i += longitud_maxima_cadena+1
		dic = mensaje[i-S if i-S >= 0 else 0:i]
		search_buffer = mensaje[i:i+L]
		output.append([simbolo,longitud_maxima_cadena,posicion])
	eof = ['EOF',output[-1][1]+1,output[-1][2]]
	output[-1] = eof
	return output

mensaje='cabracadabrarrarr'
codigo_solucion = [['c', 0, 0], ['a', 0, 0],  ['b', 0, 0],['r', 0, 0],  ['c', 1, 3],  ['d', 1, 2], ['r', 4, 7],  ['EOF', 4, 3]]
print('Encode: Ejemplo correcto?', 'Sí' if LZ77Code(mensaje,12,6) == codigo_solucion else 'No')
print()

"""
Dado un mensaje codificado con el algoritmo LZ77 hallar el mensaje 
correspondiente 

code=[['p', 0, 0], ['a', 0, 0],  ['t', 0, 0],  ['d', 1, 2],  ['e', 0, 0],
 ['c', 0, 0], ['b', 1, 4],  ['r', 0, 0], ['EOF', 1, 3]]
 
LZ77Decode(code)='patadecabra'

"""   
def LZ77Decode(codigo):
	mensaje_recuperado = ''
	for [simbolo, longitud_maxima_cadena, posicion] in codigo:
		if longitud_maxima_cadena == 0 and posicion == 0:
			mensaje_recuperado += simbolo
		else:
			i = 0
			while i < longitud_maxima_cadena:
				mensaje_recuperado += mensaje_recuperado[-posicion]
				i += 1
			mensaje_recuperado += simbolo
	return mensaje_recuperado[:-3]
    
code=[['p', 0, 0], ['a', 0, 0],  ['t', 0, 0],  ['d', 1, 2],  ['e', 0, 0],['c', 0, 0], ['b', 1, 4],  ['r', 0, 0], ['EOF', 1, 3]]
print('Decode: Ejemplo correcto?','Sí' if LZ77Decode(code)=='patadecabra' else 'No')
print()
mensaje='cabracadabrarrarr'
print("mensaje='cabracadabrarrarr'")
print("LZ77Code(mensaje,12,6) =",LZ77Code(mensaje,12,6))
mensaje_recuperado = LZ77Decode(LZ77Code(mensaje,12,6))
print("mensaje = LZ77Decode(LZ77Code(mensaje,12,6)?",mensaje==mensaje_recuperado)
print()

"""
Jugar con los valores de S y L (bits_o y bits_l)
para ver sus efectos (tiempo, tamaño...)
"""
import time
def ver_efectos_S_L(mensaje,valores_a_provar):
	for bits_o, bits_l in valores_a_provar:
		S=2**bits_o
		L=2**bits_l
		print('Probando con S = ',S,'y L =',L)
		start_time = time.clock()
		mensaje_codificado=LZ77Code(mensaje,S,L)
		print (time.clock() - start_time, "seconds code")
		start_time = time.clock()
		mensaje_recuperado=LZ77Decode(mensaje_codificado)
		print (time.clock() - start_time, "seconds decode")
		ratio_compresion=8*len(mensaje)/((bits_o+bits_l+8)*len(mensaje_codificado))
		print('Longitud de mensaje codificado:', len(mensaje_codificado))
		print('Ratio de compresión:', ratio_compresion)
		print('Correcto?', 'Sí' if mensaje==mensaje_recuperado else 'No')
		print()
	print()
	print()
	print('''
La mejor ratio de compresión, alrededor de 4, la hemos obtenido con
S =  4096 y L = 256. Además, no es la que tarda más tiempo, aunque tampoco es la más rápida.\n
S en principio disminuye el tamaño de la compresión, pero aumenta el tiempo de ejecución, porque el espacio
de búsqueda es mayor. Esto para el encoding, el decoding es muy rápido en general y no cambia significativamente
para los distintos valores.\n
Respecto a L, para los valores que hemos probado, a mayor L, mejor compresión y menor tiempo, aunque hasta un límite.\n
Hay que encontrar unos valores de S y L según cómo valoremos el trade-off entre compresión vs tiempo.\n
Pero no sólo esto, también hay que encontrar un valor de L que se comporte bien con S.''')
	print()
	print()



mensaje='La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte. En las calles no había más ruido que el rumor estridente de los remolinos de polvo, trapos, pajas y papeles que iban de arroyo en arroyo, de acera en acera, de esquina en esquina revolando y persiguiéndose, como mariposas que se buscan y huyen y que el aire envuelve en sus pliegues invisibles. Cual turbas de pilluelos, aquellas migajas de la basura, aquellas sobras de todo se juntaban en un montón, parábanse como dormidas un momento y brincaban de nuevo sobresaltadas, dispersándose, trepando unas por las paredes hasta los cristales temblorosos de los faroles, otras hasta los carteles de papel mal pegado a las esquinas, y había pluma que llegaba a un tercer piso, y arenilla que se incrustaba para días, o para años, en la vidriera de un escaparate, agarrada a un plomo. Vetusta, la muy noble y leal ciudad, corte en lejano siglo, hacía la digestión del cocido y de la olla podrida, y descansaba oyendo entre sueños el monótono y familiar zumbido de la campana de coro, que retumbaba allá en lo alto de la esbeltatorre en la Santa Basílica. La torre de la catedral, poema romántico de piedra,delicado himno, de dulces líneas de belleza muda y perenne, era obra del siglo diez y seis, aunque antes comenzada, de estilo gótico, pero, cabe decir, moderado por uninstinto de prudencia y armonía que modificaba las vulgares exageraciones de estaarquitectura. La vista no se fatigaba contemplando horas y horas aquel índice depiedra que señalaba al cielo; no era una de esas torres cuya aguja se quiebra desutil, más flacas que esbeltas, amaneradas, como señoritas cursis que aprietandemasiado el corsé; era maciza sin perder nada de su espiritual grandeza, y hasta sussegundos corredores, elegante balaustrada, subía como fuerte castillo, lanzándosedesde allí en pirámide de ángulo gracioso, inimitable en sus medidas y proporciones.Como haz de músculos y nervios la piedra enroscándose en la piedra trepaba a la altura, haciendo equilibrios de acróbata en el aire; y como prodigio de juegosmalabares, en una punta de caliza se mantenía, cual imantada, una bola grande debronce dorado, y encima otra más pequenya, y sobre ésta una cruz de hierro que acababaen pararrayos. La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte. En las calles no había más ruido que el rumor estridente de los remolinos de polvo, trapos, pajas y papeles que iban de arroyo en arroyo, de acera en acera, de esquina en esquina revolando y persiguiéndose, como mariposas que se buscan y huyen y que el aire envuelve en sus pliegues invisibles. Cual turbas de pilluelos, aquellas migajas de la basura, aquellas sobras de todo se juntaban en un montón, parábanse como dormidas un momento y brincaban de nuevo sobresaltadas, dispersándose, trepando unas por las paredes hasta los cristales temblorosos de los faroles, otras hasta los carteles de papel mal pegado a las esquinas, y había pluma que llegaba a un tercer piso, y arenilla que se incrustaba para días, o para años, en la vidriera de un escaparate, agarrada a un plomo. Vetusta, la muy noble y leal ciudad, corte en lejano siglo, hacía la digestión del cocido y de la olla podrida, y descansaba oyendo entre sueños el monótono y familiar zumbido de la campana de coro, que retumbaba allá en lo alto de la esbeltatorre en la Santa Basílica. La torre de la catedral, poema romántico de piedra,delicado himno, de dulces líneas de belleza muda y perenne, era obra del siglo diez y seis, aunque antes comenzada, de estilo gótico, pero, cabe decir, moderado por uninstinto de prudencia y armonía que modificaba las vulgares exageraciones de estaarquitectura. La vista no se fatigaba contemplando horas y horas aquel índice depiedra que señalaba al cielo; no era una de esas torres cuya aguja se quiebra desutil, más flacas que esbeltas, amaneradas, como señoritas cursis que aprietandemasiado el corsé; era maciza sin perder nada de su espiritual grandeza, y hasta sussegundos corredores, elegante balaustrada, subía como fuerte castillo, lanzándosedesde allí en pirámide de ángulo gracioso, inimitable en sus medidas y proporciones.Como haz de músculos y nervios la piedra enroscándose en la piedra trepaba a la altura, haciendo equilibrios de acróbata en el aire; y como prodigio de juegosmalabares, en una punta de caliza se mantenía, cual imantada, una bola grande debronce dorado, y encima otra más pequenya, y sobre ésta una cruz de hierro que acababaen pararrayos. La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte. En las calles no había más ruido que el rumor estridente de los remolinos de polvo, trapos, pajas y papeles que iban de arroyo en arroyo, de acera en acera, de esquina en esquina revolando y persiguiéndose, como mariposas que se buscan y huyen y que el aire envuelve en sus pliegues invisibles. Cual turbas de pilluelos, aquellas migajas de la basura, aquellas sobras de todo se juntaban en un montón, parábanse como dormidas un momento y brincaban de nuevo sobresaltadas, dispersándose, trepando unas por las paredes hasta los cristales temblorosos de los faroles, otras hasta los carteles de papel mal pegado a las esquinas, y había pluma que llegaba a un tercer piso, y arenilla que se incrustaba para días, o para años, en la vidriera de un escaparate, agarrada a un plomo. Vetusta, la muy noble y leal ciudad, corte en lejano siglo, hacía la digestión del cocido y de la olla podrida, y descansaba oyendo entre sueños el monótono y familiar zumbido de la campana de coro, que retumbaba allá en lo alto de la esbeltatorre en la Santa Basílica. La torre de la catedral, poema romántico de piedra,delicado himno, de dulces líneas de belleza muda y perenne, era obra del siglo diez y seis, aunque antes comenzada, de estilo gótico, pero, cabe decir, moderado por uninstinto de prudencia y armonía que modificaba las vulgares exageraciones de estaarquitectura. La vista no se fatigaba contemplando horas y horas aquel índice depiedra que señalaba al cielo; no era una de esas torres cuya aguja se quiebra desutil, más flacas que esbeltas, amaneradas, como señoritas cursis que aprietandemasiado el corsé; era maciza sin perder nada de su espiritual grandeza, y hasta sussegundos corredores, elegante balaustrada, subía como fuerte castillo, lanzándosedesde allí en pirámide de ángulo gracioso, inimitable en sus medidas y proporciones.Como haz de músculos y nervios la piedra enroscándose en la piedra trepaba a la altura, haciendo equilibrios de acróbata en el aire; y como prodigio de juegosmalabares, en una punta de caliza se mantenía, cual imantada, una bola grande debronce dorado, y encima otra más pequenya, y sobre ésta una cruz de hierro que acababaen pararrayos. La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte. En las calles no había más ruido que el rumor estridente de los remolinos de polvo, trapos, pajas y papeles que iban de arroyo en arroyo, de acera en acera, de esquina en esquina revolando y persiguiéndose, como mariposas que se buscan y huyen y que el aire envuelve en sus pliegues invisibles. Cual turbas de pilluelos, aquellas migajas de la basura, aquellas sobras de todo se juntaban en un montón, parábanse como dormidas un momento y brincaban de nuevo sobresaltadas, dispersándose, trepando unas por las paredes hasta los cristales temblorosos de los faroles, otras hasta los carteles de papel mal pegado a las esquinas, y había pluma que llegaba a un tercer piso, y arenilla que se incrustaba para días, o para años, en la vidriera de un escaparate, agarrada a un plomo. Vetusta, la muy noble y leal ciudad, corte en lejano siglo, hacía la digestión del cocido y de la olla podrida, y descansaba oyendo entre sueños el monótono y familiar zumbido de la campana de coro, que retumbaba allá en lo alto de la esbeltatorre en la Santa Basílica. La torre de la catedral, poema romántico de piedra,delicado himno, de dulces líneas de belleza muda y perenne, era obra del siglo diez y seis, aunque antes comenzada, de estilo gótico, pero, cabe decir, moderado por uninstinto de prudencia y armonía que modificaba las vulgares exageraciones de estaarquitectura. La vista no se fatigaba contemplando horas y horas aquel índice depiedra que señalaba al cielo; no era una de esas torres cuya aguja se quiebra desutil, más flacas que esbeltas, amaneradas, como señoritas cursis que aprietandemasiado el corsé; era maciza sin perder nada de su espiritual grandeza, y hasta sussegundos corredores, elegante balaustrada, subía como fuerte castillo, lanzándosedesde allí en pirámide de ángulo gracioso, inimitable en sus medidas y proporciones.Como haz de músculos y nervios la piedra enroscándose en la piedra trepaba a la altura, haciendo equilibrios de acróbata en el aire; y como prodigio de juegosmalabares, en una punta de caliza se mantenía, cual imantada, una bola grande debronce dorado, y encima otra más pequenya, y sobre ésta una cruz de hierro que acababaen pararrayos.'
bits_o=12
bits_l=4
S=2**bits_o
L=2**bits_l

ver_efectos_S_L(mensaje,[(12,4),(16,4),(18,4),(12,2),(12,8),(12,10)])

import time
start_time = time.clock()
mensaje_codificado=LZ77Code(mensaje,S,L)
print (time.clock() - start_time, "seconds code")
start_time = time.clock()
mensaje_recuperado=LZ77Decode(mensaje_codificado)
print (time.clock() - start_time, "seconds decode")
ratio_compresion=8*len(mensaje)/((bits_o+bits_l+8)*len(mensaje_codificado))
print('Longitud de mensaje codificado:', len(mensaje_codificado))
print('Ratio de compresión:', ratio_compresion)
print('Correcto?','Sí' if mensaje==mensaje_recuperado else 'No')