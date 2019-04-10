def prefix(palabras):
	for palabra1 in palabras:
		for palabra2 in palabras:
			if palabra1 != palabra2:
				if palabra1.startswith(palabra2) or palabra2.startswith(palabra1):
					return False
	return True

from functools import reduce
def kraft1(L, q=2):
    return reduce(lambda x, y: x+pow(q,-y), L,0) <= 1

def kraft2_(L, q=2):
    nwords = 0
    while kraft1(L,q):
        L.append(max(L))
        if kraft1(L,q):
        	nwords += 1
    return nwords

def kraft3_(L, Ln, q=2):
    nwords = 0
    while kraft1(L,q):
        L.append(Ln)
        if kraft1(L,q):
        	nwords += 1
    return nwords

def EncodeRLE(mensaje):
    codigo = ''
    i = 0
    while i < len(mensaje):
        c = mensaje[i]
        j = i + 1
        count = 1
        while j < len(mensaje) and mensaje[j] == c:
            count += 1
            j += 1
        i = j
        codigo += c + str(count)
    return codigo

def DecodeRLE(codigo):
    i = 0
    mensaje = ''
    while i < len(codigo):
        j = i + 1
        count = ''
        while j < len(codigo) and codigo[j].isdigit():
            count += codigo[j]
            j += 1
        count = int(count)
        mensaje += codigo[i] * count
        i = j
    return mensaje
