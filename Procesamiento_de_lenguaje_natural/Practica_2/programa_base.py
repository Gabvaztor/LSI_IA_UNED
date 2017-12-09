# -*- coding: utf-8 -*-


import sys
from collections import Counter
import nltk
#nltk.download()
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from UtilsFunctions import *


#reload(sys)
#sys.setdefaultencoding('utf8')

#Lectura del fichero de texto
f = open('D:\\Master\\PLN\\Practica3\\instrumentos.txt',encoding="utf-8")
file_read = f.read()
freqdist = nltk.FreqDist()
words = nltk.word_tokenize(file_read)
fd = nltk.FreqDist(word.lower() for word in words)
fdf= fd.most_common()

pt('Palabras del texto ordenadas por frecuencia')
t=''
for w in fdf:
    t+='('+w[0]+','+str(w[1])+') '
pt(t)



def add_prepositions(dict):
    dict['a'] = 'PREP'
    dict['ante'] = 'PREP'
    dict['bajo'] = 'PREP'
    dict['con'] = 'PREP'
    dict['contra'] = 'PREP'
    dict['de'] = 'PREP'
    dict['desde'] = 'PREP'
    dict['hacia'] = 'PREP'
    dict['hasta'] = 'PREP'
    dict['para'] = 'PREP'
    dict['por'] = 'PREP'
    dict['según'] = 'PREP'
    dict['segun'] = 'PREP'
    dict['sin'] = 'PREP'
    dict['sobre'] = 'PREP'
    dict['tras'] = 'PREP'
    dict['durante'] = 'PREP'
    dict['mediante'] = 'PREP'
    return dict

dict = {}
dict['.']='PUNT'
dict['la']='DET'
# Preposiciones
dict = add_prepositions(dict)
dict['que']='CONJ'
dict['en']='PREP'
dict['el']='DET'
#Aquí hay se añaden las palabras del diccionario y sus etiquetas




p=[
    (r'.*amos$','VIP1S'),
    (r'.*imos$','VIP1S'),
    (r'.*a$','NCFS'),
    (r'.*$','NCMS'),
    #Aquí hay se añaden los patrones necesarios
    ]



rt=nltk.RegexpTagger(p)
taggedText=rt.tag(words)
for item in taggedText:
    if item[0] in dict:
        pt(item[0]+' '+dict[item[0]])
    else:
        pt(item[0]+' '+item[1])
    


sys.exit()
