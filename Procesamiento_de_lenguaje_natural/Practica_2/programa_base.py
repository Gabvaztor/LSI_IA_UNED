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
    dict['cabe'] = 'PREP'
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
    dict['en'] = 'PREP'
    dict['entre'] = 'PREP'
    dict['so'] = 'PREP'
    dict['más'] = 'PREP'
    dict['salvo'] = 'PREP'
    dict['incluso'] = 'PREP'
    dict['menos'] = 'PREP'
    dict['excepto'] = 'PREP'
    dict['durante'] = 'PREP'
    dict['tras'] = 'PREP'
    return dict

def add_punctuation_marks(dict):
    dict['.'] = 'PUNT'
    dict[','] = 'PUNT'
    dict[';'] = 'PUNT'
    dict[':'] = 'PUNT'
    dict['!'] = 'PUNT'
    dict['¿'] = 'PUNT'
    dict['?'] = 'PUNT'
    dict['¡'] = 'PUNT'
    dict['('] = 'PUNT'
    dict[')'] = 'PUNT'
    dict['/'] = 'PUNT'
    dict['º'] = 'PUNT'
    return dict

def add_interjections(dict):
    dict['¡AH!'] = 'INT'
    dict['¡Ajá!'] = 'INT'
    dict['¡Arre!'] = 'INT'
    dict['¡Arrea!'] = 'INT'
    dict['¡Aúpa!'] = 'INT'
    dict['¡Ay!'] = 'INT'
    dict['¡Bah!'] = 'INT'
    dict['¡Buah!'] = 'INT'
    dict['¡Buu!'] = 'INT'
    dict['¡Cachis!'] = 'INT'
    dict['¡Caray!'] = 'INT'
    dict['¡Cáspita!'] = 'INT'
    dict['¡Chachi!'] = 'INT'
    dict['¡Chao!'] = 'INT'
    dict['¡Ea!'] = 'INT'
    dict['¡Chupi!'] = 'INT'
    dict['¡Eh!'] = 'INT'
    dict['¡Ey!'] = 'INT'
    dict['¡Eureka!'] = 'INT'
    dict['¡Guau!'] = 'INT'
    dict['¡Hala!'] = 'INT'
    dict['¡Hola!'] = 'INT'
    dict['¡Hurra!'] = 'INT'
    dict['¡Huy!'] = 'INT'
    dict['¡Jo!'] = 'INT'
    dict['¡Jolín!'] = 'INT'
    dict['¡Leñe!'] = 'INT'
    dict['¡Oh!'] = 'INT'
    dict['¡Okay!'] = 'INT'
    dict['¡Olé!'] = 'INT'
    dict['¡Ojalá!'] = 'INT'
    dict['¡Puaj!'] = 'INT'
    dict['¡Puf!'] = 'INT'
    dict['¡Anda!'] = 'INT'
    dict['¡Caracoles!'] = 'INT'
    dict['¡Caramba!'] = 'INT'
    dict['¡Cojonudo!'] = 'INT'
    dict['¡Cielos!'] = 'INT'
    dict['¡Genial!'] = 'INT'
    dict['¡Narices!'] = 'INT'
    dict['¡Ostras!'] = 'INT'
    dict['¡Rayos!'] = 'INT'
    dict['¡Viva!'] = 'INT'
    dict['¡Vamos!'] = 'INT'
    # Con varias palabras
    dict['¡Válgame Dios!'] = 'INT'
    dict['¡Madre mía!'] = 'INT'
    dict['¡Dios santo!'] = 'INT'
    dict['¡Ay de mí!'] = 'INT'
    dict['¡Ahí va!'] = 'INT'
    return dict

def add_determinants(dict):
    dict['la'] = 'DET'
    dict['aquel'] = 'DET'
    dict['el'] = 'DET'
    dict['las'] = 'DET'
    dict['los'] = 'DET'
    dict['lo'] = 'DET'
    dict['un'] = 'DET'
    dict['una'] = 'DET'
    dict['unos'] = 'DET'
    dict['unas'] = 'DET'
    dict['este'] = 'DET'
    dict['esta'] = 'DET'
    dict['estos'] = 'DET'
    dict['estas'] = 'DET'
    dict['ese'] = 'DET'
    dict['esas'] = 'DET'
    dict['esos'] = 'DET'
    dict['esa'] = 'DET'
    dict['aquella'] = 'DET'
    dict['aquellos'] = 'DET'
    dict['aquellas'] = 'DET'
    dict['mi'] = 'DET'
    dict['tu'] = 'DET'
    dict['nuestra'] = 'DET'
    dict['vuestra'] = 'DET'
    dict['nuestro'] = 'DET'
    dict['vuestro'] = 'DET'
    dict['nuestros'] = 'DET'
    dict['nuestras'] = 'DET'
    dict['vuestros'] = 'DET'
    dict['vuestras'] = 'DET'
    #dict[''] = 'DET'
    return dict
def add_conjuntions(dict):
    dict['que'] = 'CONJ'
    dict['y'] = 'CONJ'
    dict['aunque'] = 'CONJ'
    dict['con'] = 'CONJ'
    dict['pero'] = 'CONJ'
    dict['e'] = 'CONJ'
    dict['ni'] = 'CONJ'
    dict['o'] = 'CONJ'
    dict['u'] = 'CONJ'
    dict['bien'] = 'CONJ'
    dict['bien;'] = 'CONJ'
    dict['ora'] = 'CONJ'
    dict['ora;'] = 'CONJ'
    dict['ya'] = 'CONJ'
    dict['mas'] = 'CONJ'
    dict['sino'] = 'CONJ'
    return dict



dict = {}
# Signos de puntuación
dict = add_punctuation_marks(dict)
# Interjecciones
dict = add_interjections(dict)
# Determinantes
dict = add_determinants(dict)
# Preposiciones
dict = add_prepositions(dict)
# Conjunciones
dict = add_conjuntions(dict)


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
pt(taggedText)

for item in taggedText:
    if item[0] in dict:
        pt(item[0]+' '+dict[item[0]])
    else:
        pt(item[0]+' '+item[1])
    


sys.exit()
