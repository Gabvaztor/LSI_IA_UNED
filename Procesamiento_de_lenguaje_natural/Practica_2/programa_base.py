# -*- coding: utf-8 -*-


import sys
from collections import Counter
import nltk
#nltk.download()
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from UtilsFunctions import *
import os

#reload(sys)
#sys.setdefaultencoding('utf8')

#Lectura del fichero de texto
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
#f = open('D:\\Master\\PLN\\Practica3\\instrumentos.txt',encoding="utf-8")
f = open(os.path.join(__location__, 'instrumentos.txt'),encoding="utf-8")
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
    # Con varias palabras / analizar
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
    dict['su'] = 'DET'
    dict['sus'] = 'DET'
    dict['tus'] = 'DET'
    dict['mis'] = 'DET'
    dict['nuestra'] = 'DET'
    dict['vuestra'] = 'DET'
    dict['nuestro'] = 'DET'
    dict['vuestro'] = 'DET'
    dict['nuestros'] = 'DET'
    dict['nuestras'] = 'DET'
    dict['vuestros'] = 'DET'
    dict['vuestras'] = 'DET'
    dict['cuyo'] = 'DET'
    dict['cuyos'] = 'DET'
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

def add_pronouns(dict):
    dict['yo'] = 'PRON'
    dict['tú'] = 'PRON'
    dict['él'] = 'PRON'
    dict['nosotros'] = 'PRON'
    dict['vosotros'] = 'PRON'
    dict['ellos'] = 'PRON'
    dict['éste'] = 'PRON'
    dict['éstos'] = 'PRON'
    dict['ése'] = 'PRON'
    dict['aquél'] = 'PRON'
    dict['ésa'] = 'PRON'
    dict['aquélla'] = 'PRON'
    dict['aquéllo'] = 'PRON'
    dict['éstos'] = 'PRON'
    dict['ésos'] = 'PRON'
    dict['aquéllos'] = 'PRON'
    dict['aquéllas'] = 'PRON'
    dict['éstas'] = 'PRON'
    dict['ésas'] = 'PRON'
    dict['ello'] = 'PRON'
    dict['se'] = 'PRON'
    return dict

def add_articles(dict):
    # Preferencia de los determinantes
    return dict
def add_numbers(dict):
    dict['cero'] = 'NUM'
    dict['uno'] = 'NUM'
    dict['dos'] = 'NUM'
    dict['tres'] = 'NUM'
    dict['cuatro'] = 'NUM'
    dict['cinco'] = 'NUM'
    dict['seis'] = 'NUM'
    dict['siete'] = 'NUM'
    dict['ocho'] = 'NUM'
    dict['nueve'] = 'NUM'
    dict['diez'] = 'NUM'
    dict['once'] = 'NUM'
    dict['doce'] = 'NUM'
    dict['trece'] = 'NUM'
    dict['catorce'] = 'NUM'
    dict['quice'] = 'NUM'
    dict['once'] = 'NUM'
    return dict
def add_adverbs(dict):
    dict['ahí'] = 'ADV'
    dict['allí'] = 'ADV'
    dict['aquí'] = 'ADV'
    dict['acá'] = 'ADV'
    dict['delante'] = 'ADV'
    dict['detrás'] = 'ADV'
    dict['arriba'] = 'ADV'
    dict['abajo'] = 'ADV'
    dict['cerca'] = 'ADV'
    dict['lejos'] = 'ADV'
    dict['encima'] = 'ADV'
    dict['fuera'] = 'ADV'
    dict['dentro'] = 'ADV'
    dict['ya'] = 'ADV'
    dict['aún'] = 'ADV'
    dict['hoy'] = 'ADV'
    dict['tarde'] = 'ADV'
    dict['pronto'] = 'ADV'
    dict['todavía'] = 'ADV'
    dict['ayer'] = 'ADV'
    dict['recién'] = 'ADV'
    dict['nunca'] = 'ADV'
    dict['siempre'] = 'ADV'
    dict['jamás'] = 'ADV'
    dict['ahora'] = 'ADV'
    dict['mal'] = 'ADV'
    dict['bien'] = 'ADV'
    dict['regular'] = 'ADV'
    dict['despacio'] = 'ADV'
    dict['así'] = 'ADV'
    dict['mejor'] = 'ADV'
    dict['peor'] = 'ADV'
    dict['claro'] = 'ADV'
    dict['similar'] = 'ADV'
    dict['muy'] = 'ADV'
    dict['más'] = 'ADV'
    dict['poco'] = 'ADV'
    dict['bastante'] = 'ADV'
    dict['demasiado'] = 'ADV'
    dict['menos'] = 'ADV'
    dict['mucho'] = 'ADV'
    dict['algo'] = 'ADV'
    dict['casi'] = 'ADV'
    dict['acaso'] = 'ADV'
    dict['quizás'] = 'ADV'
    dict['tampoco'] = 'ADV'
    dict['dónde'] = 'ADV'
    dict['cuándo'] = 'ADV'
    dict['qué'] = 'ADV'
    dict['cuán'] = 'ADV'
    dict['cuánto'] = 'ADV'
    dict['cuánta'] = 'ADV'
    dict['cierto'] = 'ADV'
    dict['cierta'] = 'ADV'
    dict['ciertos'] = 'ADV'
    dict['ciertas'] = 'ADV'
    dict['varios'] = 'ADV'
    dict['ordenadas'] = 'ADV'
    dict['dispuestas'] = 'ADV'
    return dict

def add_adjectives(dict):
    dict['pequeño'] = 'ADJ'
    dict['agudo'] = 'ADJ'
    dict['accionados'] = 'ADJ'
    dict['golpeadas'] = 'ADJ'
    dict['cuerda'] = 'ADJ'
    dict['largo'] = 'ADJ'
    dict['grande'] = 'ADJ'
    dict['grave'] = 'ADJ'
    dict['cilíndrica'] = 'ADJ'
    dict['hueco'] = 'ADJ'
    dict['hueca'] = 'ADJ'
    dict['estirada'] = 'ADJ'
    dict['grave'] = 'ADJ'
    dict['intermedio'] = 'ADJ'
    dict['circulares'] = 'ADJ'
    dict['metálicas'] = 'ADJ'
    return dict
# Verbos
def add_verbs(dict):
    dict['compone'] = 'VMIP3S0'
    dict['hacen'] = 'VMIP3S0'
    dict['compuesto'] = 'VMPS000'
    dict['tocado'] = 'VMPS000'
    dict['cubierto'] = 'VMPS000'
    dict['tocado'] = 'VMIP3S0'
    dict['sentado'] = 'VMIP3S0'
    dict['responden'] = 'VMIP3P0'
    dict['consistente'] = 'VMIP3P0'
    dict['va'] = 'VMIP3S0'
    dict['produce'] = 'VMIP3S0'
    dict['impele'] = 'VMIP3S0'
    dict['está'] = 'VMIP3S0'
    dict['coloca'] = 'VMIP3S0'
    dict['formado'] = 'VMIP3S0'
    dict['cierran'] = 'VMIP3P0'
    dict['juegan'] = 'VMIP3P0'
    dict['permiten'] = 'VMIP3P0'
    dict['tapan'] = 'VMIP3P0'
    return dict
# Sustantivos
def add_nouns(dict):
    dict['trastes'] = 'NCMP'
    dict['dedos'] = 'NCMP'
    dict['metal'] = 'NCMS'
    dict['bases'] = 'NCNP'
    dict['piel'] = 'NCFS'
    dict['percusión'] = 'NCFS'
    dict['palillos'] = 'NCMP'
    dict['especie'] = 'NCFS'
    dict['voces'] = 'NCFP'
    dict['extremos'] = 'NCMP'
    dict['estuches'] = 'NCMP'
    dict['llaves'] = 'NCFP'
    dict['sonidos'] = 'NCMP'
    dict['material'] = 'NCMS'
    dict['agujeros'] = 'NCMP'
    dict['teclado'] = 'NCMS'
    return dict

def add_others(dict):
    dict['al'] = 'PREP y DET'
    dict['del'] = 'PREP y DET'
    dict['tocarlo'] = 'VMN0000 y DET'
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
# Pronombres
dict = add_pronouns(dict)
# Adjetivos
dict = add_adjectives(dict)
# Artículos
dict = add_articles(dict)
# Numeral
dict = add_numbers(dict)
# Advervios
dict = add_adverbs(dict)
# Verbos
dict = add_verbs(dict)
# Sustantivos
dict = add_nouns(dict)
# Otros
dict = add_others(dict)

p=[
    # Aquí hay se añaden los patrones necesarios
    # Abreviaturas
    (r'.*(?:(?<=\.|\s)[A-Z]\.)+', 'ABRV'),
    # Advervios
    (r'.*mente$', 'ADV'),
    # Adjetivos
    (r'.*al$', 'ADJ'),
    (r'.*oso$', 'ADJ'),
    (r'.*osa$', 'ADJ'),
    (r'.*ivo$', 'ADJ'),
    (r'.*usco$', 'ADJ'),
    # Numerales
    (r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?','NUM'),
    (r'.*uno$', 'NUM'),
    (r'.*doa$', 'NUM'),
    (r'.*tres$', 'NUM'),
    (r'.*cuatro$', 'NUM'),
    (r'.*cinco$', 'NUM'),
    (r'.*seis$', 'NUM'),
    (r'.*siete$', 'NUM'),
    (r'.*ocho$', 'NUM'),
    (r'.*nueve$', 'NUM'),
    (r'.*mil$', 'NUM'),
    (r'.*miles$', 'NUM'),
    (r'.*millón$', 'NUM'),
    (r'.*millones$', 'NUM'),
    (r'.*entos$', 'NUM'),
    (r'.*enta$', 'NUM'),
    (r'.*einte$', 'NUM'),
    (r'.*einta$', 'NUM'),
    # Verbos
    (r'.*ríamos$', 'VMCP1P0'), # condicional presente plural
    (r'.*amos$', 'VM0P1P'),
    (r'.*emos$', 'VM0P10'),
    (r'.*imos$', 'V0IP1P'),
    (r'.*iste$', 'VMI02S0'), # pretérito perfecto simple segunda persona singular
    (r'.*isteis$', 'VMI02P0'), # pretérito perfecto simple segunda persona plural
    (r'.*ría$', 'VMCP0S0'), # condicional presente singular 1 y 3 persona
    (r'.*rías$', 'VMCP2S0'), # condicional presente singular
    (r'.*ados$', 'VMCP2S0'), #
    (r'.*ado$', 'VMCP2S0'), #
    (r'.*í$', 'VMCP2S0'), # pasado simple
    (r'.*ó$', 'VMCP2S0'), # pasado simple
    (r'.*rá$', 'VMIF3S0'), # futuro simple
    (r'.*ré$', 'VMIF1S0'), # futuro simple primera persona
    (r'.*ar$', 'VMN0000'), # infinitivo
    (r'.*er$', 'VMN0000'), # infinitivo
    (r'.*ir$', 'VMN0000'), # infinitivo
    # Sustantivos
    (r'.*las$', 'NCFP'),
    (r'.*lase$', 'NCFS'),
    (r'.*lases$', 'NCFP'),
    (r'.*as$', 'NCFP'),
    (r'.*a$', 'NCFS'),
    (r'.*os$', 'NCMP'),
    (r'.*$', 'NCMS')
    ]



rt=nltk.RegexpTagger(p)
taggedText=rt.tag(words)

for item in taggedText:
    if item[0] in dict:
        pt(item[0]+' ('+dict[item[0]]+')')
    else:
        pt(item[0]+' ('+item[1]+')')
    


sys.exit()
