"""
Práctica a realizar:

El estudiante generará un conjunto de datos artificial compuesto por 100 instancias caracterizadas por una variable
relevante en sentido fuerte, tres variables relevantes en sentido débil y una variable totalmente irrelevante. Esta 
última se puede generar mediante números aleatorios extraídos de una distribución de probabilidad uniforme o normal
(gaussiana). Como indicación sugerimos extender el ejemplo XOR a tres dimensiones. A continuación, aplicará
diferentes técnicas de selección de variables disponibles en weka (un mínimo de tres de filtrado, el análisis
de componentes principales y la técnica de envoltura, WrapperSubsetEval, con BayesNet como clasificador y 
empleando todos los valores por defecto, salvo el número máximo de padres que se debe modificar a 3).

"""

import numpy as np
import arff
from UtilsFunctions import *
# Crear 100 instancias de datos para S={X1,X2,X3,X4,X5} con valores Booleanos
# Existen 16 posibles instancias si extendemos el problema XOR a tres dimensiones.
# Trabajamos con el siguiente conjunto:
X1 = [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]
X2 = [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]
X3 = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
X4 = [0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0]
X5 = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
# Y = X1 XOR X2 XOR X3 (Problema XOR con 3 dimensiones)
# Se cumple la propiedad conmutativa.
Y = np.logical_xor(X1,np.logical_xor(X2,X3))

# X4 = X2 XOR X3
#X4 = np.logical_xor(X2,X3)
# Vamos a trabajar de tal forma que X1 sea fuertemente relevante, es decir, que si se elimina esa característica
# por sí sola, va a deteriorar el rendimiento del clasificador de Bayes óptimo.

# Como se nos pide 100 instancias (con 0 y 1), creamos las 84 restantes con el siguiente método:
def create_x_instances_to_list(number_of_instances=84, list_to_add=None):
    new_list = list(np.random.randint(2, size=number_of_instances))
    list_to_return = list_to_add + new_list
    # Necesario para que no sean valores numpy
    x = [int(elem) for elem in list_to_return]
    return x

def create_columns_per_list():
    """
    Unimos los datos de  
    """
    data = list(zip(X1, X2, X3, X4, X5))
    return data

def convert_lists_to_arff(lists):
    """
    # Convertimos los datos a .arff para poder leerlos en weka con la siguiente función
    """
    x = arff.dumps(lists,
                   relation="arff",
                   names=['X1','X2','X3','X4','X5'])
    pt(x)
    write_json_to_pathfile(x,"D:\\result.arff")

# Como se nos pide 100 instancias (con 0 y 1), creamos las 84 restantes con los siguientes métodos:
X1 = create_x_instances_to_list(84,X1)
X2 = create_x_instances_to_list(84,X2)
X3 = create_x_instances_to_list(84,X3)
X4 = create_x_instances_to_list(84,X4)
X5 = create_x_instances_to_list(84,X5)
# Unimos listas en columnas para weka
data = create_columns_per_list()

def find_relevants_information(data, y):
    """
    Encuentra los datos que son relevantes fuertes, débiles o irrelevantes.
    """
    # p1 = p(Xi=xi,Si=si)>0
    # p2 = p(Y=y | Xi=xi, Si=si)
    # p3 = p(Y=y | Si=si)
    pt(np.asarray(data).shape)
    pt(len(data[0]))
    array = np.asarray(data)
    for i in range(len(data)):
        row = array[i]


find_relevants_information(data=data, y=Y)
# Creamos archivo arff
#convert_lists_to_arff(create_columns_per_list())