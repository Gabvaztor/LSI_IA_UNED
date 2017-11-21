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

# X4 = X2 XOR X3
#X4 = np.logical_xor(X2,X3)
# Vamos a trabajar de tal forma que X1 sea fuertemente relevante, es decir, que si se elimina esa característica
# por sí sola, va a deteriorar el rendimiento del clasificador de Bayes óptimo.

# Como se nos pide 100 instancias (con 0 y 1), creamos las 84 restantes con el siguiente método:
def create_x_instances_to_list(number_of_instances=84, list_to_add=None):
    new_list = list(np.random.randint(2, size=number_of_instances))
    list_to_return = list_to_add + new_list
    # Necesario para que no sean valores numpy
    x = change_bool_to_int(list_to_return)
    return x

def change_bool_to_int(list):
    return [int(elem) for elem in list]

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
# Y = X1 XOR X2 XOR X3 (Problema XOR con 3 dimensiones)
# Se cumple la propiedad conmutativa.
Y = list(np.logical_xor(X1,np.logical_xor(X2,X3)))
Y = change_bool_to_int(Y)
pt("Y", Y)
# Unimos listas en columnas para weka
data = create_columns_per_list()



def calculate_subsets_sx(array, possibilities, Y):
    s1, s2, s3, s4, s5 = [], [], [], [], []
    all_rows = []
    p1_X1, p1_X2, p1_X3, p1_X4, p1_X5 = False, False , False, False, False
    add_to_all_rows = True
    for string_feature, possibilities in possibilities.items():
        for i in range(len(data)):
            row = array[i]
            y = Y[i]
            if add_to_all_rows:
                to_add = list(row)+[y]
                all_rows.append(to_add)
            if len(all_rows) == 100:
                add_to_all_rows = False
            if string_feature == 'x1':
                s1.append(list(row[1:5]))
                if p1_X1 is False:
                    if row[0] == possibilities[0] or row[0] == possibilities[1]:
                        p1_X1 = True
            elif string_feature == 'x2':
                s2.append(list(np.hstack([row[0], row[2:5]])))
                if p1_X2 is False:
                    if row[1] == possibilities[0] or row[1] == possibilities[1]:
                        p1_X2 = True
            elif string_feature == 'x3':
                s3.append(list(np.hstack([row[0:2], row[3:5]])))
                if p1_X3 is False:
                    if row[2] == possibilities[0] or row[2] == possibilities[1]:
                        p1_X3 = True
                # pt(str(i + 1), sx)
            elif string_feature == 'x4':
                s4.append(list(np.hstack([row[0:3], row[4]])))
                if p1_X4 is False:
                    if row[3] == possibilities[0] or row[3] == possibilities[1]:
                        p1_X4 = True
            elif string_feature == 'x5':

                s5.append(list(row[0:4]))
                if p1_X5 is False:
                    if row[4] == possibilities[0] or row[4] == possibilities[1]:
                        p1_X5 = True
    return all_rows, s1, s2, s3, s4, s5, p1_X1, p1_X2, p1_X3, p1_X4, p1_X5


def calculate_count_p2(y_to_analize, x1, data_set):
    pass


def calculate_strong_relevants(data_set, y_to_analize, possibilities_of_xi):
    # p2 = p(Y=y | Xi=xi, Si=si)
    # p3 = p(Y=y | Si=si)
    count_p2 = 0
    count_p3 = 0
    p2 = None
    p3 = None
    strong_relevant = []
    for string_feature, possibilities in possibilities_of_xi.items():
        for possibility in possibilities:
            if string_feature == 'x1':
                x1 = possibility
                all_rows_with_x1_possibility = find_all_rows_with_x1_possibility(0,)
                count_p2 = calculate_count_p2(y_to_analize,x1,data_set)
            elif string_feature == 'x2':
                pass
            elif string_feature == 'x3':
                pass
            elif string_feature == 'x4':
                pass
            elif string_feature == 'x5':
                pass


def create_dict_p2():
    dict_p2 = {}
    possibilities = [[0,0],[0,1],[1,0],[1,1]]
    for x, y in possibilities:
        dict_p2[x,y] = []
    pt("dict_p2",dict_p2)
    return dict_p2


def find_relevants_information(data, Y, possibilities):
    """
    Encuentra los datos que son relevantes fuertes, débiles o irrelevantes.
    """
    # p1 = p(Xi=xi,Si=si)>0 para all xi
    # p2 = p(Y=y | Xi=xi, Si=si)
    # p3 = p(Y=y | Si=si)

    irrelevant = []
    strong_relevant = []
    weak_relevant = []
    pt(np.asarray(data).shape)
    pt(len(data[0]))
    array_data = np.asarray(data)
    # All subsets sx
    all_rows, s1, s2, s3, s4, s5, p1_X1, p1_X2, p1_X3, p1_X4, p1_X5 = calculate_subsets_sx(array_data, possibilities,Y)
    y=[0,1]
    #calculate_strong_relevants(all_rows,y,possibilities)

    # Todo si para débil debe coger subconjuntos
    # Count numbers of times apper
    x1_ = True
    x2_ = True
    x3_ = True
    x4_ = True
    x5_ = True
    dict_p2 = create_dict_p2() # {[x,y]:[[0000][0001]...]}
    for string_feature, possibilities in possibilities.items():
        row_count = 0
        for row in all_rows:
            if string_feature == 'x1' and p1_X1 and x1_:
                y_ = row.pop()
                x1 = s1[row_count][0]
                dict_get = dict_p2.get((x1,y_))
                pt("dic", dict_get)
                dict_get2 = dict_get.append(s1[row_count])
                pt("s", s1[row_count])
                asf
                # TODO Finish this
                dict_p2[x1,y_] = to_append
                if row_count == 3:
                    pt("update dict", dict_p2)
                    asd
                row_count += 1
            elif string_feature == 'x2' and p1_X2 and x2_:
                pass
            elif string_feature == 'x3' and p1_X3 and x3_:
                pass
                # pt(str(i + 1), sx)
            elif string_feature == 'x4' and p1_X4 and x4_:
                pass
            elif string_feature == 'x5' and p1_X5 and x5_:
                pass

possibilities = {'x1':[0,1], 'x2':[0,1], 'x3':[0,1], 'x4':[0,1], 'x5':[0,1]}

find_relevants_information(data=data, Y=Y, possibilities=possibilities)
# Creamos archivo arff
#convert_lists_to_arff(create_columns_per_list())