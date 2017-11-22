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
    #write_json_to_pathfile(x,"D:\\result.arff")

# Como se nos pide 100 instancias (con 0 y 1), creamos las 84 restantes con los siguientes métodos:

X1 = create_x_instances_to_list(84,X1)
X2 = create_x_instances_to_list(84,X2)
X3 = create_x_instances_to_list(84,X3)
X4 = create_x_instances_to_list(84,X4)
X5 = create_x_instances_to_list(84,X5)

# Y = X1 XOR X2 XOR X3 (Problema XOR con 3 dimensiones)
# Se cumple la propiedad conmutativa.
Y = change_bool_to_int(list(np.logical_xor(X1,np.logical_xor(X2,X3))))
pt("Y", Y)
# Unimos listas en columnas para weka
data = create_columns_per_list()



def calculate_subsets_sx(array, possibilities, Y):
    s1, s2, s3, s4, s5 = [], [], [], [], []
    sy1, sy2, sy3, sy4, sy5 = [], [], [], [], []
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
                sy1.append(list(row[1:5])+[y])
                if p1_X1 is False:
                    if row[0] == possibilities[0] or row[0] == possibilities[1]:
                        p1_X1 = True
            elif string_feature == 'x2':
                s2.append(list(np.hstack([row[0], row[2:5]])))
                sy2.append(list(np.hstack([row[0], row[2:5]]))+[y])
                if p1_X2 is False:
                    if row[1] == possibilities[0] or row[1] == possibilities[1]:
                        p1_X2 = True
            elif string_feature == 'x3':
                s3.append(list(np.hstack([row[0:2], row[3:5]])))
                sy3.append(list(np.hstack([row[0:2], row[3:5]]))+[y])
                if p1_X3 is False:
                    if row[2] == possibilities[0] or row[2] == possibilities[1]:
                        p1_X3 = True
                # pt(str(i + 1), sx)
            elif string_feature == 'x4':
                s4.append(list(np.hstack([row[0:3], row[4]])))
                sy4.append(list(np.hstack([row[0:3], row[4]]))+[y])
                if p1_X4 is False:
                    if row[3] == possibilities[0] or row[3] == possibilities[1]:
                        p1_X4 = True
            elif string_feature == 'x5':
                s5.append(list(row[0:4]))
                sy5.append(list(row[0:4])+[y])
                if p1_X5 is False:
                    if row[4] == possibilities[0] or row[4] == possibilities[1]:
                        p1_X5 = True
    return all_rows, s1, s2, s3, s4, s5, sy1, sy2, sy3, sy4, sy5


def create_dict_p2():
    dict_p2 = {}
    possibilities = [[0,0],[0,1],[1,0],[1,1]]
    for x, y in possibilities:
        dict_p2[x,y] = []
    #pt("dict_p2",dict_p2)
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
    all_rows, s1, s2, s3, s4, s5, sy1, sy2, sy3, sy4, sy5 = calculate_subsets_sx(array_data, possibilities,Y)
    #calculate_strong_relevants(all_rows,y,possibilities)
    # Todo si para débil debe coger subconjuntos
    # Count numbers of times apper
    dict_p2_x1 = create_dict_p2() # {[x,y]:[[0000][0001]...]}
    dict_p2_x2 = create_dict_p2() # {[x,y]:[[0000][0001]...]}
    dict_p2_x3 = create_dict_p2() # {[x,y]:[[0000][0001]...]}
    dict_p2_x4 = create_dict_p2() # {[x,y]:[[0000][0001]...]}
    dict_p2_x5 = create_dict_p2() # {[x,y]:[[0000][0001]...]}

    for string_feature, possibility in possibilities.items():
        row_count = 0
        #for row_count, row in enumerate(all_rows):
        for row in all_rows:
            y_ = row[-1]
            if string_feature == 'x1':
                x1 = row[0]
                dict_get = dict_p2_x1.get((x1,y_))
                if not dict_get: # First case
                    dict_get = row[1:5]
                    dict_p2_x1[x1,y_] = [dict_get]
                else:
                    dict_get.append(row[1:5])
                    dict_p2_x1[x1, y_] = dict_get
                row_count += 1
            elif string_feature == 'x2':
                x2 = row[1]
                dict_get = dict_p2_x2.get((x2,y_))
                if not dict_get: # First case
                    dict_get = list(np.hstack([row[0], row[2:5]]))
                    dict_p2_x2[x2,y_] = [dict_get]
                else:
                    dict_get.append(list(np.hstack([row[0], row[2:5]])))
                    dict_p2_x2[x2, y_] = dict_get
                row_count += 1
            elif string_feature == 'x3':
                x3 = row[2]
                dict_get = dict_p2_x3.get((x3,y_))
                if not dict_get: # First case
                    dict_get = list(np.hstack([row[0:2], row[3:5]]))
                    dict_p2_x3[x3,y_] = [dict_get]
                else:
                    dict_get.append(list(np.hstack([row[0:2], row[3:5]])))
                    dict_p2_x3[x3, y_] = dict_get
                row_count += 1
            elif string_feature == 'x4':
                x4 = row[1]
                dict_get = dict_p2_x4.get((x4,y_))
                if not dict_get: # First case
                    dict_get = list(np.hstack([row[0:2], row[3:5]]))
                    dict_p2_x4[x4,y_] = [dict_get]
                else:
                    dict_get.append(list(np.hstack([row[0:2], row[3:5]])))
                    dict_p2_x4[x4, y_] = dict_get
                row_count += 1
            elif string_feature == 'x5':
                x5 = row[1]
                dict_get = dict_p2_x5.get((x5,y_))
                if not dict_get: # First case
                    dict_get = list(row[0:4])
                    dict_p2_x5[x5,y_] = [dict_get]
                else:
                    dict_get.append(list(row[0:4]))
                    dict_p2_x5[x5, y_] = dict_get
                row_count += 1
    pt("dict_p2_x1", dict_p2_x1)
    pt("dict_p2_x2", dict_p2_x2)
    pt("dict_p2_x3", dict_p2_x3)
    pt("dict_p2_x4", dict_p2_x4)
    pt("dict_p2_x5", dict_p2_x5)

    # ----------------------------------------------------
    # ----------------------------------------------------
    # ----------------------------------------------------
    # TO calculate p1 and p2 probabilities
    # p1 = p(Xi=xi,Si=si)>0
    # p2 = p(Y=y | Xi=xi, Si=si)
    p2x1, p2x2, p2x3, p2x4, p2x5 = 0., 0., 0. ,0. ,0.
    for string_feature, possibility in possibilities.items():
        y_counts = 0
        for i, row in enumerate(all_rows):
            y_ = row[-1]
            # X1
            if string_feature == 'x1':
                si = row[1:5]
                number_of_times = s1.count(si)
                if number_of_times > 0:
                    # delvolver los indices que coinciden con el count
                    indexs = []
                    for index, s in enumerate(sy1):
                        if si == s[0:4]:
                            indexs.append(index)
                    for index_ in indexs:
                        y_1 = sy1[index_][-1]
                        if y_ == y_1:
                            y_counts += 1
                if y_counts > 0:
                    p2x1 = number_of_times/y_counts
            # X2
            elif string_feature == 'x2':
                si = list(np.hstack([row[0], row[2:5]]))
                number_of_times = s2.count(si)
                if number_of_times > 0:
                    # delvolver los indices que coinciden con el count
                    indexs = []
                    for index, s in enumerate(sy2):
                        if si == s[0:4]:
                            indexs.append(index)
                    for index_ in indexs:
                        y_1 = sy2[index_][-1]
                        if y_ == y_1:
                            y_counts += 1
                if y_counts > 0:
                    p2x2 = number_of_times/y_counts
            # X3
            elif string_feature == 'x3':
                si = list(np.hstack([row[0:2], row[3:5]]))
                number_of_times = s3.count(si)
                if number_of_times > 0:
                    # delvolver los indices que coinciden con el count
                    indexs = []
                    for index, s in enumerate(sy3):
                        if si == s[0:4]:
                            indexs.append(index)
                    for index_ in indexs:
                        y_1 = sy3[index_][-1]
                        if y_ == y_1:
                            y_counts += 1
                if y_counts > 0:
                    p2x3 = number_of_times / y_counts
            # X4
            elif string_feature == 'x4':
                si = list(np.hstack([row[0:2], row[3:5]]))
                number_of_times = s4.count(si)
                if number_of_times > 0:
                    # delvolver los indices que coinciden con el count
                    indexs = []
                    for index, s in enumerate(sy4):
                        if si == s[0:4]:
                            indexs.append(index)
                    for index_ in indexs:
                        y_1 = sy4[index_][-1]
                        if y_ == y_1:
                            y_counts += 1
                if y_counts > 0:
                    p2x4 = number_of_times / y_counts
            # X5
            elif string_feature == 'x5':
                si = list(row[0:4])
                number_of_times = s5.count(si)
                if number_of_times > 0:
                    # delvolver los indices que coinciden con el count
                    indexs = []
                    for index, s in enumerate(sy5):
                        if si == s[0:4]:
                            indexs.append(index)
                    for index_ in indexs:
                        y_1 = sy5[index_][-1]
                        if y_ == y_1:
                            y_counts += 1
                if y_counts > 0:
                    p2x5 = number_of_times / y_counts
    pt("p2x1",p2x1)
    pt("p2x2",p2x2)
    pt("p2x3",p2x3)
    pt("p2x4",p2x4)
    pt("p2x5",p2x5)
    #------------------------------------------------------
    # ----------------------------------------------------
    # ----------------------------------------------------

    # p3 = p(Y=y | Si=si)
    p3x1, p3x2 , p3x3, p3x4, p3x5 = 0, 0, 0, 0, 0
    # TODO p3


    # Dataset, column, y, x


def calculate_strong_relevant(dictionary):
    """
    dict_p2_x1 = {(x1,y): [x2,x3,x4,x5]}
    """
    # p2 = p(Y=y | Xi=xi, Si=si)
    # p3 = p(Y=y | Si=si)
    is_strong = False
    y_count = 0
    si_count = 0
    for xy, si in dictionary.items():
        if xy[0] == 0:
            if si:
                y_count = len(si)
        if xy[0] == 1:
            if si:
                y_count = len(si)






possibilities = {'x1':[0,1], 'x2':[0,1], 'x3':[0,1], 'x4':[0,1], 'x5':[0,1]}
#possibilities = ['x1', 'x2', 'x3', 'x4', 'x5' ]
find_relevants_information(data=data, Y=Y, possibilities=possibilities)
# Creamos archivo arff
#convert_lists_to_arff(create_columns_per_list())