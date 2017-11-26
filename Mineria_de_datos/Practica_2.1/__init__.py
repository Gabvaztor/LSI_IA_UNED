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
import pandas as pd
# Crear 100 instancias de datos para S={X1,X2,X3,X4,X5} con valores Booleanos
# Existen 16 posibles instancias si extendemos el problema XOR a tres dimensiones.
# Trabajamos con el siguiente conjunto:
#X1 = [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]
#X2 = [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]
#X3 = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
#X4 = [0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0]
#X5 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
X1 = []
X2 = []
X3 = []
#X4 = []
#X5 = []
strong = 3  # If a Xi is strong
weak = 2  # If a Xi is weak
irrelevant = 1  # If a Xi is irrelevant
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
    data = list(zip(X1, X2, X3, X4, X5, Y))
    return data

def convert_lists_to_arff(lists):
    """
    # Convertimos los datos a .arff para poder leerlos en weka con la siguiente función
    """
    x = arff.dumps(lists,
                   relation="arff",
                   names=['X1','X2','X3','X4','X5','Y'])
    pt(x)
    write_json_to_pathfile(x,"D:\\result.arff")

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
    """
    pt("dict_p2_x1", dict_p2_x1)
    pt("dict_p2_x2", dict_p2_x2)
    pt("dict_p2_x3", dict_p2_x3)
    pt("dict_p2_x4", dict_p2_x4)
    pt("dict_p2_x5", dict_p2_x5)
    """
    # ----------------------------------------------------
    # ----------------------------------------------------
    # ----------------------------------------------------
    # TO calculate p1 and p2 probabilities
    # p1 = p(Xi=xi,Si=si)>0
    # p3 = p(Y=y | Si=si)
    p3x1y0, p3x1y1,\
    p3x2y0, p3x2y1,\
    p3x3y0, p3x3y1,\
    p3x4y0, p3x4y1,\
    p3x5y0, p3x5y1 = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    y_x1_values = []
    y_x2_values = []
    y_x3_values = []
    y_x4_values = []
    y_x5_values = []
    for string_feature, possibility in possibilities.items():
        y_counts = 0
        for i, row in enumerate(all_rows):
            y_ = row[-1]
            # X1
            if string_feature == 'x1' and y_ not in y_x1_values:
                y_x1_values.append(y_)
                si = s1[i]
                number_of_times = s1.count(si)
                pt("si", si)
                pt("number_of_times", number_of_times)
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
                    if y_ == 0:
                        p3x1y0 = number_of_times/y_counts
                    elif y_ == 1:
                        p3x1y1 = number_of_times/y_counts
                else:
                    if y_ == 0:
                        p3x1y0 = 0
                    elif y_ == 1:
                        p3x1y1 = 0
            # X2
            elif string_feature == 'x2' and y_ not in y_x2_values:
                y_x2_values.append(y_)
                #si = list(np.hstack([row[0], row[2:5]]))
                si = s2[i]
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
                    if y_ == 0:
                        p3x2y0 = number_of_times/y_counts
                    elif y_ == 1:
                        p3x2y1 = number_of_times/y_counts
                else:
                    if y_ == 0:
                        p3x2y0 = 0
                    elif y_ == 1:
                        p3x2y1 = 0
            # X3
            elif string_feature == 'x3' and y_ not in y_x3_values:
                y_x3_values.append(y_)
                #si = list(np.hstack([row[0:2], row[3:5]]))
                si = s3[i]
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
                    if y_ == 0:
                        p3x3y0 = number_of_times/y_counts
                    elif y_ == 1:
                        p3x3y1 = number_of_times/y_counts
                else:
                    if y_ == 0:
                        p3x3y0 = 0
                    elif y_ == 1:
                        p3x3y1 = 0
            # X4
            elif string_feature == 'x4' and y_ not in y_x4_values:
                y_x4_values.append(y_)
                #si = list(np.hstack([row[0:2], row[3:5]]))
                si = s4[i]
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
                    if y_ == 0:
                        p3x4y0 = number_of_times/y_counts
                    elif y_ == 1:
                        p3x4y1 = number_of_times/y_counts
                else:
                    if y_ == 0:
                        p3x4y0 = 0
                    elif y_ == 1:
                        p3x4y1 = 0
            # X5
            elif string_feature == 'x5' and y_ not in y_x5_values:
                y_x5_values.append(y_)
                #si = list(row[0:4])
                si = s5[i]
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
                    if y_ == 0:
                        p3x5y0 = number_of_times/y_counts
                    elif y_ == 1:
                        p3x5y1 = number_of_times/y_counts
                else:
                    if y_ == 0:
                        p3x5y0 = 0
                    elif y_ == 1:
                        p3x5y1 = 0

    #------------------------------------------------------
    # ----------------------------------------------------
    # ----------------------------------------------------



    # p2 = p(Y=y | Xi=xi, Si=si)
    p2x1_0y_0, p2x1_1y_0, p2x1_0y_1, p2x1_1y_1 = 0, 0, 0 ,0
    p2x2_0y_0, p2x2_1y_0, p2x2_0y_1, p2x2_1y_1 = 0, 0, 0 ,0
    p2x3_0y_0, p2x3_1y_0, p2x3_0y_1, p2x3_1y_1 = 0, 0, 0 ,0
    p2x4_0y_0, p2x4_1y_0, p2x4_0y_1, p2x4_1y_1 = 0, 0, 0 ,0
    p2x5_0y_0, p2x5_1y_0, p2x5_0y_1, p2x5_1y_1 = 0, 0, 0 ,0

    x1_y_values = []
    x2_y_values = []
    x3_y_values = []
    x4_y_values = []
    x5_y_values = []
    for string_feature, possibility in possibilities.items():
        xy_counts = 0
        x_si_counts = 0
        for i, row in enumerate(all_rows):
            y_ = row[-1]
            x = row[0]
            to_check = [x,y_]
            # X1
            if string_feature == 'x1' and to_check not in x1_y_values:
                x1_y_values.append(to_check)
                si = s1[i]
                number_of_times = s1.count(si)
                if number_of_times > 0:
                    indexs_x = [] # Contiene todos los indices donde está el xi condicional
                    for index_x, row_x in enumerate(all_rows):
                        if row_x[0] == x:
                            indexs_x.append(index_x)
                    sys = []
                    for index in indexs_x:
                        sys.append(sy1[index])
                    for siy in sys:
                        y_s = siy[-1]
                        s_i = siy[0:4]
                        if y_s == y_ and s_i == si:
                            xy_counts += 1
                        if s_i == si:
                            x_si_counts += 1
                    if xy_counts > 0:
                        if x == 0 and y_ == 0:
                            p2x1_0y_0 = xy_counts/x_si_counts
                        elif x == 0 and  y_ == 1:
                            p2x1_0y_1 = xy_counts / x_si_counts
                        elif x == 1 and y_ == 0 :
                            p2x1_1y_0 = xy_counts / x_si_counts
                        elif x == 1 and y_ == 1:
                            p2x1_1y_1 = xy_counts / x_si_counts
            # X2
            elif string_feature == 'x2' and to_check not in x2_y_values:
                x2_y_values.append(to_check)
                si = s2[i]
                number_of_times = s2.count(si)
                if number_of_times > 0:
                    indexs_x = [] # Contiene todos los indices donde está el xi condicional
                    for index_x, row_x in enumerate(all_rows):
                        if row_x[0] == x:
                            indexs_x.append(index_x)
                    sys = []
                    for index in indexs_x:
                        sys.append(sy2[index])
                    for siy in sys:
                        y_s = siy[-1]
                        s_i = siy[0:4]
                        if y_s == y_ and s_i == si:
                            xy_counts += 1
                        if s_i == si:
                            x_si_counts += 1
                    if xy_counts > 0:
                        if x == 0 and y_ == 0:
                            p2x2_0y_0 = xy_counts/x_si_counts
                        elif x == 0 and  y_ == 1:
                            p2x2_0y_1 = xy_counts / x_si_counts
                        elif x == 1 and y_ == 0 :
                            p2x2_1y_0 = xy_counts / x_si_counts
                        elif x == 1 and y_ == 1:
                            p2x2_1y_1 = xy_counts / x_si_counts
            # X3
            elif string_feature == 'x3' and to_check not in x3_y_values:
                x3_y_values.append(to_check)
                si = s3[i]
                number_of_times = s3.count(si)
                if number_of_times > 0:
                    indexs_x = [] # Contiene todos los indices donde está el xi condicional
                    for index_x, row_x in enumerate(all_rows):
                        if row_x[0] == x:
                            indexs_x.append(index_x)
                    sys = []
                    for index in indexs_x:
                        sys.append(sy3[index])
                    for siy in sys:
                        y_s = siy[-1]
                        s_i = siy[0:4]
                        if y_s == y_ and s_i == si:
                            xy_counts += 1
                        if s_i == si:
                            x_si_counts += 1
                    if xy_counts > 0:
                        if x == 0 and y_ == 0:
                            p2x3_0y_0 = xy_counts/x_si_counts
                        elif x == 0 and  y_ == 1:
                            p2x3_0y_1 = xy_counts / x_si_counts
                        elif x == 1 and y_ == 0 :
                            p2x3_1y_0 = xy_counts / x_si_counts
                        elif x == 1 and y_ == 1:
                            p2x3_1y_1 = xy_counts / x_si_counts
            # X4
            elif string_feature == 'x4' and to_check not in x4_y_values:
                x4_y_values.append(to_check)
                si = s4[i]
                number_of_times = s4.count(si)
                if number_of_times > 0:
                    indexs_x = [] # Contiene todos los indices donde está el xi condicional
                    for index_x, row_x in enumerate(all_rows):
                        if row_x[0] == x:
                            indexs_x.append(index_x)
                    sys = []
                    for index in indexs_x:
                        sys.append(sy4[index])
                    for siy in sys:
                        y_s = siy[-1]
                        s_i = siy[0:4]
                        if y_s == y_ and s_i == si:
                            xy_counts += 1
                        if s_i == si:
                            x_si_counts += 1
                    if xy_counts > 0:
                        if x == 0 and y_ == 0:
                            p2x4_0y_0 = xy_counts/x_si_counts
                        elif x == 0 and  y_ == 1:
                            p2x4_0y_1 = xy_counts / x_si_counts
                        elif x == 1 and y_ == 0 :
                            p2x4_1y_0 = xy_counts / x_si_counts
                        elif x == 1 and y_ == 1:
                            p2x4_1y_1 = xy_counts / x_si_counts
            # X5
            elif string_feature == 'x5' and to_check not in x5_y_values:
                x5_y_values.append(to_check)
                si = s5[i]
                number_of_times = s5.count(si)
                if number_of_times > 0:
                    indexs_x = [] # Contiene todos los indices donde está el xi condicional
                    for index_x, row_x in enumerate(all_rows):
                        if row_x[0] == x:
                            indexs_x.append(index_x)
                    sys = []
                    for index in indexs_x:
                        sys.append(sy5[index])
                    for siy in sys:
                        y_s = siy[-1]
                        s_i = siy[0:4]
                        if y_s == y_ and s_i == si:
                            xy_counts += 1
                        if s_i == si:
                            x_si_counts += 1
                    if xy_counts > 0:
                        if x == 0 and y_ == 0:
                            p2x5_0y_0 = xy_counts/x_si_counts
                        elif x == 0 and  y_ == 1:
                            p2x5_0y_1 = xy_counts / x_si_counts
                        elif x == 1 and y_ == 0 :
                            p2x5_1y_0 = xy_counts / x_si_counts
                        elif x == 1 and y_ == 1:
                            p2x5_1y_1 = xy_counts / x_si_counts

    pt("p3x1y0, p3x1y1, p3x2y0, p3x2y1, p3x3y0, p3x3y1, p3x4y0, p3x4y1, p3x5y0, p3x5y1",
       [p3x1y0, p3x1y1, p3x2y0, p3x2y1, p3x3y0, p3x3y1, p3x4y0, p3x4y1, p3x5y0, p3x5y1])

    pt("p2x1", [p2x1_0y_0, p2x1_1y_0, p2x1_0y_1, p2x1_1y_1])
    pt("p2x2", [p2x2_0y_0, p2x2_1y_0, p2x2_0y_1, p2x2_1y_1])
    pt("p2x3", [p2x3_0y_0, p2x3_1y_0, p2x3_0y_1, p2x3_1y_1])
    pt("p2x4", [p2x4_0y_0, p2x4_1y_0, p2x4_0y_1, p2x4_1y_1])
    pt("p2x5", [p2x5_0y_0, p2x5_1y_0, p2x5_0y_1, p2x5_1y_1])

def create_dict_from_data(X1, X2, X3, X4, X5, Y):
    dictionary = {}
    dictionary['X1'] = X1
    dictionary['X2'] = X2
    dictionary['X3'] = X3
    dictionary['X4'] = X4
    dictionary['X5'] = X5
    dictionary['Y'] = Y
    return dictionary


def is_strong(data, Xi,Y, y, x, Si,index):
    #pt("data",data)
    strong = False
    p2 = None
    p3 = None
    counts_y_xi = 0
    counts_y = 0
    counts_xi = 0
    counts_si = 1
    indexs = [index]
    for i, si in data.iterrows(): # Contamos las si que son iguales al Si de entrada sin tener en cuenta el de entrada
        if index != i: # Si no es el mismo si que estamos calculando
            if list(Si) == list(si):
                counts_si += 1
                indexs.append(i)
    indexs = list(set(indexs))
    for j in indexs:  # para todos los indices de todos los si que son iguales, contamos todos los xi que coinciden con el xi de entrada
        if x == Xi[j]:
            counts_xi +=1
    p1 = counts_xi/ Y.size #p1 = p(Xi=xi, Si=si)  Si existe para si y xi
    if p1 > 0:
        for e in indexs:
            if Y[e] == y and x == Xi[e]:
                counts_y_xi += 1
        p2 = counts_y_xi / counts_xi  # Número de veces que y es condicional a xi y a si entre el número de veces que Si=si y Xi=xi
        for r in indexs:
            if Y[r] == y:
                counts_y += 1
        p3 = counts_y/ len(indexs)
    if p2 is not None and p3 is not None:
        if p2 != p3:
            strong = True
    return strong


def calculate_strong_relevant(data, columns, x_y_possibilities):
    """
    """
    # p1 = p(Xi=xi,Si=si)>0 para all xi
    # p2 = p(Y=y | Xi=xi, Si=si)
    # p3 = p(Y=y | Si=si)
    Y = data.pop("Y")
    strong_relevant = []
    for column in columns:
        Xi = data.pop(column)
        change_column = False
        for x, y in x_y_possibilities:
            if change_column is False:
                for index, si in data.iterrows():
                    strong = is_strong(data,Xi,Y,y,x,si,index)
                    if strong:
                        pt(column + " is Strong")
                        strong_relevant.append(column)
                        change_column = True
                        break
            else:
                break
    return strong_relevant


def is_weak(data, Xi, Y, y, x, Si, index):
    weak = False
    p2 = None
    p3 = None
    counts_y_xi = 0
    counts_y = 0
    counts_xi = 0
    counts_si = 1
    indexs = [index]
    for i, si in data.iterrows():  # Contamos las si que son iguales al Si de entrada sin tener en cuenta el de entrada
        subsets_si = calcultate_subsets_si(si)
        for s_i_i in subsets_si:
            if index != i:  # Si no es el mismo si que estamos calculando
                if list(Si) == list(s_i_i):
                    counts_si += 1
                    indexs.append(i)
    indexs = list(set(indexs))
    for j in indexs:  # para todos los indices de todos los si que son iguales, contamos todos los xi que coinciden con el xi de entrada
        if x == Xi[j]:
            counts_xi += 1
    p1 = counts_xi / Y.size  # p1 = p(Xi=xi, Si=si)  Si existe para si y xi
    if p1 > 0:
        for e in indexs:
            if Y[e] == y and x == Xi[e]:
                counts_y_xi += 1
        p2 = counts_y_xi / counts_xi  # Número de veces que y es condicional a xi y a si entre el número de veces que Si=si y Xi=xi
        for r in indexs:
            if Y[r] == y:
                counts_y += 1
        p3 = counts_y / len(indexs)
    if p2 is not None and p3 is not None:
        if p2 != p3:
            weak = True
    return weak


def calcultate_subsets_si(si):
    from itertools import chain, combinations
    x = list(set(chain.from_iterable(combinations(si, n) for n in range(len(si) + 1))))
    to_return = []
    for i, element in enumerate(x):
        if not element:
            del x[i]
    for element in x:
        to_return.append(list(element))
    return to_return


def calculate_weak_relevant(data, columns, x_y_possibilities, strong_relevant):
    # p1 = p(Xi=xi,S'i=s'i)>0 para all xi
    # p2 = p(Y=y | Xi=xi, S'i=s'i)
    # p3 = p(Y=y | S'i=s'i)
    Y = data.pop("Y")
    weak_relevant = []
    for column in columns:
        if column not in strong_relevant:
            Xi = data.pop(column)
            change_column = False
            for x, y in x_y_possibilities:
                if change_column is False:
                    for index, si in data.iterrows():
                        if change_column is False:
                            subsets_si = calcultate_subsets_si(si)
                            for si_ in subsets_si:
                                weak = is_weak(data,Xi,Y,y,x,si_,index)
                                if weak:
                                    pt(column + " is Weak")
                                    weak_relevant.append(column)
                                    change_column = True
                                    break
                else:
                    break
    return weak_relevant
    


def find_relevants_information_pandas(dataset):
    import copy
    #pt("dataset", dataset)
    # Dataset, column, y, x
    dataframe = pd.DataFrame(data=dataset)
    #pt("dataframe",dataframe)
    columns = ['X1', 'X2', 'X3', 'X4', 'X5']
    x_y_possibilities = [[0, 0], [0, 1], [1, 0], [1, 1]]
    irrelevant = []
    data = copy.deepcopy(dataframe) # Create copy
    strong_relevant = calculate_strong_relevant(data=data,columns=columns,x_y_possibilities=x_y_possibilities)
    weak_relevant = calculate_weak_relevant(data=dataframe,columns=columns,x_y_possibilities=x_y_possibilities, strong_relevant=strong_relevant)
    for e in columns:
        if e not in strong_relevant+weak_relevant:
            irrelevant.append(e)
    pt("strong_relevant", strong_relevant)
    pt("weak_relevant", weak_relevant)
    pt("irrelevant", irrelevant)

# Como se nos pide 100 instancias (con 0 y 1), creamos las 84 restantes con los siguientes métodos:
X1 = create_x_instances_to_list(100,X1)
X2 = create_x_instances_to_list(100,X2)
X3 = create_x_instances_to_list(100,X3)
X4 = change_bool_to_int(list(np.logical_xor(X2,X3)))
X5 = [1] * 100
pt("X4", X4)
pt("X5", X5)
# Y = X1 XOR X2 XOR X3 (Problema XOR con 3 dimensiones)
# Se cumple la propiedad conmutativa.
Y = change_bool_to_int(list(np.logical_xor(X1,np.logical_xor(X2,X3))))
#pt("Y", Y)
# Unimos listas en columnas para weka
#data = create_columns_per_list()
dataset_dictionary = create_dict_from_data(X1,X2,X3,X4,X5,Y)
find_relevants_information_pandas(dataset=dataset_dictionary)

# Creamos archivo arff
#convert_lists_to_arff(create_columns_per_list())