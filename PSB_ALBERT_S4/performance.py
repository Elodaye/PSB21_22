import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf

import sklearn.metrics.classification

print('hello')
import sys




def get_class(filename):
    """
    Cette fonction retourne un dictionnaire des différentes classes issues du fichier listClass
    :param filename: Le nom du fichier texte contenant au moins 1 classe
    :return: un dictionnaire contenant les classes ex: {0: 'car', 1: 'moto', 2: 'pie', 3: 'chien', 4: 'voix'}
    """

    try:  # Ouverture du fichier texte en lecture seule
        f = open(filename)
    except IOError as e:  # En cas d’exception , afficher un message d’erreur
        print("Ouverture du fichier impossible \n", e)
        sys.exit(1)  # et quitter le programme

    # Lire le fichier sous forme de liste de lignes
    lignes = f.readlines()

    # Création du dictionnaire
    i = 0 # Indice de la clé
    dict_class = {} # Dictionnaire vide

    for ligne in lignes[1:]:
        dict_class[i] = ligne.split(",")[0]
        i+=1

    return dict_class

def get_prediction(filename):
    """

    :param filename_perf:
    :return:
    """
    try:  # Ouverture du fichier texte en lecture seule
        f = open(filename)
    except IOError as e:  # En cas d’exception , afficher un message d’erreur
        print("Ouverture du fichier impossible \n", e)
        sys.exit(1)  # et quitter le programme

    # Lire le fichier sous forme de liste de lignes
    lignes = f.readlines()

    y_true = []
    y_pred = []

    for ligne in lignes:
        sep = ligne.split(" ")
        true_class = sep[0][1:-1].split("'")

        try:
            while True:
                true_class.remove('')
        except ValueError:
            pass
        try:
            while True:
                true_class.remove(",")
        except ValueError:
            pass

        pred_class = sep[1].rstrip("\n")[2:-2].split(",")

        if pred_class == ['elo']:
            pred_class = ['velo']
        if pred_class == ['oix']:
            pred_class = ['voix']
        if pred_class == ['ar']:
            pred_class = ['car']
        if pred_class == ['ie']:
            pred_class = ['pie']
        if pred_class == ['hien']:
            pred_class = ['chien']
        if pred_class == ['oto']:
            pred_class = ['moto']
        if pred_class == ['ouette']:
            pred_class = ['mouette']

        if ',' in pred_class[0]:
            print(True)


        y_true.append(true_class)
        y_pred.append(pred_class)

    return y_true, y_pred

def str_to_nbe(y, dict):
    """
    :param y: list de string, y_pred ou y_true ex : [['car'], ['pie', 'chien', 'car'], ['car'], ['car'], ['pie']]
    :param dict: dictionnaire des classes possibles ex : {['car']:1, ['chien']:2, ['car', 'chien']:3}
    :return: y: list d'entiers entre [0 et len(dict)]
    """
    y_return = []
    for l in y:
        y_return.append(dict[tuple(l)])
    return y_return

def create_dic_from_y(y_t, y_p):
    dic = {tuple(y_t[0]):0}
    i = 1
    if len(y_t)>1:
        for l in y_t[1:]:
            try:
                if tuple(l) in list(dic.keys()):
                    pass
                else:
                    dic[tuple(l)] = i
                    i += 1
            except KeyError:
                pass
        for l in y_p:
            try:
                if tuple(l) in list(dic.keys()):
                    pass
                else:
                    dic[tuple(l)] = i
                    i += 1
            except KeyError:
                pass
    return dic


if __name__ == '__main__':

#    filename_lsCl = "listClass.txt"
#    dict_cls = get_class(filename_lsCl)

    filename_perf = "donnees_pour_performances(2)"
    y_true, y_pred = get_prediction(filename_perf)
    print("y_true: ", y_true)
    print("y_pred: ",y_pred)

    # les matrices de confusion ne supportent pas les string
    # il faut faire une fonction qui accorde un numéro à chaque prédiction

    dic = create_dic_from_y(y_true, y_pred)
    print("dictionnaire classes", dic)
    y_true_nb = str_to_nbe(y_true, dic)
    y_pred_nb = str_to_nbe(y_pred, dic)

    from sklearn.metrics import confusion_matrix
    labels=[i for i in range(0, len(list(dic.keys())))]
    conf_mx = confusion_matrix(y_true_nb, y_pred_nb, labels=labels)
    print("matrice de confusion \n", conf_mx)

    #filename_perf = "donnees_pour_performances(2)"
    #filename_perf = "donnees_pour_performances_2_1"
    filename_perf = "donnees_pour_performances_4"
    y_true, y_pred = get_prediction(filename_perf)
    print("y_true: ", y_true)
    print("y_pred: ", y_pred)

    # les matrices de confusion ne supportent pas les string
    # il faut faire une fonction qui accorde un numéro à chaque prédiction

    dic = create_dic_from_y(y_true, y_pred)
    print("dictionnaire classes", dic)
    y_true_nb = str_to_nbe(y_true, dic)
    y_pred_nb = str_to_nbe(y_pred, dic)

    from sklearn.metrics import confusion_matrix

    labels = [i for i in range(0, len(list(dic.keys())))]

    print(dic)
    conf_mx = confusion_matrix(y_true_nb, y_pred_nb, labels=labels)
    print("matrice de confusion \n", conf_mx)

    import matplotlib.pyplot as plt
    import tensorflow as tf
    import seaborn as sns

    #confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mx, annot=True, fmt="d")
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()
    print(np.sum(conf_mx))
