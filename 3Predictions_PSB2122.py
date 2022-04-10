import numpy as np
import matplotlib.pyplot as plt
import os

import random
import sklearn
import torch
import torchaudio
from keras.models import load_model
from torchaudio import transforms
from scipy import signal
from scipy.io import wavfile
from keras import layers
from keras import models
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
from tqdm import tqdm
from PIL import ImageFile
import json
import copy


model = load_model('Pretrained_128_20.h5')

print("-----------------------Réalisation de prédictions par le modèle entraîné--------------------------------")

test_datagen = ImageDataGenerator(rescale=1. / 255.)
df_0 = pd.read_csv("C:/Users/Utilisateur/Documents/ENSTA/2A/UE 3.4/Projet système/Machine_learning/Donnees_label/mes_datas.txt", encoding='latin-1')
df_0["labels"] = df_0["labels"].apply(lambda x: x.split(", "))
df = df_0
#df = sklearn.utils.shuffle(df_0)  # TODO attention !!!
height,length = 1000,1000
test_size = 1
chemin = "C://Users//Utilisateur//Documents//ENSTA//2A//UE 3.4//Projet système//Machine_learning//Donnees_label//"
chemin_bis = "C:/Users/Utilisateur/Documents/ENSTA/2A/UE 3.4/Projet système/Machine_learning/Donnees_label//listClass.txt"

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df[::],
    #directory=test_dir,
    x_col="path",
    y_col="labels",
    color_mode='grayscale',
    target_size=(height,length),
    batch_size=test_size,
    class_mode= 'categorical',
    classes=None)

STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

test_generator.reset()   # Reset obligatoire pour avoir le bon ordre des outputs

pred = model.predict(test_generator, steps=STEP_SIZE_TEST, verbose=1)

# Interupteur pour lever la prediction si on a au moins 10% de confidence
booleanPrediction = (pred > 0.5)
print ("pred",pred)  # la matrice des probabilités
print (booleanPrediction, "booleanPrediction") # la matrice des indices représentants les sons estimés présents (si True)

f2 = open(chemin + "donnees_pour_carte", "w+")
for i in range (2095):
    f2.write("x : dans le titre de l image {}".format(i)+ ", " + "y: dans le titre de l image {}".format(i) + ", " + "t : dans le titre de l'image {}".format(i) + ", " + str (booleanPrediction[i]) + ", " + "l intensité maximale, a recueillir\n" )
f2.close()


listPrediction = []

# On récupere le dictionnaire JSON des labels avec leur indice respectif
labelsList = json.load(open(chemin + "classIndice.txt"))
labelsList = dict((v, k) for k, v in labelsList.items())

for img in booleanPrediction:
    print (img) # pour une image, les sons prédits comme présents
    correctPredictList = []
    for index, cls in enumerate(img):
        if cls:
            correctPredictList.append(labelsList[index])
            print(cls, "jcccccp")

    listPrediction.append(",".join(correctPredictList))  # on enregistre les prédictions effectuées pour chaque image

print (listPrediction)
print ("laaaaaaaaaaaaaaaaaaaaaaaaaaaa" , df[:]["labels"])

# Tableau contenant l'ensemble des chemin des images
pathImg = test_generator._filepaths

# On creer un dataframe pour concat les images avec leurs predictions correctes
results = pd.DataFrame({"Chemin img": pathImg, "Predictions": listPrediction})
#print (results)


predict = open(chemin + "predictions.txt", "w+")
if os.path.getsize(chemin + "predictions.txt") == 0:
    predict.write("path,labels_pred\n")

f3 = open(chemin + "donnees_pour_performances_pretrained", "w+")
# TODO 104 est à modifier selon le nb d'échantillons
for i in range(2095): #str(df.iat[nb_train + nb_valid + i, 0] ) + ", " +
    f3.write(str(df.iat[i, 1] ) + " " + str([ listPrediction[i]]) +  "\n")
    predict.write (pathImg[i] + "," + str( listPrediction[i]) +  "\n" )
    #df[nb_train + nb_valid + i: nb_train + nb_valid + i + 1]["labels"]
f3.close()


###  affichage des performances
# acc = model.history.history['accuracy'] # précision à l'entrainement
# val_acc = model.history.history['val_acc'] # précision sur les données de test
# loss = model.history.history['loss'] # erreur
# val_loss = model.history.history['val_loss'] # erreur sur les données de

#
# plt.plot(acc,'bo', label='Training acc')
# plt.plot(val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()
# plt.plot(loss,'bo', label='Training loss')
# plt.plot(val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()
