import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.io import wavfile
from keras import layers
from keras import models
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
from tqdm import tqdm
from PIL import ImageFile
import json

def preparer_reps(src_audios, rep_dst, taille_wav):
    """
    Préparation du système de répertores de fichiers pour l'entrainement du CNN
    PLUS D'INTERET :param src_audios: répertoire avec les fichiers audio d'entrée. Il doit être composé de 2 dossiers car et notcar contenant les fichiers correspondants
    :param rep_dst: répertoire de destination où les fichiers d'entrainement vont être stockés (il est quelconque)
    :param taille_wav: durée en secondes des échantillons audio
    :return:nb_train: nombre d'images dans le dossier train
            nb_valid: nombre d'images dans le dossier validation
            chemin_data_train,chemin_data_validation,chemin_data_test: les chemins des 3 répertoires pour l'entrainement,la validation et le test
    """

    print("------------------------Conversion de tous les fichiers---------------------------")

    chemin_conv = src_audios  # là où se trouvent les fichiers audio d'entrée (où ils sont déjà triés dans 2 dossiers car et notcar)
    #spec = os.path.join(chemin_conv, 'spec')
    spec = chemin_conv + "//spec"
    if not os.path.isdir(spec):
        os.mkdir(spec)
   # audio_src = os.path.join(chemin_conv, 'audio')
    audio_src = chemin_conv + "//audio"
    if not os.path.isdir(audio_src):
        os.mkdir(audio_src)
    convertir_repertoire(audio_src, spec, taille_wav)

    print("-----------------------Fin de la préparation--------------------------------")

    # nb_train = len(os.listdir(chemin_data_train)) # 10
    # nb_valid = len(os.listdir(chemin_data_validation))  # 5  # ne fonctionne plus 

    nb_train, nb_valid = 21, 13
    # TODO a moduler
    return nb_train, nb_valid

# ----------------------------Conversion en spectrogramme---------------------------

def wav_to_spect(wav_name,output_name, output_dir, expected_time):
    """
    Conversion d'un fichier .wav, à partir de son nom, en spectrogramme
    et enregistrement dans le dossier indiqué par le path sous le nom
    output_name
    """

    ### On récupère les données, qui sont toutes en .vaw et de 10 secondes
    sample_rate, samples = wavfile.read(wav_name)  # fréquence d'échantillonage //  Sample 1D si audio et 2D si stéréo
    time = samples.size / sample_rate

    if time == expected_time:  # si le fichier fait bien 10 secondes
        nperseg = 512
        nfft = nperseg  # i.e. pas de zero-padding

        ### Obtention du spectrogramme, frequencies est un array
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=nperseg, nfft=nfft)
        ##spectrogram: fréquence en absisse, temps en ordonnées

        ### Coupure du spectrogramme (limitation des fréquences)
        fmin = 100
        fmax = 15000
        freqs_to_keep = (frequencies == frequencies)
        freqs_to_keep *= fmin <= frequencies
        freqs_to_keep *= frequencies <= fmax
        spectrogram = spectrogram[freqs_to_keep, :]
        frequencies = frequencies[freqs_to_keep]
        human_spectrogram = 10 * np.log(spectrogram)
        # la plupart des fréquences sont basses, donc pour que les différentes classes soient différenciées plus clairement,
        # étale les faible fréquences, avec le log

        ### Création de la figure
        t_max = times[-1]
        t_ref_normalize = int(t_max / 10) + 1  # Taille de l'image proportionelle au temps
        fig = plt.figure(frameon=False, figsize=(5 * t_ref_normalize, 5))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.pcolormesh(times, frequencies, human_spectrogram , cmap='gray_r')
       #output = os.path.join(output_dir,output_name)
        output = output_dir + "//" +  output_name
        plt.savefig(output, dpi=fig.dpi)
        plt.close(fig)
    else:
        print("Fichier audio de mauvaise longueur! Attendu: ", expected_time, "| Reçu: ", time)


def convertir_repertoire(src, dest, expected_time):
    fichiers_src = [f for f in os.listdir(src)]
    fichiers_dest = [d for d in os.listdir(dest)]
    for f in fichiers_src:
        chemin = os.path.join(src, f)
        if f[:-4] + '.png' not in fichiers_dest:
            wav_to_spect(chemin, f[:-4] + '.png', dest, expected_time)  # le -4 pour enlever le .png
            print(chemin + ' converti en image')
        else:
            print(chemin + ' déjà converti')

# --------------------------------------------------------------------
def CNN(img_height,img_length, nb_train,nb_valid, n_epochs):
    """
    Mise en place et entraînement d'un algorithme de réseau de neurones convolutifs.
    :param train_dir: répertoire où se trouvent les spectrogrammes d'entraînement
    :param validation_dir: répertoire où se trouvent les spectrogrammes de validation
    :param img_height: hauteur des spectrogrammes
    :param img_length: largeur des spectrogrammes
    :param num_of_images: nombre d'images à traiter
    :param n_epochs: nombre d'epochs(itérations) d'entraînement du cnn
    """
    print (nb_train, nb_valid)
    # taille des images:
    height = img_height
    length = img_length
    # taille des lots d'entrainement et de test
    train_size = 1
    test_size = 1

    print("-----------------------Création et Apprentissage du modèle--------------------------------")


    # création du modèle:

    NB_CLASSES = 4
    model = models.Sequential()
    # Conv2D(nb_filtres, taille de filtre, activation=fct activation, imput_shape=forme de l'image d'entrée)
    #model.add(layers.Dense(16, activation='relu'))  # 512 neurones reliés de manière dense
    model.add(layers.Conv2D(32,(3,3), activation='relu',input_shape=(height,length,1)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))   # à moduler au fur et à mesure du traitement des données
    model.add(layers.MaxPooling2D(2,2))
    #model.add(layers.Conv2D(128,(3,3),activation='relu'))
    # model.add(layers.MaxPooling2D(2,2))
   ## model.add(layers.Conv2D(128,(3,3),activation='relu'))
  ##  model.add(layers.MaxPooling2D(2,2))
  ##  model.add(layers.Dense(16, activation='relu')) # 512 neurones reliés de manière dense
    model.add(layers.Flatten())
    model.add(layers.Dense(NB_CLASSES, activation ='sigmoid')) # 1 neurone en sortie

    # configuration du modèle pour l'entrainement
    model.compile(loss='binary_crossentropy', optimizer='adam'  ,metrics=['acc']) # RMSprop(lr=1e-4)

    model.summary()

    # Lecture du fichier des classes, donc toutes les classes avec leur index
    CLASSE = pd.read_csv(chemin_bis, sep=',',
                         names=["classe", "index"], encoding='latin-1')

    df = pd.read_csv("C:/Users/Utilisateur/Documents/ENSTA/2A/UE 3.4/Projet système/Machine_learning/Donnees_label/mes_datas.txt", encoding='latin-1')
    df["labels"] = df["labels"].apply(lambda x: x.split(", "))
    print (df['path'])

    LIST_CLASS = []

    for index, row in tqdm(CLASSE.iterrows(), total=CLASSE.shape[0]):
        LIST_CLASS.append(row['classe'])

    LIST_CLASS.remove('classe')
    print("les classes sont ", LIST_CLASS)

    train_datagen = ImageDataGenerator(rescale=1./255 )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe = df[:nb_train],
        x_col="path",
        y_col="labels",
        color_mode='grayscale',
        target_size=(height,length),
        batch_size=train_size,
        class_mode= 'categorical',
        classes = None)

    valid_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = valid_datagen.flow_from_dataframe(
        dataframe = df [nb_train:nb_train + nb_valid],
        x_col="path",
        y_col="labels",
        color_mode='grayscale',
        target_size=(height,length),
        batch_size=test_size,
        class_mode= 'categorical',
        classes=None
    )

    # apprentissage du modèle

    history = model.fit(train_generator, epochs=n_epochs, validation_data=test_generator, steps_per_epoch=nb_train, validation_steps=nb_valid)

    # Permet de sauvegarder les indices de labels de classe  # Utilisé lors de prediction de nouveaux fichiers
    labels = train_generator.class_indices
    with open(chemin + "classIndice.txt", 'w') as file:
        file.write(json.dumps(labels))

    print("-----------------------Réalisation de prédictions par le modèle entraîné--------------------------------")

    test_datagen = ImageDataGenerator(rescale=1. / 255.)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df[nb_train + nb_valid:],
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

    pred = model.predict_generator(test_generator,
                                   steps=STEP_SIZE_TEST,
                                   verbose=1)

    # Interupteur pour lever la prediction si on a au moins 10% de confidence
    booleanPrediction = (pred > 0.2)
    print ("pred",pred)  # la matrice des probabilités
    print (booleanPrediction, "booleanPrediction") # la matrice des indices représentants les sons estimés présents (si True)

    f2 = open(chemin + "donnees_pour_carte", "w+")
    for i in range (5):
        f2.write("x : dans le titre de l image {}".format(i)+ ", " + "y: dans le titre de l image {}".format(i) + ", " + "t : dans le titre de l'image {}".format(i) + ", " + str (booleanPrediction[i]) + ", " + "l intensité maximale, a recueillir\n" )

    listPrediction = []

    # On récuperer le dictionnaire JSON des labels avec leur indice respectif
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
    print (df[15:]["labels"])

    # Tableau contenant l'ensemble des chemin des images
    pathImg = test_generator._filepaths

    # On creer un dataframe pour concat les images avec leurs predictions correctes
    results = pd.DataFrame({"Chemin img": pathImg, "Predictions": listPrediction})

   ###  affichage des performances
    acc = history.history['acc'] # précision à l'entrainement
    val_acc = history.history['val_acc'] # précision sur les données de test
    loss = history.history['loss'] # erreur
    val_loss = history.history['val_loss'] # erreur sur les données de

    #
    plt.plot(acc,'bo', label='Training acc')
    plt.plot(val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
    plt.plot(loss,'bo', label='Training loss')
    plt.plot(val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

chemin = "C://Users//Utilisateur//Documents//ENSTA//2A//UE 3.4//Projet système//Machine_learning//Donnees_label//"
chemin_bis = "C:/Users/Utilisateur/Documents/ENSTA/2A/UE 3.4/Projet système/Machine_learning/Donnees_label//listClass.txt"
ImageFile.LOAD_TRUNCATED_IMAGES = True  # permet de traiter les images corrompues
path_dataset_base = '\\Users\\Utilisateur\\Documents\\ENSTA\\2A\\UE 3.4\\Projet système\\dataset_m\\dictionnaire'
nom_rep_wav = 'wav'
nom_rep_dst = 'spec'
rep_dst = os.path.join(path_dataset_base,nom_rep_dst)
src_audios = os.path.join(path_dataset_base,nom_rep_wav)
taille_wav = 10  # durée en seconde des échantillons audios
#nb_train,nb_valid = preparer_reps(src_audios,rep_dst,taille_wav)
nb_train,nb_valid = preparer_reps(chemin + "wav",chemin + "spec",taille_wav)

CNN(500,500,nb_train,nb_valid,10)


