import numpy as np
import matplotlib.pyplot as plt
import os

import random
import sklearn
import torch
#import torchaudio
#from torchaudio import transforms
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
from keras.applications.vgg16 import VGG16


def preparer_reps(src_audios, rep_dst, taille_wav):

    print("------------------------Conversion de tous les fichiers---------------------------")

    chemin_conv = src_audios  # la ou se trouvent les fichiers audio d'entree (ou ils sont deja tries dans 2 dossiers car et notcar)
    #spec = os.path.join(chemin_conv, 'spec')
    spec = chemin_conv + "//spec"
    if not os.path.isdir(spec):
        os.mkdir(spec)
   # audio_src = os.path.join(chemin_conv, 'audio')
    audio_src = chemin_conv + "//audio"
    if not os.path.isdir(audio_src):
        os.mkdir(audio_src)
    convertir_repertoire(audio_src, spec, taille_wav)

    print("-----------------------Fin de la preparation--------------------------------")

    # nb_train = len(os.listdir(chemin_data_train)) # 10
    # nb_valid = len(os.listdir(chemin_data_validation))  # 5  # ne fonctionne plus

    nb_train, nb_valid = 518, 518 # multiples de 2 si possible
    # TODO a moduler
    return nb_train, nb_valid

# ----------------------------Conversion en spectrogramme---------------------------

def wav_to_spect(wav_name,output_name, output_dir, expected_time):
    """
    Conversion d'un fichier .wav, a partir de son nom, en spectrogramme
    et enregistrement dans le dossier indique par le path sous le nom
    output_name
    """

    ### On recupere les donnees, qui sont toutes en .vaw et de 10 secondes
    sample_rate, samples = wavfile.read(wav_name)  # frequence d'echantillonage //  Sample 1D si audio et 2D si stereo
    time = samples.size / sample_rate

    if time == expected_time:  # si le fichier fait bien 10 secondes
        nperseg = 512 #4094*2
        nfft = nperseg  # i.e. pas de zero-padding

        ### Obtention du spectrogramme, frequencies est un array
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=nperseg, nfft=nfft)
        ##spectrogram: frequence en absisse, temps en ordonnees

        ### Coupure du spectrogramme (limitation des frequences)
        fmin = 100
        fmax = 15000
        freqs_to_keep = (frequencies == frequencies)
        freqs_to_keep *= fmin <= frequencies
        freqs_to_keep *= frequencies <= fmax
        spectrogram = spectrogram[freqs_to_keep, :]
        frequencies = frequencies[freqs_to_keep]
        #print("hu_spectrogram", spectrogram, spectrogram.shape)

        #print(spectrogram.max(), spectrogram.min())

        spectrogram_t = - np.log(spectrogram)
        spectrogram_ok = (spectrogram_t/ 24* 255) + 100
        #print(spectrogram_ok.max(), spectrogram_ok.min())

        for i, ligne in enumerate(spectrogram_ok):
            for j, colonne in enumerate(ligne):
                if colonne > 120:
                    spectrogram_ok[i][j] = 120
  

                    #human_spectrogram = 100 * np.log10(spectrogram)
        #human_spectrogram = np.exp(spectrogram) / 10
        #print("huuuuuuuman spectrogrammmm", human_spectrogram, human_spectrogram.shape)
        #human_spectrogram = spectrogram
        # la plupart des frequences sont basses, donc pour que les differentes classes soient differenciees plus clairement,
        # etale les faible frequences, avec le log

        ### Creation de la figure
        t_max = times[-1]
        t_ref_normalize = int(t_max / 10) + 1  # Taille de l'image proportionelle au temps --> = 1

        fig = plt.figure(frameon=False, figsize=(10 * t_ref_normalize, 10))
        # largeur en pouce 5, hauteur en pouces 5, et dpi de 100 donne une image de 500*500 pixels
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        plt.pcolormesh(times, frequencies, spectrogram_ok , cmap='binary') ##gray_r
        #plt.colorbar()
       #output = os.path.join(output_dir,output_name)
        output = output_dir + "//" +  output_name
        plt.savefig(output, dpi=fig.dpi) # fig.dpi = 100 --> 100 points par pouces
        #print(fig.dpi)
        #plt.show()
        plt.close(fig)

        spec_for_aug = copy.deepcopy(spectrogram_ok)
        spec_for_aug2 = copy.deepcopy(spectrogram_ok)

        spectrogram_okc = spectro_augment(spec_for_aug[:,:])
        #print(np.array_equal(spectrogram_ok, spectrogram_okc))

        fig0 = plt.figure(frameon=False, figsize=(10 * t_ref_normalize, 10))
        ax0 = plt.Axes(fig0, [0., 0., 1., 1.])
        ax0.set_axis_off()
        fig0.add_axes(ax0)
        plt.pcolormesh(times, frequencies, spectrogram_okc, cmap='binary')  ##gray_r
        output0 = output_dir + "//" + output_name[0:-4] +"_r1" + output_name[-4::]
        plt.savefig(output0, dpi=fig0.dpi)  # fig.dpi = 100 --> 100 points par pouces
        plt.close(fig0)

        spectrogram_okc2 = spectro_augment(spec_for_aug2[:,:])
        #print(np.array_equal(spectrogram_ok, spectrogram_okc2))
        fig02 = plt.figure(frameon=False, figsize=(10 * t_ref_normalize, 10))
        ax02 = plt.Axes(fig02, [0., 0., 1., 1.])
        ax02.set_axis_off()
        fig02.add_axes(ax02)
        plt.pcolormesh(times, frequencies, spectrogram_okc2, cmap='binary')  ##gray_r  spec_for_aug
        output02 = output_dir + "//" + output_name[0:-4] +"_r2" + output_name[-4::]
        plt.savefig(output02, dpi=fig02.dpi)  # fig.dpi = 100 --> 100 points par pouces
        plt.close(fig02)

    else:
        print("Fichier audio de mauvaise longueur Attendu: ", expected_time, "| Recu: ", time)


def convertir_repertoire(src, dest, expected_time):
    fichiers_src = [f for f in os.listdir(src)]
    fichiers_dest = [d for d in os.listdir(dest)]
    for f in fichiers_src:
        chemin = os.path.join(src, f)
        if f[:-4] + '.png' not in fichiers_dest:
            wav_to_spect(chemin, f[:-4] + '.png', dest, expected_time)  # le -4 pour enlever le .png
            print(chemin + ' converti en image')
        else:
            print(chemin + ' deja converti')


def spectro_augment(spect, n_freq_masks=12, n_time_masks=200):
    aug_spec = spect[:, :]
    n_mels, n_steps = aug_spec.shape
    mask_value = aug_spec.mean()


    max_mask_pct_f = random.random()
    freq_mask_param = max_mask_pct_f * n_mels
    freq_mask_param = int (min(freq_mask_param, n_mels - n_freq_masks-1))
    aug_spec[freq_mask_param: freq_mask_param+int(n_freq_masks),:] = mask_value

    max_mask_pct_t = random.random()
    time_mask_param = max_mask_pct_t * n_steps
    time_mask_param = int(min(time_mask_param, n_steps - n_time_masks-1))
    aug_spec [:,time_mask_param: time_mask_param + int(n_time_masks)] = mask_value

    return aug_spec[:,:]


def time_shift(aud, shift_limit):
    sig, sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)


# --------------------------------------------------------------------
def CNN(img_height,img_length, nb_train,nb_valid, n_epochs):
    """
    Mise en place et entrainement d'un algorithme de reseau de neurones convolutifs.
    :param train_dir: repertoire ou se trouvent les spectrogrammes d entrainement
    :param validation_dir: repertoire ou se trouvent les spectrogrammes de validation
    :param img_height: hauteur des spectrogrammes
    :param img_length: largeur des spectrogramme 
    :param num_of_images: nombre d images a traiter
    :param n_epochs: nombre d epochs(iterations) d'entrainement du cnn
    """
    print (nb_train, nb_valid)
    # taille des images:
    height = img_height
    length = img_length
    # taille des lots d'entrainement et de test
    train_size =  128 # 1
    test_size = 128 # 1

    print("-----------------------Creation et Apprentissage du modele--------------------------------")


    # creation du modele:

    NB_CLASSES = 7

    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(height, length, 3))

    conv_base.summary()



    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(128,activation = "relu"))
    model.add(layers.Dense(NB_CLASSES, activation ='sigmoid')) # 1 neurone en sortie

    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
    	if layer.name == 'block5_conv1':
             set_trainable = True
    	if set_trainable:
    	     layer.trainable = True
    	else:
    	     layer.trainable = False



    # configuration du mod le pour l entrainement
    model.compile(loss='categorical_crossentropy', optimizer='adam'  ,metrics=['acc']) # RMSprop(lr=1e-4)

    model.summary()

    # Lecture du fichier des classes, donc toutes les classes avec leur index
    CLASSE = pd.read_csv(chemin_bis, sep=',',
                         names=["classe", "index"], encoding='latin-1')

    df_0 = pd.read_csv("Donnees_label/mes_datas.txt", encoding='latin-1')
    df_0["labels"] = df_0["labels"].apply(lambda x: x.split(", "))
    df_0["path"] = df_0["path"].apply(lambda x: x[79::])  ## on enleve certains elements du path
    df = sklearn.utils.shuffle(df_0)
    print (df.shape)
    #print (df, df_0)

    #print (df['path'])

    LIST_CLASS = []

    memo = {}
    for index, row in tqdm(CLASSE.iterrows(), total=CLASSE.shape[0]):
        if index > 0 :
            memo [index-1] = row['classe']
        LIST_CLASS.append(row['classe'])
    print (memo)


    LIST_CLASS.remove('classe')
    print("les classes sont ", LIST_CLASS)

    train_datagen = ImageDataGenerator(rescale=1. / 255)  # , width_shift_range=0.1, zoom_range=0.1, brightness_range = (0.8,1.2))

    features_train = np.zeros(shape=(nb_train, 1000, 1000, 3))
    labels_train = np.zeros(shape=(nb_train,NB_CLASSES))

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df[:nb_train],
        x_col="path",
        y_col="labels",
        color_mode='grayscale',
        target_size=(height, length),
        batch_size= train_size,
        class_mode='categorical',
        classes=None)

    i = 0
    for inputs_batch_x, labels_batch in train_generator:
        inputs_batch_0 = np.concatenate ((inputs_batch_x, inputs_batch_x), axis = 3)
        inputs_batch = np.concatenate((inputs_batch_0, inputs_batch_x), axis=3)
        features_batch = inputs_batch
        features_train[i * train_size: (i + 1) * train_size] = features_batch

        print (labels_batch)
        labels_train[i * train_size: (i + 1) * train_size] = labels_batch
        i += 1
        print (i)
        if i * train_size >= nb_train:
            break
  

    valid_datagen = ImageDataGenerator(rescale=1. / 255)  # , width_shift_range=0.1, zoom_range=0.1, brightness_range = (0.8,1.2))

    features_valid = np.zeros(shape=(nb_train, 1000, 1000, 3))
    labels_valid = np.zeros(shape=(nb_train, NB_CLASSES))

    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=df [nb_train:nb_train + nb_valid],
        x_col="path",
        y_col="labels",
        color_mode='grayscale',
        target_size=(height, length),
        batch_size=test_size,
        class_mode='categorical',
	classes = None)

    i = 0
    for inputs_batch_x, labels_batch in valid_generator:
        inputs_batch_0 = np.concatenate ((inputs_batch_x, inputs_batch_x), axis = 3)
        inputs_batch = np.concatenate((inputs_batch_0, inputs_batch_x), axis=3)
        features_batch = inputs_batch
        features_valid[i * train_size: (i + 1) * train_size] = features_batch
        print (labels_batch,  labels_valid[i * train_size: (i + 1) * train_size])
        labels_valid[i * train_size: (i + 1) * train_size] = labels_batch
        i += 1
        print (i)
        if i * train_size >= nb_valid:
            break


    #train_features = np.reshape(features_train, (nb_train, 31 * 31 * 512)) ## laaaaaaaaaaaaa pb 
    #validation_features = np.reshape(features_valid, (nb_valid, 31 * 31 * 512))



  
    test_datagen = ImageDataGenerator(rescale=1. / 255)  # , width_shift_range=0.1, zoom_range=0.1, brightness_range = (0.8,1.2))


    # apprentissage du modele

    #history = model.fit(train_generator, epochs=n_epochs, validation_data=valid_generator, steps_per_epoch=nb_train, validation_steps=nb_valid)
    history = model.fit (features_train, labels_train,
                        epochs=n_epochs,
                        batch_size=train_size,
                        validation_data=(features_valid, labels_valid))

    model.save('Pretrained2_128_20.h5')


chemin = "Donnees_label//"
chemin_bis = "Donnees_label//listClass.txt"
ImageFile.LOAD_TRUNCATED_IMAGES = True  # permet de traiter les images corrompues

nom_rep_wav = 'wav'
nom_rep_dst = 'spec'
#rep_dst = os.path.join(path_dataset_base,nom_rep_dst)
#src_audios = os.path.join(path_dataset_base,nom_rep_wav)
taille_wav = 10  # duree en seconde des echantillons audios
#nb_train,nb_valid = preparer_reps(src_audios,rep_dst,taille_wav)
nb_train,nb_valid = preparer_reps(chemin + "wav",chemin + "spec",taille_wav)

CNN(1000,1000,1024,1024,20)
