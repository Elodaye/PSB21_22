import os

from PyQt5 import QtGui, QtCore, QtWidgets, uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit , QFormLayout, QAction
from PyQt5.QtWidgets import QApplication, QMainWindow
import time
import pygame
from scipy.io import wavfile
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

class Window_Menu (QtWidgets.QMainWindow) :

    def __init__(self, *args, **kwargs):
        super(Window_Menu, self).__init__(*args, **kwargs)

        self.label = []
        pixmap = QtGui.QPixmap("arriere_plan_lancement.png")
        pal = QtGui.QPalette()
        pal.setBrush(QtGui.QPalette.Background, QtGui.QBrush(pixmap))
        zoneCentrale = QWidget()
        zoneCentrale.lower()
        zoneCentrale.stackUnder(self)
        zoneCentrale.setAutoFillBackground(True)
        zoneCentrale.setPalette(pal)

        self.setWindowTitle("Entrez les labels de l'échantillon")
        self.setFixedSize(300, 570)

        self.bouton_car = QtWidgets.QPushButton("car", self)  # on crée un bouton qui affichera 'Lancer la partie'
        self.bouton_car.setFixedSize(150, 40)  # taille du bouton
        self.bouton_car.setFont(QFont('Calibri', 12))   # type et taille de police
        self.bouton_car.move(120, 120)  # position du bouton

        self.bouton_voix =  QtWidgets.QPushButton("voix", self)
        self.bouton_voix.setFixedSize(150, 40)  # taille du bouton
        self.bouton_voix.setFont(QFont('Calibri', 12))  # type et taille de police
        self.bouton_voix.move(120, 120)  # position du bouton

        self.bouton_velo = QtWidgets.QPushButton("velo", self)
        self.bouton_velo.setFixedSize(150, 40)  # taille du bouton
        self.bouton_velo.setFont(QFont('Calibri', 12))  # type et taille de police
        self.bouton_velo.move(120, 120)  # position du bouton

        self.bouton_mouette = QtWidgets.QPushButton("mouette", self)
        self.bouton_mouette.setFixedSize(150, 40)  # taille du bouton
        self.bouton_mouette.setFont(QFont('Calibri', 12))  # type et taille de police
        self.bouton_mouette.move(120, 120)  # position du bouton

        self.bouton_moto = QtWidgets.QPushButton("moto", self)
        self.bouton_moto.setFixedSize(150, 40)  # taille du bouton
        self.bouton_moto.setFont(QFont('Calibri', 12))  # type et taille de police
        self.bouton_moto.move(120, 120)  # position du bouton

        self.bouton_pie = QtWidgets.QPushButton("pie", self)
        self.bouton_pie.setFixedSize(150, 40)  # taille du bouton
        self.bouton_pie.setFont(QFont('Calibri', 12))  # type et taille de police
        self.bouton_pie.move(120, 120)  # position du bouton

        self.bouton_chien = QtWidgets.QPushButton("chien", self)
        self.bouton_chien.setFixedSize(150, 40)  # taille du bouton
        self.bouton_chien.setFont(QFont('Calibri', 12))  # type et taille de police
        self.bouton_chien.move(120, 120)  # position du bouton


        self.bouton_valider = QtWidgets.QPushButton("Save", self)
        self.bouton_valider.setFixedSize(150, 40)  # taille du bouton
        self.bouton_valider.setFont(QFont('Calibri', 12))  # type et taille de police
        self.bouton_valider.setStyleSheet("background-color : green")
        self.bouton_valider.move(120, 120)  # position du bouton

        self.bouton_play = QtWidgets.QPushButton("Play", self)
        self.bouton_play.setFixedSize(150, 40)  # taille du bouton
        self.bouton_play.setFont(QFont('Calibri', 12))  # type et taille de police
        self.bouton_play.setStyleSheet("background-color : yellow")
        self.bouton_play.move(120, 120)  # position du bouton

        self.bouton_efface = QtWidgets.QPushButton("Discard", self)
        self.bouton_efface.setFixedSize(150, 40)  # taille du bouton
        self.bouton_efface.setFont(QFont('Calibri', 12))  # type et taille de police
        self.bouton_efface.setStyleSheet("background-color : red")
        self.bouton_efface.move(120, 120)  # position du bouton

        self.nb_tour = QLineEdit(self)
        self.nb_tour.setFixedSize(150,30)
        self.nb_tour.move(20, 100)



        self.layout = QFormLayout()

        self.titre1 = QLabel ("Echantillon n° {}".format(1))
        self.titre1.setFont(QFont('Calibri', 12))
        self.titre1.setFixedSize(150,20)
        self.titre1.move(50,0)
        self.titre1.show()

        self.titre3= QLabel("Labels")
        self.titre3.setFont(QFont('Calibri', 10))
        self.titre3.setFixedSize(80,30)
        self.titre3.move(20, 100)
        self.titre3.show()

        # self.titre4= QLabel("Séparer les labels par des ', ' ")
        # self.titre4.setFont(QFont('Calibri', 12))
        # self.titre4.setFixedSize(280,30)
        # self.titre4.move(20, 200)
        # self.titre4.show()

        self.layout.addRow(self.titre1)
        self.layout.addRow(self.titre3, self.nb_tour)


        self.layout.addWidget(self.bouton_car)
        self.layout.addWidget(self.bouton_voix)
        self.layout.addWidget(self.bouton_velo)
        self.layout.addWidget(self.bouton_mouette)
        self.layout.addWidget(self.bouton_moto)
        self.layout.addWidget(self.bouton_pie)
        self.layout.addWidget(self.bouton_chien)

        self.layout.addWidget(self.bouton_play)
        self.layout.addWidget(self.bouton_efface)
        self.layout.addWidget(self.bouton_valider)

        # self.layout.addRow(self.titre4)

        zoneCentrale.setLayout (self.layout)
        self.setCentralWidget(zoneCentrale)

        self.bouton_valider.clicked.connect(self.recueille_label)
        self.bouton_efface.clicked.connect(self.efface)
        self.bouton_play.clicked.connect(self.ecoute)

        self.bouton_car.clicked.connect(self.ajoute_label_car)
        self.bouton_voix.clicked.connect(self.ajoute_label_voix)
        self.bouton_velo.clicked.connect(self.ajoute_label_velo)
        self.bouton_mouette.clicked.connect(self.ajoute_label_mouette)
        self.bouton_moto.clicked.connect(self.ajoute_label_moto)
        self.bouton_pie.clicked.connect(self.ajoute_label_pie)
        self.bouton_chien.clicked.connect(self.ajoute_label_chien)


        sr, self.sig = wavfile.read(chemin_court + "musique_1.wav")
        #TODO mettre le bon fichier à lire !!
        #sr = fréquence d'échantillonage
        self.duree = 245     # durée de l'audio en entrée
        # TODO cette valeur doit être maj à chaque fois
        self.temps_spec = 10 # durée de l'enregistrement, choisie à 10 secondes.
        nb_points = sr * self.duree
        self.pas2 = nb_points * self.temps_spec / self.duree  # nb de points correspondant à un échantillon de 10 secondes

        self.f = open(chemin_court + "mes_datas.txt", "a")
        if os.path.getsize(chemin_court + "mes_datas.txt") == 0 :
            self.f.write("path,labels\n")
        self.sig_splits = []
        self.sig_splits_tr_n = []
        self.sig_splits_tr_b = []
        self.depart = 285
        # TODO cette valeur doit être maj à chaque fois
        self.i = 285  # anciens comptés
        # TODO cette valeur doit être maj à chaque fois
        self.ii = 0

        self.split = self.sig[self.ii * (int(self.pas2)): (self.ii + 1) * (int(self.pas2)),0]
        self.sig_splits.append(self.split)
        print(self.sig_splits)
        self.freq = sr   # ça me sort 32 000

        self.show()

    def ajoute_label_car (self):
        if str(self.nb_tour.text()) == '':
            self.nb_tour.setText(self.nb_tour.text() + "car")
        else :
            self.nb_tour.setText(self.nb_tour.text() + ", " + "car")

    def ajoute_label_voix (self):
        if str(self.nb_tour.text()) == '':
            self.nb_tour.setText(self.nb_tour.text() + "voix")
        else :
            self.nb_tour.setText(self.nb_tour.text() + ", " + "voix")

    def ajoute_label_velo (self):
        if str(self.nb_tour.text()) == '':
            self.nb_tour.setText(self.nb_tour.text() + "velo")
        else :
            self.nb_tour.setText(self.nb_tour.text() + ", " + "velo")

    def ajoute_label_mouette (self):
        if str(self.nb_tour.text()) == '':
            self.nb_tour.setText(self.nb_tour.text() + "mouette")
        else :
            self.nb_tour.setText(self.nb_tour.text() + ", " + "mouette")

    def ajoute_label_moto (self):
        if str(self.nb_tour.text()) == '':
            self.nb_tour.setText(self.nb_tour.text() + "moto")
        else :
            self.nb_tour.setText(self.nb_tour.text() + ", " + "moto")

    def ajoute_label_pie (self):
        if str(self.nb_tour.text()) == '':
            self.nb_tour.setText(self.nb_tour.text() + "pie")
        else :
            self.nb_tour.setText(self.nb_tour.text() + ", " + "pie")

    def ajoute_label_chien(self):
        if str(self.nb_tour.text()) == '':
            self.nb_tour.setText(self.nb_tour.text() + "chien")
        else:
            self.nb_tour.setText(self.nb_tour.text() + ", " + "chien")

    def efface(self):
        self.nb_tour.setText('')

    def recueille_label (self) :
        # Appelée à chaque fois que l'on appuie sur Save, après avoir rentré les labels de l'échantillon à labelliser
        # Sélectionne la partie de l'enregistrement (d'une durée de 10 sec.) que l'on souhaite labelliser et créer un nouvel enregistrement correspondant à la partie sélectionnée.
        # Stocke les labels associés à l'enregistrement dans un fichier mes_data.
        # Propose le son suivant à labelliser

        self.label.append (str(self.nb_tour.text()))
        nb = self.nb_tour.text()
        print ("enregistrement n°{}, le label est {}".format (self.i +1, nb))
        if self.ii == 0 :
            return None
        try :
            nb = str (nb)
            self.close()
            self.i += 1


            if self.ii < int(self.duree / self.temps_spec-1):

                self.split_tr_b = self.sig[self.ii * (int(self.pas2)) - int(self.pas2 / 8): (self.ii + 1) * (int(self.pas2)) - int( self.pas2 / 8), 0]
                self.sig_splits_tr_b.append(self.split_tr_b)

                self.split_tr_n = self.sig[self.ii * (int(self.pas2)) + int(self.pas2 / 8): (self.ii + 1) * (int(self.pas2)) + int(self.pas2 / 8), 0]
                self.sig_splits_tr_n.append(self.split_tr_n)

                self.split = self.sig[self.ii * (int(self.pas2)): (self.ii + 1) * (int(self.pas2)),0]  # ,0 si on a un audio
                self.sig_splits.append(self.split)

                self.titre1.setText(("Echantillon n° {}".format(self.i +1)))
                self.show()
                self.nb_tour.setText('')
                self.ii += 1

            else :
                for i in range(int(self.duree / self.temps_spec)):
                    write(chemin_court + "wav/audio/" + "recording{}_trb.wav".format(i + self.depart + 1), self.freq, self.sig_splits_tr_b[i])

                    write(chemin_court + "wav/audio/" + "recording{}_trn.wav".format(i + self.depart + 1), self.freq,self.sig_splits_tr_n[i])
                    write(chemin_court + "wav/audio/" + "recording{}.wav".format(i +self.depart + 1), self.freq, self.sig_splits[i])


                    if "," in self.label[i]: # si on a plusieurs labels
                        self.label[i] = '"' + self.label[i] + '"'

                    self.f.write(chemin + "recording{}_trb.png,{}\n".format(i + 1 + self.depart, self.label[i]))
                    self.f.write(chemin + "recording{}_trb_r1.png,{}\n".format(i + 1 + self.depart, self.label[i]))
                    self.f.write(chemin + "recording{}_trb_r2.png,{}\n".format(i + 1 + self.depart, self.label[i]))

                    self.f.write(chemin + "recording{}_trn.png,{}\n".format(i + 1 + self.depart, self.label[i]))
                    self.f.write(chemin + "recording{}_trn_r1.png,{}\n".format(i + 1 + self.depart, self.label[i]))
                    self.f.write(chemin + "recording{}_trn_r2.png,{}\n".format(i + 1 + self.depart, self.label[i]))

                    self.f.write(chemin + "recording{}.png,{}\n".format(i + 1 + self.depart, self.label[i]))
                    self.f.write(chemin + "recording{}_r1.png,{}\n".format(i + 1 + self.depart, self.label[i]))
                    self.f.write(chemin + "recording{}_r2.png,{}\n".format(i + 1 + self.depart, self.label[i]))


                self.f.close()

        except ValueError :   # on ne peut faire un nombre de coup qui ne soit pas un nombre entier
            self.titre3.setText('Labels valables requis !')   # il va falloir changer la valeur de nb
            self.titre3.show()

    def ecoute(self):
        # Appelée lorsque l'on clique le "play" lors de la labellisation.
        # Elle joue la partie de l'enregistrement que l'on s'apprête à labelliser

       pygame.mixer.pre_init(48000, size=-16, channels=1)
       pygame.mixer.init()
       print (self.split)
       mySound = np.array([[i,0] for i in self.split]) # , dtype = object
       sound = pygame.sndarray.make_sound(mySound)  # objet de type Sound
       sound.play()

chemin = "C:/Users/Utilisateur/Documents/ENSTA/2A/UE 3.4/Projet système/Machine_learning/Donnees_label/wav/spec/"
chemin_court = "C:/Users/Utilisateur/Documents/ENSTA/2A/UE 3.4/Projet système/Machine_learning/Donnees_label/"

if __name__ == "__main__" :

    app = QApplication([])
    win = Window_Menu()
    app.exec()
