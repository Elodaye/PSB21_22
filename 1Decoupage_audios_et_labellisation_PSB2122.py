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
        self.setFixedSize(300, 210)

        self.bouton_valider = QtWidgets.QPushButton("Save", self)  # on crée un bouton qui affichera 'Lancer la partie'
        self.bouton_valider.setFixedSize(150, 40)  # taille du bouton
        self.bouton_valider.setFont(QFont('Calibri', 12))   # type et taille de police
        self.bouton_valider.move(120, 120)  # position du bouton
        self.bouton_play =  QtWidgets.QPushButton("Play", self)
        self.bouton_play.setFixedSize(150, 40)  # taille du bouton
        self.bouton_play.setFont(QFont('Calibri', 12))  # type et taille de police
        self.bouton_play.move(120, 120)  # position du bouton
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

        self.titre4= QLabel("Séparer les labels par des ', ' ")
        self.titre4.setFont(QFont('Calibri', 12))
        self.titre4.setFixedSize(280,30)
        self.titre4.move(20, 200)
        self.titre4.show()

        self.layout.addRow(self.titre1)
        self.layout.addRow(self.titre3, self.nb_tour)
        self.layout.addWidget(self.bouton_play)
        self.layout.addWidget(self.bouton_valider)
        self.layout.addRow(self.titre4)
        zoneCentrale.setLayout (self.layout)
        self.setCentralWidget(zoneCentrale)
        self.bouton_valider.clicked.connect(self.recueille_label)
        self.bouton_play.clicked.connect(self.ecoute)

        sr, self.sig = wavfile.read("C://Users//Utilisateur//Downloads//premier_test.wav")
        # sr = fréquence d'échantillonage
        duree = 300          # durée de l'audio en entrée
        self.temps_spec = 10 # durée de l'enregistrement, choisie à 10 secondes.
        nb_points = sr * duree
        self.pas2 = nb_points * self.temps_spec / duree  # nb de points correspondant à un échantillon de 10 secondes

        self.f = open("C://Users//Utilisateur//Downloads//mes_datas.txt", "w")
        self.f.write("path,labels\n")
        self.sig_splits = []
        self.i = 0

        self.split = self.sig[self.i * (int(self.pas2)): (self.i + 1) * (int(self.pas2))]
        self.sig_splits.append(self.split)
        self.freq = sr   # ça me sort 32 000

        self.show()

    def recueille_label (self) :
        # Appelée à chaque fois que l'on appuie sur Save, après avoir rentré les labels de l'échantillon à labelliser
        # Sélectionne la partie de l'enregistrement (d'une durée de 10 sec.) que l'on souhaite labelliser et créer un nouvel enregistrement correspondant à la partie sélectionnée.
        # Stocke les labels associés à l'enregistrement dans un fichier mes_data.
        # Propose le son suivant à labelliser

        self.label.append (str(self.nb_tour.text()))
        nb = self.nb_tour.text()
        print ("enregistrement n°{}, le label est {}".format (self.i +1, nb))
        try :
            nb = str (nb)
            self.close()
            self.i += 1

            if self.i < int(300 / self.temps_spec - 20):
                self.split = self.sig[self.i * (int(self.pas2)): (self.i + 1) * (int(self.pas2))]
                self.sig_splits.append(self.split)
                self.titre1.setText(("Echantillon n° {}".format(self.i +1)))
                self.show()

            else :
                for i in range(int(300 / self.temps_spec - 20)):
                    write("C://Users//Utilisateur//Downloads//recording{}.wav".format(i + 1), self.freq, self.sig_splits[i])
                    if "," in self.label[i]: # si on a plusieurs labels
                        self.label[i] = '"' + self.label[i] + '"'
                    self.f.write("C://Users//Utilisateur//Downloads//recording{}.png,{}\n".format(i + 1, self.label[i]))
                self.f.close()

        except ValueError :   # on ne peut faire un nombre de coup qui ne soit pas un nombre entier
            self.titre3.setText('Labels valables requis !')   # il va falloir changer la valeur de nb
            self.titre3.show()

    def ecoute(self):
        # Appelée lorsque l'on clique le "play" lors de la labellisation.
        # Elle joue la partie de l'enregistrement que l'on s'apprête à labelliser

       pygame.mixer.pre_init(48000, size=-16, channels=1)
       pygame.mixer.init()
       mySound = np.array([[i,0] for i in self.split])
       sound = pygame.sndarray.make_sound(mySound)  # objet de type Sound
       sound.play()


if __name__ == "__main__" :

    app = QApplication([])
    win = Window_Menu()
    app.exec()
