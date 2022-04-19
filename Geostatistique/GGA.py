import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.image as img
from pyproj import Proj, transform
from abc import ABC, abstractmethod

class GPS(ABC):
    def __init__(self, file):
        self.file = file

    def posGNSS(self, f):
        '''posGNSS renvoie une liste des donnee de hauteur.'''
        dataEasting = []
        dataNorthing = []

        for c in f:
            if c[:6] == "$GPGGA":
                cc = c.split(",")
                print(cc)
                if (cc[-4] != ',' or cc[-4] != '') and (cc[2] != ',' or cc[2] != '') and (cc[4] != ',' or cc[4] != ''):
                    dataEasting.append(cc[4])
                    dataNorthing.append(cc[2])

        f.close()

        return dataEasting, dataNorthing

class GGA(GPS):
    def __init__(self, fich):
        super().__init__(fich)

    def WGS84toLambert93(self, E, N):
        alpha = 0.622672164
        beta = -0.093309478
        ct1 = 546305.0875

        gamma = -0.046338701
        delta = -0.620534551
        cte2 = 4251139.032

        x = alpha * E + beta * N + ct1
        y = gamma * E + delta * N + cte2
        return x, y

    def deg_to_Lamb(self, x1, y1):
        outProj = Proj(init='epsg:2154')
        inProj = Proj(init='epsg:4326')
        x2, y2 = transform(inProj, outProj, x1, y1)
        return x2, y2

    def coordonne(self, fich):
        lignes = fich.readlines()
        GPA = []
        GSV = []
        for ligne in lignes:
            if '$GPGGA' in ligne:
                GPA.append(ligne)
            if '$GPGSV' in ligne:
                GSV.append(ligne)

        print(GSV)

        #######traitement de coordonnées

        liste_lat = []
        liste_long = []

        liste_Xlamb = []
        liste_Ylamb = []



        for mesure in GPA:
            mes = mesure.split(",")[2:6]
            if mes[1] == 'N':
                liste_long.append(float(mes[0][0:2]) + (float(mes[0][2:])) / 60)
            else:
                liste_long.append(-float(mes[0][0:2]) - (float(mes[0][2:])) / 60)
            if mes[3] == 'E':
                liste_lat.append(float(mes[2][0:3]) + (float(mes[2][3:])) / 60)
            else:
                liste_lat.append(-float(mes[2][0:3]) - (float(mes[2][3:])) / 60)

        for i in range(len(liste_lat)):
            X, Y = self.deg_to_Lamb(liste_lat[i], liste_long[i])
            liste_Xlamb.append(X)
            liste_Ylamb.append(Y)


        X_image = []
        Y_image = []

        for i in range(len(liste_Xlamb)):
            X, Y = self.WGS84toLambert93(liste_Xlamb[i], liste_Ylamb[i])
            X_image.append(X)
            Y_image.append(Y)

        return X_image, Y_image

class stat(GGA):
    def __init__(self, fich):
        super().__init__(fich)


    def statTest(self, fich, seuil):
        def incertitudeEllipse(m, seuil):
            '''incertitudeEllipse trace l'ellipse d'incertitude des donnees.
            Un test khi2 est utlise.

            Parametres:
            m: liste de liste (donne)
            seuil: float (erreur du test Khi2)'''
            tabKhi2 = [[0.1, 4.61], [0.05, 5.99], [0.01, 9.21]]
            k = 0

            for i in range(len(tabKhi2)):
                if tabKhi2[i][0] == seuil:
                    k = tabKhi2[i][1]

            if k == 0:
                k = tabKhi2[1][1]

            cov_xy = np.cov(m)
            x_mean = np.mean(m[0])
            y_mean = np.mean(m[1])

            r1 = ((cov_xy[0, 0] + cov_xy[1, 1]) / 2) + (
                        ((cov_xy[0, 0] - cov_xy[1, 1]) / 2) ** 2 + cov_xy[0, 1] ** 2) ** 0.5
            r2 = ((cov_xy[0, 0] + cov_xy[1, 1]) / 2) - (
                        ((cov_xy[0, 0] - cov_xy[1, 1]) / 2) ** 2 + cov_xy[0, 1] ** 2) ** 0.5

            if cov_xy[0, 1] == 0 and cov_xy[0, 0] > cov_xy[1, 1]:
                theta = 0
            elif cov_xy[0, 1] == 0 and cov_xy[0, 0] < cov_xy[1, 1]:
                theta = np.pi / 2
            elif cov_xy[0, 0] == 0:
                theta = np.pi / 2
            else:
                theta = np.arctan(cov_xy[1, 1] / cov_xy[0, 0])

            x = []
            y = []
            x1 = []
            y1 = []
            x2 = []
            y2 = []

            t = np.linspace(0, 2 * np.pi, 100)

            for i in range(len(t)):
                x.append((k * r1) ** 0.5 * np.cos(theta) * np.cos(t[i]) - (k * r2) ** 0.5 * np.sin(theta) * np.sin(
                    t[i]) + x_mean)
                y.append((k * r1) ** 0.5 * np.sin(theta) * np.cos(t[i]) + (k * r2) ** 0.5 * np.cos(theta) * np.sin(
                    t[i]) + y_mean)
                x1.append(
                    (4.61 * r1) ** 0.5 * np.cos(theta) * np.cos(t[i]) - (4.61 * r2) ** 0.5 * np.sin(theta) * np.sin(
                        t[i]) + x_mean)
                y1.append(
                    (4.61 * r1) ** 0.5 * np.sin(theta) * np.cos(t[i]) + (4.61 * r2) ** 0.5 * np.cos(theta) * np.sin(
                        t[i]) + y_mean)
                x2.append(
                    (9.21 * r1) ** 0.5 * np.cos(theta) * np.cos(t[i]) - (9.21 * r2) ** 0.5 * np.sin(theta) * np.sin(
                        t[i]) + x_mean)
                y2.append(
                    (9.21 * r1) ** 0.5 * np.sin(theta) * np.cos(t[i]) + (9.21 * r2) ** 0.5 * np.cos(theta) * np.sin(
                        t[i]) + y_mean)

            plt.scatter(m[0], m[1])
            plt.plot(x1, y1, label="90%", linestyle='--', color='#1f77b4')
            plt.plot(x, y, label=str(1 - seuil)[2:] + '%', linestyle='--', color='#1f22b4')
            plt.plot(x2, y2, label="99%", linestyle='--', color='#1f006c')
            plt.suptitle("Ellipses d'incertitudes sur des mesures en differentiel", x=0.5, y=0.98)
            plt.xlabel("Longitute")
            plt.ylabel("Latitude")
            plt.legend()
            plt.show()

        tabEasting, tabNorthing = self.posGNSS(fich)

        m = np.zeros((len(tabNorthing), 2))

        for i in range(len(tabNorthing)):
            m[i][0] = tabEasting[i]
            m[i][1] = tabNorthing[i]

        m = m.T

        incertitudeEllipse(m, seuil)

if __name__ == "__main__":
    fich = open("test.txt", "r")

    v1 = GGA(fich)

    v1.coordonne(fich)

    try:
        fich = open("ue24_9000_20190614_100000.txt", "r")
    except IOError as e:
        print("Erreur ouverture fichier.\n", e)


    v2 = stat(fich)

    v2.statTest(fich, 0.05)
