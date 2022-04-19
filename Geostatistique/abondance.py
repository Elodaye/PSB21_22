import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

class Abondance():
    def __init__(self, mode, a = 0, b = 0):
        '''
        :param mode: mode de la grille.
        :param a: longueur d'une maille.
        :param b: largeur d'une maille.
        '''
        self._mode = mode
        self._a = a
        self._b = b

    def abondance(self, tab, g, d = 0):
        '''abondance estime l'abondance a partir de z'''
        if self._mode == "regulier":
            abd = 0

            for i in range(len(tab)):
                abd += tab[i][2]

            return self._a * self._b * abd

        elif self._mode == "aleatoireStratifie":
            abd = 0

            for i in range(len(tab)):
                abd += tab[i][2]

            return self._a * self._b * abd

        elif self._mode == "aleatoireUniforme":
            abd = 0
            theta = len(tab) / d

            for i in range(len(tab)):
                abd += tab[i][2]

            return theta * abd

        else:
            print("Monde inconnue.")
            return -1

    def dist(self, p1, p2):
        '''Renvoie la distance entre les points p1 et p2 en dimension deux.'''
        return ((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2) ** 0.5

    def distH(self, tab, x0=[0, 0]):
        '''
        distH calcul les matrices des distances

        :param:
        tab: tableau des donnees (colonne: x, y, val).
        x0: tableau position du site a interpoller.

        :return
        matrixH: matrice des distances entre les points evalues.
        matrixH0: matrice des distance entre les points evalues et le point a interpoller.'''
        matrixH = np.zeros((len(tab), len(tab)))
        matrixH0 = np.zeros((len(tab), 1))

        for i in range(len(matrixH)):
            for j in range(len(matrixH)):
                matrixH[i, j] = self.dist(tab[i][:2], tab[j][:2])

            matrixH0[i] = self.dist(tab[i][:2], x0)

        return matrixH, matrixH0

    def semiCoariogrammeTransitiveLocal(self, tab, matrixH, h, dh):
        '''
        semiVariogrammeLocal calcul le variogramme experimental a une distance entre h - (dh / 2) et h + (dh / 2).

        :param:
        tab: tableau des donnees (colonne: x, y, val).
        matrixH: matrice des distances entre les points de mesures.
        h: distance.
        dh: dispersion sur la distance.

        :return
        nan si aucun points traite, sinon le variogramme experimental.
        '''
        n = 0
        var = 0

        for i in range(len(tab)):
            for j in range(len(tab) - i):
                if matrixH[i, j] > h - (dh / 2) and matrixH[i, j] <= h + (dh / 2):
                    var += (tab[i, 2] - tab[j, 2]) ** 2
                    n += 1

        if n == 0:
            return np.nan

        return var / (2 * n)

    def semiCovariogrammeTransitive(self, tab, matrixH, dh):
        '''
        semiVariogramme calcul le variogramme pour une dispersion en distance de dh.

        :param:
        tab: tableau des donnees (colonne: x, y, val).
        matrixH: matrice des distances entre les points de mesures.
        dh: dispersion sur la distance.

        :return
        Le variogramme experimental.
        '''
        tabH = np.arange(dh / 2, 10 * dh - (dh / 2), dh)
        variogramme = np.zeros(len(tabH), dtype='float')

        for i in range(len(tabH)):
            variogramme[i] = self.semiCoariogrammeTransitiveLocal(tab, matrixH, tabH[i], dh)

        return variogramme

if __name__ == "__main__":
    tab = np.array([[0, 0, 2], [1, 0, 1.8], [0, 1, 2]])  # Mesure.

    v1 = Abondance("regulier", 10, 10)

    matrixH, matrixH0 = v1.distH(tab)

    covariogramme = v1.semiCovariogrammeTransitive(tab, matrixH, 2)


    print(v1.abondance(tab, covariogramme))
