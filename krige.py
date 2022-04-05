import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Krige(ABC):
    def __init__(self, mode):
        self._mode = mode

    @property
    def mode(self):
        return self.mode

    @mode.setter
    def mode(self, value):
        self._mode = value

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

class SemiVariogramme(Krige):
    def __init__(self, mode):
        super().__init__(mode)

    def semiVariogrammeLocal(self, tab, matrixH, h, dh):
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

    def semiVariogramme(self, tab, matrixH, dh):
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
            variogramme[i] = self.semiVariogrammeLocal(tab, matrixH, tabH[i], dh)

        return variogramme



class Krigeage(Krige):
    def __init__(self, mode, psiMin, psiMax, phiMin, phiMax, dPsi, dPhi, f):
        super().__init__(mode)
        self.psiMin = psiMin
        self.psiMax = psiMax
        self.phiMin = phiMin
        self.phiMax = phiMax
        self.dPsi = dPsi
        self.dPhi = dPhi
        self.f = f



    def varGamma(self, matrixH, matrixH0):
        '''
        varGamme calcul les variogramme matricielle.

        :param:
        gamma: modele du variagramme (fonction).
        matrixH: matrice des distances entre les points evalues.
        matrixH0: matrice des distance entre les points evalues et le point a interpoller.

        :return
        matrixGamma: matrice des variagramme  entre les points evalues.
        matrixGamma0: matrice des variogramme entre les points evalues et le point a interpoller.
        '''
        matrixGamma = np.zeros(matrixH.shape)
        matrixGamma0 = np.zeros(matrixH0.shape)

        for i in range(len(matrixH)):
            for j in range(len(matrixH)):
                matrixGamma[i, j] = self.f(matrixH[i, j])

            matrixGamma0[i] = self.f(matrixH0[i])

        return matrixGamma, matrixGamma0

    def krigeageOrdinnaire(self, tab, x0):
        '''
        krigeageOrdinnaire interpolle la valeur en x0 par un krigeage ordinnaire.

        :param:
        tab: tableau des donnees (colonne: x, y, val).
        x0: tableau position du site a interpoller.

        :return
        z: valeur en x0 (float)
        varEstim: variance estime (float)
        '''
        matrixH, matrixH0 = self.distH(tab, x0)

        matrixGamma, matrixGamma0 = self.varGamma(matrixH, matrixH0)

        mv = np.ones((matrixH0.T).shape)
        mh = np.ones((len(matrixH0) + 1, 1))
        mh[-1] = 0

        matrixA = np.vstack((matrixGamma, mv))
        matrixA = np.hstack((matrixA, mh))

        matrixB = np.vstack((matrixGamma0, [1]))

        w = self.ls(matrixB, matrixA)

        # print(w)

        z = np.dot(w[:-1].T, tab[:, 2])

        # print("Valeur estime: ", z)

        varEstim = np.dot(w[:-1].T, matrixGamma0) + z[-1]

        # print("Ecart type estime: ", varEstim ** 0.5)

        return z[0], varEstim[0][0]


    def ls(self, Y, A):
        return la.inv(A.T @ A) @ (A.T @ Y)

    def start(self, data):
        nPsi = int((self.psiMax - self.psiMin) / self.dPsi)
        nPhi = int((self.phiMax - self.phiMin) / self.dPhi)

        print("nTheta: ", nPsi)
        print("nPhi: ", nPhi)

        tabX0 = np.zeros((nPsi, nPhi, 4)) * np.nan

        for i in range(nPsi):
            for j in range(nPhi):
                tabX0[i][j][0] = psiMin + i * self.dPsi
                tabX0[i][j][1] = phiMin + j * self.dPhi

                tabX0[i][j][2], tabX0[i][j][3] = v2.krigeageOrdinnaire(data, [tabX0[i][j][0], tabX0[i][j][1]])

        tabMesure = np.zeros((nPsi, nPhi))

        for i in range(nPsi):
            for j in range(nPhi):
                tabMesure[i][j] = tabX0[i][j][2]

        plt.figure()
        plt.imshow(tabMesure)
        plt.show()

        return tabX0

def f(h):
    '''f est le variogramme du modele.

    :param
    h: distance (float).

    :return
    Image de h par le variogramme.'''

    if h > 1000:
        return 10000

    a = 10

    return a * h

def g(h):
    '''f est le variogramme du modele.

        :param
        h: distance (float).

        :return
        Image de h par le variogramme.'''

    if h > 1000:
        return 10000

    a = 12

    return a * h

if __name__ == "__main__":
    tab1 = np.array(
        [[0, 0, 2], [1, 0, 1.8], [0, 1, 2], [1, 2, 2], [2, 0, 1.5], [3, 0, 1.4], [3, 2, 1], [4, 0, 1.2], [4, 1, 1],
         [4, 2, 0.8], [4, 3, 0.7], [5, 4, 0.5], [5, 3, 0.7], [5, 2, 0.8], [5, 1, 1]])  # Mesure.
    tab = np.array([[0, 0, 2], [1, 0, 1.8], [0, 1, 2]])  # Mesure.

    v1 = SemiVariogramme("Ordinaire")

    matrixH, matrixH0 = v1.distH(tab)
    print("_______")
    print(matrixH)
    print(matrixH0)

    variogramme = v1.semiVariogramme(tab, matrixH, 2)

    plt.plot(np.arange(0, len(variogramme), 1), variogramme)
    plt.title("Variogramme experimental.")
    plt.xlabel("distance (en m)")
    plt.ylabel("Semi-variogramme.")
    plt.legend()
    plt.show()

    # __________________________________________#



    psiMin = 369137
    psiMax = 369537
    phiMin = 6822872
    phiMax = 6824200
    dPsi = 10
    dPhi = 100

    nPsi = int((psiMax - psiMin) / dPsi)
    nPhi = int((phiMax - phiMin) / dPhi)

    v2 = Krigeage("Ordinaire", psiMin, psiMax, phiMin, phiMax, dPsi, dPhi, g)

    data = np.loadtxt('test.txt', delimiter=',')

    tabX0 = v2.start(data)

    for i in range(len(tabX0)):
        print(i, ": ", tabX0[i][0][2])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in range(len(data)):
        ax.scatter(data[i][0], data[i][1], data[i][2], color='red')

    for i in range(nPsi):
        for j in range(nPhi):
            ax.scatter(tabX0[i][j][0], tabX0[i][j][1], tabX0[i][j][2], color='blue')
    plt.title("Estimation par krigeage ordinaire.")
    plt.xlabel("Phi")
    plt.ylabel("Psi")
    plt.show()
