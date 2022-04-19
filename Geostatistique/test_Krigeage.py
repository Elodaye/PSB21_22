import numpy as np
import unittest

import krige


class TestKrige(unittest.TestCase):
    def test_varGamma(self):
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

        psiMin = 369137
        psiMax = 369537
        phiMin = 6822872
        phiMax = 6824200
        dPsi = 10
        dPhi = 100


        tab = np.array([[0, 0, 2], [1, 0, 1.8], [0, 1, 2]])
        v1 = krige.Krigeage("Ordinaire", psiMin, psiMax, phiMin, phiMax, dPsi, dPhi, g)

        matrixH, matrixH0 = v1.distH(tab)

        matrixGamma, matrixGamma0 = v1.varGamma(matrixH, matrixH0)

        self.assertEqual(matrixGamma[0][1], 12.)
        self.assertEqual(matrixGamma0[1][0], 12.)

    def test_krigeageOrdinnaire(self):
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

        psiMin = 369137
        psiMax = 369537
        phiMin = 6822872
        phiMax = 6824200
        dPsi = 10
        dPhi = 100

        tab = np.array([[0, 0, 2], [1, 0, 1.8], [0, 1, 2]])
        x0 = [0.5, 0.5]
        v1 = krige.Krigeage("Ordinaire", psiMin, psiMax, phiMin, phiMax, dPsi, dPhi, g)

        z, varEstim = v1.krigeageOrdinnaire(tab, x0)

        self.assertEqual(z, 1.9226540919661135)
        self.assertEqual(varEstim, 10.407935466204744)

    def test_ls(self):
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

        psiMin = 369137
        psiMax = 369537
        phiMin = 6822872
        phiMax = 6824200
        dPsi = 10
        dPhi = 100

        tab = np.array([[0, 0, 2], [1, 0, 1.8], [0, 1, 2]])
        x0 = [0.5, 0.5]
        v1 = krige.Krigeage("Ordinaire", psiMin, psiMax, phiMin, phiMax, dPsi, dPhi, g)

        y = np.array([1, 2, 3.1, 4.3, 5.1])
        a = np.array([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]])

        x = v1.ls(y, a)

        self.assertEqual(x[0], 1.0500000000000012)
        self.assertEqual(x[1], 0.9999999999999982)


if __name__ == '__main__':
    unittest.main()
    