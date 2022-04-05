import numpy as np
import unittest

import krige


class TestKrige(unittest.TestCase):
    def test_semiVariogrammeLocal(self):
        tab = np.array(
            [[0, 0, 2], [1, 0, 1.8], [0, 1, 2], [1, 2, 2], [2, 0, 1.5], [3, 0, 1.4], [3, 2, 1], [4, 0, 1.2], [4, 1, 1],
             [4, 2, 0.8], [4, 3, 0.7], [5, 4, 0.5], [5, 3, 0.7], [5, 2, 0.8], [5, 1, 1]])  # Mesure.

        v1 = krige.SemiVariogramme("Ordinaire")

        matrixH, matrixH0 = v1.distH(tab)

        dh = 2
        tabH = np.arange(dh / 2, 10 * dh - (dh / 2), dh)

        variogramme1 = v1.semiVariogrammeLocal(tab, matrixH, tabH[0], dh)
        variogramme2 = v1.semiVariogrammeLocal(tab, matrixH, tabH[1], dh)

        self.assertEqual(variogramme1, 0.06933333333333334)
        self.assertEqual(variogramme2, 0.30541666666666667)

    def test_semiVariogramme(self):
        tab = np.array(
            [[0, 0, 2], [1, 0, 1.8], [0, 1, 2], [1, 2, 2], [2, 0, 1.5], [3, 0, 1.4], [3, 2, 1], [4, 0, 1.2], [4, 1, 1],
             [4, 2, 0.8], [4, 3, 0.7], [5, 4, 0.5], [5, 3, 0.7], [5, 2, 0.8], [5, 1, 1]])  # Mesure.

        v1 = krige.SemiVariogramme("Ordinaire")

        matrixH, matrixH0 = v1.distH(tab)

        dh = 2

        variogramme = v1.semiVariogramme(tab, matrixH, dh)

        self.assertEqual(variogramme[0], 0.06933333333333334)
        self.assertEqual(variogramme[1], 0.30541666666666667)

if __name__ == '__main__':
    unittest.main()