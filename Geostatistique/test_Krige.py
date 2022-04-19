import numpy as np
import unittest

import krige


class TestKrige(unittest.TestCase):
    def test_dist(self):
        v1 = krige.Krige("Ordinaire")

        self.assertEqual(v1.dist([0, 0], [0, 1]), 1.0)

    def test_distH(self):
        tab = np.array([[0, 0, 2], [1, 0, 1.8], [0, 1, 2]])
        v1 = krige.Krige("Ordinaire")

        matrixH, matrixH0 = v1.distH(tab)

        self.assertEqual(matrixH[1][2], 1.4142135623730951)
        self.assertEqual(matrixH0[1][0], 1.0)

if __name__ == '__main__':
    unittest.main()