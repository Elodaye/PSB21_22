import numpy as np
import unittest

import conversionData

class TestConversionData(unittest.TestCase):
    def test_conv(self):
        p = conversionData()

        self.assertEqual(p[0][0], "1")
        self.assertEqual(p[0][1], "2")
        self.assertEqual(p[0][2], "voiture")
        self.assertEqual(p[0][3], 73.0)
        self.assertEqual(p[0][4], 2)

if __name__ == '__main__':
    unittest.main()
