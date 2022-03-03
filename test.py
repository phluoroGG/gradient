import unittest
import main
import numpy as np


class TestGradient(unittest.TestCase):
    def test_analytical_numerical_gradient(self):
        e = 1e-3
        for x in np.arange(-5, 5, 0.05):
            for y in np.arange(-5, 5, 0.05):
                self.assertTrue(np.abs(main.analytical_derivative_x(x, y)
                                       - main.numerical_derivative_x(x, y, 1e-3)) < e)
                self.assertTrue(np.abs(main.analytical_derivative_y(x, y)
                                       - main.numerical_derivative_y(x, y, 1e-3)) < e)


if __name__ == '__main__':
    unittest.main()
