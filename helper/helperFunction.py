import numpy as np

# Fungsi Matriks D-H
def matrixDH(a, al, d, t):
    """Generate D-H Matrix with parameter a, alpha, d, tetha.
    Returning the matrix
    """
    return np.matrix([[np.cos(t), -np.sin(t)*np.cos(al), np.sin(t)*np.sin(al), a*np.cos(t)],
                      [np.sin(t), np.cos(t)*np.cos(al), -np.cos(t)*np.sin(al), a*np.sin(t)],
                      [0, np.sin(al), np.cos(al), d],
                      [0, 0, 0, 1]])