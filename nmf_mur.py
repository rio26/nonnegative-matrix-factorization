import numpy as np
from time import time
_smallnumber = 1E-5

def mu_solve(self, tol = None, timelimit = None, max_iter = None, r = None):
    """
    Input:
      -- V: m x n matrix, the dataset

    Optional Input/Output:

      -- tol: tolerance error (stopping condition)
      -- timelimit, maxiter: limit of time and maximum iterations (default 1000)
      -- Output: w, h
      -- r: decompose the marix m x n  -->  (m x r) x (r x n), default 2
    """
    if (tol is None):
        self.tol = _smallnumber
    else:
        self.tol = tol

    if (timelimit is None):
        self.timelimit = 3600
    else:
        self.timelimit = timelimit

    if (max_iter is None):
        self.max_iter = 1000
    else:
        self.max_iter = max_iter

    # n_iter = 0
    for n_iter in range(self.max_iter):
        self.h = np.multiply(self.h, (np.dot(self.w.T, self.v) /  (np.dot(np.dot(self.w.T, self.w), self.h) + 2 ** -8)))
        # denominator = np.dot(np.dot(self.w.T, self.w), self.h) + 2 ** -8
        self.w = np.multiply(self.w, (np.dot(self.v, self.h.T) / (np.dot(self.w, np.dot(self.h, self.h.T)) + 2**-8)))
        # denominator = np.dot(self.w, np.dot(self.h, self.h.T)) + 2**-8
