import numpy as np
from time import time

# magic numbers
_smallnumber = 1E-5

def als_solve(self, tol = None, timelimit = None, maxiter = None, r = None):
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

    if (maxiter is None):
        self.maxiter = 1000
    else:
        self.maxiter = maxiter

    if (r is None):
        self.r = 2
    else:
        self.r = r

    # gradient of w and h repectively
    grad_w = np.matmul(self.w, np.matmul(self.h, self.h.T)) - np.matmul(self.v, self.h.T)
    grad_h = np.matmul(np.matmul(self.w.T, self.w), self.h) - np.matmul(self.w.T, self.v)

#       for i in range(self.maxiter):
