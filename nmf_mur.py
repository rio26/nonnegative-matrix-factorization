import numpy as np
_smallnumber = 1E-5

def mur_solve(matrix_v, tol = None, timelimit = None, max_iter = None, r = None):
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
        tol = _smallnumber
    else:
        tol = tol

    if (timelimit is None):
        timelimit = 3600
    else:
        timelimit = timelimit

    if (max_iter is None):
        max_iter = 1000
    else:
        max_iter = max_iter

    # # n_iter = 0
    # for n_iter in range(self.max_iter):
    #     self.h = np.multiply(self.h, (np.dot(self.w.T, self.v) /  (np.dot(np.dot(self.w.T, self.w), self.h) + 2 ** -8)))
    #     # denominator = np.dot(np.dot(self.w.T, self.w), self.h) + 2 ** -8
    #     self.w = np.multiply(self.w, (np.dot(self.v, self.h.T) / (np.dot(self.w, np.dot(self.h, self.h.T)) + 2**-8)))
    #     # denominator = np.dot(self.w, np.dot(self.h, self.h.T)) + 2**-8
    
    # n_iter = 0
    for n_iter in range(max_iter):
        matrix_v.h = np.multiply(matrix_v.h, (np.dot(matrix_v.w.T, matrix_v.v) /  (np.dot(np.dot(matrix_v.w.T, matrix_v.w), matrix_v.h) + 2 ** -8)))
        # denominator = np.dot(np.dot(self.w.T, self.w), self.h) + 2 ** -8
        matrix_v.w = np.multiply(matrix_v.w, (np.dot(matrix_v.v, matrix_v.h.T) / (np.dot(matrix_v.w, np.dot(matrix_v.h, matrix_v.h.T)) + 2**-8)))
        # denominator = np.dot(self.w, np.dot(self.h, self.h.T)) + 2**-8
