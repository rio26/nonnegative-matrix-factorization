import numpy as np
import numpy.linalg as LA
from scipy.stats import entropy
from nmf_mur import*
from nmf_als import*

# magic numbers
_smallnumber = 1E-5

# Multiplicative Update Rule for solving NMF
class NMF:
    """
    Input:
      -- V: m x n matrix, the dataset

    Optional Input/Output:
      -- n_components: desired size of the basis set, default

      -- w_init: basis matrix with size m x r
      -- h_init: weight matrix with size r x n  (want r as small as possible)
      -- tol: tolerance error (stopping condition)
      -- timelimit, maxiter: limit of time and maximum iterations (default 1000)
      -- Output: w, h
    """
    def __init__(self, v, w_init = None, h_init = None, r = None):
        self.v = v

        if (r is None):
            self.r = 2
        else:
            self.r = r

        if (w_init is None):
            self.w = np.random.rand(self.v.shape[0], self.r)
        else:
            self.w = np.matrix(w_init)

        if (h_init is None):
            self.h = np.random.rand(self.r, self.v.shape[1])
        else:
            self.h = np.matrix(h_init)

    def frobenius_norm(self):
        """ Euclidean error between v and w*h """

        if hasattr(self, 'h') and hasattr(self, 'w'):  # if it has attributes w and h
            error = LA.norm(self.v - np.dot(self.w, self.h))
        else:
            error = None
        return error

    def kl_divergence(self):
        """ KL Divergence between X and W*H """

        if hasattr(self, 'h') and hasattr(self, 'w'):
            error = entropy(self.v, np.dot(self.w, self.h)).sum()
        else:
            error = None
        return error

#test_mat = np.matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
test_mat = np.matrix([[100, 1, 234], [78, 12, 727], [123 , 235 ,6572]])
test_nmf = NMF(test_mat)
print(test_nmf.frobenius_norm())
mu_solve(test_nmf)
print(test_nmf.frobenius_norm())