import numpy.linalg as LA
import numpy as np
import scipy.io as sio # not working for me
# import h5py 
#from scipy.stats import entropy
# from nmf_mur import*
from pgm2matrix import*
#from nmf_als import*
from time import time

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
            # print('init_w: ', self.w.T, 'with size: ', np.shape(self.w.T))
        else:
            self.w = np.matrix(w_init)

        if (h_init is None):
            self.h = np.random.rand(self.r, self.v.shape[1])
            # print('init_h: ', self.h , 'with size: ', np.shape(self.h))
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


#---------------------------Multiplicative Update Rule-----------------------------------#
    def mur_solve(self, tol = None, timelimit = None, max_iter = None, r = None):
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
            print(self.max_iter)

        # n_iter = 0
        for n_iter in range(self.max_iter):
            self.h = np.multiply(self.h, (np.dot(self.w.T, self.v) /  (np.dot(np.dot(self.w.T, self.w), self.h) + 2 ** -8)))
            # denominator = np.dot(np.dot(self.w.T, self.w), self.h) + 2 ** -8
            self.w = np.multiply(self.w, (np.dot(self.v, self.h.T) / (np.dot(self.w, np.dot(self.h, self.h.T)) + 2**-8)))
            # denominator = np.dot(self.w, np.dot(self.h, self.h.T)) + 2**-8

        # print('w', np.shape(self.w), 'h', np.shape(self.h)) # w (10304, 2) h (2, 10)
        # return np.dot(self.w, self.h)
        return self.w
#-------------------------------------------------------------------#

#---------------------------Truncated Cauchy-----------------------------------#


#-------------------------------------------------------------------#

'''
test_mat = np.matrix(
[[100, 1, 234 , 98 ,359],
 [78, 12, 727, 812, 234], 
 [123 , 235 ,6572, 223, 845], 
 [356, 2342, 123, 5634, 234], 
 [235,567, 123, 4365, 243]])

test_nmf = NMF(test_mat, r=5)
#print('nmf type: ', type(test_nmf)) # nmf type:  <class '__main__.NMF'>

print('Initial error is : ', test_nmf.frobenius_norm())
t0 = time()
print(type(test_nmf.mur_solve(max_iter=10000)))
# print(type(test_result))
t1 = time()

print('Final error is: ', test_nmf.frobenius_norm(), 'Time taken: ', t1 - t0)
'''
#-------------------------------------------------------------------#
if __name__ == "__main__":
    # orl_face = pgm2matrix('orl_face/s2class/', 20)
    # face_nmf = NMF(orl_face, r=20)
    # print('Initial error is: ', face_nmf.frobenius_norm())

    # t0 = time()
    # result = face_nmf.mur_solve(max_iter=2000)
    # t1 = time()

    # print('Final error is: ', face_nmf.frobenius_norm(), 'Time taken: ', t1 - t0)
    # # print(result)
    # matrix2png(result, 112, 92, 'results/1116/')
    # print('Final error is: ', face_nmf_mur.frobenius_norm(), 'Time taken: ', t1 - t0)
    # print('target matrix V: \n', orl_face)
    # print('basis matrix W^T: \n', face_nmf_mur.w.T)
    # print('weight matrix h: \n', face_nmf_mur.h)
    # print('---------------------------------------- MUR ----------------------------------------')

    # matrix2png(result, 112, 92, 'results/0611/')
    

#-----------------------YALE B--------------------------------------------#



    # f = h5py.File('YaleB.mat','r') 
    # data = f.get('data/variable1') 
    # data = np.array(data)
    mat = sio.loadmat('YaleB.mat')
    # print(np.shape(mat['fea'])) #(1024, 2414)
    mat_fea = mat['fea']
    fea_nmf = NMF(mat_fea, r=10)

    t0 = time()
    result = fea_nmf.mur_solve(max_iter=1000)
    t1 = time()


    print('Final error is: ', fea_nmf.frobenius_norm(), 'Time taken: ', t1 - t0)
    print(np.shape(result))
    matrix2png(result, 32, 32, 'results/1116/yale_b/')