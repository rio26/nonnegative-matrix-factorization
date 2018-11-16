import numpy as np
from scipy import sparse
from time import time

# Some magic numbes
_largenumber = 1E100
_smallnumber = 1E-5

# Multiplicative Update Rule for solving NMF
class NMF:

    """
    Input:
      -- V: m x n matrix, the dataset

    Optional Input/Output:
      -- n_components: desired size of the basis set, default

      -- W: m x n_components matrix, the W matrix, usually interpreted as the basis set
      -- H: n_components x n matrix, the H matrix, usually interpreted as the weight
      -- I: m x n matrix, the weight, (usually) the inverse variance
      -- M: m x n binary matrix, the mask, False means missing/undesired data

    """
    def __init__(self, V, W=None, H=None, I=None, M=None, n_components=5):
        self.V = np.copy(V)
        if (np.count_nonzero(self.V < 0) > 0):
            print("There are negative values in X. Setting them to be zero...", flush=True)
            self.V[self.V < 0] = 0.

        self.n_components = n_components
        self.maxiters = 1000
        self.tol = _smallnumber

        if (W is None):
            self.W = np.random.rand(self.V.shape[0], self.n_components)
        else:
            if (W.shape != (self.V.shape[0], self.n_components)):
                raise ValueError("Initial W has wrong shape.")
            self.W = np.copy(W)
        if (np.count_nonzero(self.W < 0) > 0):
            print("There are negative values in W. Setting them to be zero...", flush=True)
            self.W[self.W < 0] = 0.

        if (H is None):
            self.H = np.random.rand(self.n_components, self.V.shape[1])
        else:
            if (H.shape != (self.n_components, self.V.shape[1])):
                raise ValueError("Initial H has wrong shape.")
            self.H = np.copy(H)
        if (np.count_nonzero(self.H < 0) > 0):
            print("There are negative values in H. Setting them to be zero...", flush=True)
            self.H[self.H < 0] = 0.

        if (I is None):
            self.I = np.ones(self.V.shape)
        else:
            if (I.shape != self.V.shape):
                raise ValueError("Initial I(Weight) has wrong shape.")
            self.I = np.copy(I)
        if (np.count_nonzero(self.I < 0) > 0):
            print("There are negative values in I. Setting them to be zero...", flush=True)
            self.I[self.I < 0] = 0.

        if (M is None):
            self.M = np.ones(self.V.shape, dtype=np.bool)
        else:
            if (M.shape != self.V.shape):
                raise ValueError("M(ask) has wrong shape.")
            if (M.dtype != np.bool):
                raise TypeError("M(ask) needs to be boolean.")
            self.M = np.copy(M)

            # Set masked elements to be zero
        self.I[(self.I * self.M) <= 0] = 0
        self.I_size = np.count_nonzero(self.I)

    @property
    def cost(self):
        """
        Total cost of a given set S
        """
        diff = self.V - np.dot(self.W, self.H)
        chi2 = np.einsum('ij,ij', self.I * diff, diff) / self.I_size
        return chi2

    def SolveNMF(self, W_only=False, H_only=False, sparsemode=False, maxiters=None, tol=None):
        """
        Construct the NMF basis

        Keywords:
            -- W_only: Only update W, assuming H is known
            -- H_only: Only update H, assuming W is known
               -- Only one of them can be set

        Optional Input:
            -- tol: convergence criterion, default 1E-5
            -- maxiters: allowed maximum number of iterations, default 1000

        Output:
            -- chi2: reduced final cost
            -- time_used: time used in this run


        """

        t0 = time()

        if (maxiters is not None):
            self.maxiters = maxiters
        if (tol is not None):
            self.tol = tol

        chi2 = self.cost
        oldchi2 = _largenumber

        if (W_only and H_only):
            print("Both W_only and H_only are set to be True. Returning ...", flush=True)
            return (chi2, 0.)

        if (sparsemode == True):
            I = sparse.csr_matrix(self.I)
            IT = sparse.csr_matrix(self.I.T)
            multiply = sparse.csr_matrix.multiply
            dot = sparse.csr_matrix.dot
        else:
            I = np.copy(self.I)
            IT = I.T
            multiply = np.multiply
            dot = np.dot

        # VI = self.V*self.I
        VI = multiply(I, self.V)
        VIT = multiply(IT, self.V.T)

        niter = 0

        while (niter < self.maxiters) and ((oldchi2 - chi2) / oldchi2 > self.tol):

            # Update H
            if (not W_only):
                H_up = dot(VIT, self.W)
                WHIT = multiply(IT, np.dot(self.W, self.H).T)
                H_down = dot(WHIT, self.W)
                self.H = self.H * H_up.T / H_down.T

            # Update W
            if (not H_only):
                W_up = dot(VI, self.H.T)
                WHI = multiply(I, np.dot(self.W, self.H))
                W_down = dot(WHI, self.H.T)
                self.W = self.W * W_up / W_down

            # chi2
            oldchi2 = chi2
            chi2 = self.cost

            # Some quick check. May need its error class ...
            if (not np.isfinite(chi2)):
                raise ValueError("NMF construction failed, likely due to missing data")

            if (np.mod(niter, 20) == 0):
                print("Current Chi2={0:.4f}, Previous Chi2={1:.4f}, Change={2:.4f}% @ niters={3}".format(chi2, oldchi2,
                                                                                                         (
                                                                                                                     oldchi2 - chi2) / oldchi2 * 100.,
                                                                                                         niter),
                      flush=True)

            niter += 1
            if (niter == self.maxiters):
                print("Iteration in re-initialization reaches maximum number = {0}".format(niter), flush=True)

        print("Took {0:.3f} seconds to reach current solution.".format(time() - t0), flush=True)
        print("Took {0:.3f} minutes to reach current solution.".format((time() - t0) / 60), flush=True)

        return (chi2, (time() - t0) / 60)

#test_mat = np.matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
test_mat = np.matrix([[1, 1], [1, 1]])
test = NMF(test_mat)
test.SolveNMF()