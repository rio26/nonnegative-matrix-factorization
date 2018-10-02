import numpy as np
import numpy.linalg.solve

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


    def update_h(self):

        lhs = np.float64(np.dot(self.w.T, self.w))  # lhs - left hand side, Float64
        rhs = np.float64(np.dot(self.w.T))

        self.h = np.linalg.solve(lhs, rhs)



        Q = matrix(WtW)
        G = matrix(-np.eye(self._rank))
        h = matrix(0.0, (self._rank, 1))
        samples = self.X.T
        cvxopt.solvers.options['show_progress'] = False
        for i in range(self._samples):
            p = matrix(np.float64(np.dot(-self.W.T, samples[i])))

            sol = solvers.qp(Q, p, G, h)
            self.H[:, i] = np.array(sol['x']).reshape((1, -1))

    def update_w(self):

        HHt = np.float64(np.dot(self.h, self.h.T))  # Float64 for cvxopt
        Q = matrix(HHt)
        G = matrix(-np.eye(self._rank))
        h = matrix(0.0, (self._rank, 1))
        samples = self.X
        cvxopt.solvers.options['show_progress'] = False
        for i in range(self._samples):
            p = matrix(np.float64(np.dot(-self.H, samples[i].T)))

            sol = solvers.qp(Q, p, G, h)
            self.W[i, :] = np.array(sol['x']).reshape((1, -1))