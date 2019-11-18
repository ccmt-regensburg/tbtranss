import cython

import numpy as np
cimport numpy as np

cdef extern from "ccode/selfalgorithm.h":
    cdef int d_eigendecomposition(const size_t n, void *M, void *T, void *T_t, double E, void *sigma)
    cdef int z_eigendecomposition(const size_t n, void *M, void *T, void *T_t, double E, void *sigma)

    cdef int d_decimation(const size_t n, void *M, void *T, void *T_t, double E, int iters, double eps, void *sigma)
    cdef int z_decimation(const size_t n, void *M, void *T, void *T_t, double E, int iters, double eps, void *sigma)


@cython.boundscheck(False)
@cython.wraparound(False)
def z_eigendecomposition_fast(int n, np.ndarray[complex, ndim=2, mode="c"] M, np.ndarray[complex, ndim=2, mode="c"] T, np.ndarray[complex, ndim=2, mode="c"] T_t, double E, np.ndarray[complex, ndim=2, mode="c"] sigma):

    '''
    Wrapper function to call into the eigendecomposition self-energy calculation coded in pure C for complex hamiltonians
    '''
    z_eigendecomposition(n, &M[0, 0], &T[0, 0], &T_t[0, 0], E, &sigma[0, 0])


@cython.boundscheck(False)
@cython.wraparound(False)
def d_eigendecomposition_fast(int n, np.ndarray[double, ndim=2, mode="c"] M, np.ndarray[double, ndim=2, mode="c"] T, np.ndarray[double, ndim=2, mode="c"] T_t, double E, np.ndarray[complex, ndim=2, mode="c"] sigma):
    '''
    Wrapper function to call into the eigendecomposition self-energy calculation coded in pure C for real hamiltonians
    '''
    d_eigendecomposition(n, &M[0, 0], &T[0, 0], &T_t[0, 0], E, &sigma[0, 0])


@cython.boundscheck(False)
@cython.wraparound(False)
def z_decimation_fast(int n, np.ndarray[complex, ndim=2, mode="c"] M, np.ndarray[complex, ndim=2, mode="c"] T, np.ndarray[complex, ndim=2, mode="c"] T_t, double E, int iters, double eps, np.ndarray[complex, ndim=2, mode="c"] sigma):
    '''
    Wrapper function to call into the decimation self-energy calculation coded in pure C for complex hamiltonians
    '''
    z_decimation(n, &M[0, 0], &T[0, 0], &T_t[0, 0], E, iters, eps, &sigma[0, 0])


@cython.boundscheck(False)
@cython.wraparound(False)
def d_decimation_fast(int n, np.ndarray[double, ndim=2, mode="c"] M, np.ndarray[double, ndim=2, mode="c"] T, np.ndarray[double, ndim=2, mode="c"] T_t, double E, int iters, double eps, np.ndarray[complex, ndim=2, mode="c"] sigma):
    '''
    Wrapper function to call into the decimation self-energy calculation coded in pure C for real hamiltonians
    '''
    d_decimation(n, &M[0, 0], &T[0, 0], &T_t[0, 0], E, iters, eps, &sigma[0, 0])

