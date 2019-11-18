import numpy as np
import numpy.linalg as LA
import scipy.linalg as sLA
from .selfe_cython import d_eigendecomposition_fast, z_eigendecomposition_fast, d_decimation_fast, z_decimation_fast


class SelfEnergy():
    '''
    Class for constructing the self energy of a 1d lead.
    '''

    def __init__(self, Hlist, idx):
        '''
        Intialise all important parameters for the self-energy calculation.
        If the calculation is done with the 'disjoined' argument it yields the self-enery
        of the lead without considerations about the scattering region.

        If the calculation is done with the 'attach' argument the self-energy is returned with the
        right format and index reording to add it to the scattering region.

        # Function parameters:

        *Hlist                          Lead Hamiltonian

        idx                             0 is left lead, 1 is right lead
        '''

        if idx == 0:
            self.M = np.ascontiguousarray(Hlist[0])                  # Onsite matrix
            self.T = np.ascontiguousarray(Hlist[1])                  # Next hopping slice matrix
            self.T_t = np.ascontiguousarray(self.T.T.conjugate())    # Adjoint of hopping matrix
        if idx == 1:
            self.M = np.ascontiguousarray(Hlist[0])
            self.T_t = np.ascontiguousarray(Hlist[1])
            self.T = np.ascontiguousarray(self.T_t.T.conjugate())

        self.hsize = np.size(self.M, axis=0)

    def __quad_mat_construct(self):
        '''
        Private method constructing the relevant matrices for the general eigenproblem.
        '''
        shape = (self.hsize, self.hsize)

        # Construct top + bottom row of left matrix in quadratic eigenproblem
        topr = np.concatenate((np.zeros(shape), np.eye(shape[0])), axis=1)
        botr = np.concatenate((-1*self.T_t, self.ene*np.eye(shape[0])-self.M), axis=1)

        # Left matrix of quadratic eigenproblem
        Q_l = np.concatenate((topr, botr), axis=0)

        # Construct top + bottom row of right matrix in quadratic eigenproblem
        topr = np.concatenate((np.eye(shape[0]), np.zeros(shape)), axis=1)
        botr = np.concatenate((np.zeros(shape), self.T), axis=1)

        # Right matrix of quadratic eigenproblem
        Q_r = np.concatenate((topr, botr), axis=0)

        return Q_l, Q_r

    def __general_eig_solver(self, Q_l, Q_r):
        '''
        Solver for the general eigenproblem.
        '''
        eig, eigv = sLA.eig(Q_l, b=Q_r, left=False, right=True, overwrite_a=True, overwrite_b=True, check_finite=False)

        # Throw away bottom half of each eigenvector
        # and renormalise to top half
        eigv = eigv[:self.hsize]
        eigv = eigv / np.linalg.norm(eigv, axis=0)

        return eig, eigv

    def __mode_filter(self, eig, eigv):
        '''
        Method finding the modes that lie inside the integration circle of the
        complex integral of the Greens function.
        '''

        # Find eigenvalues and eigenvectors with abs <= 1
        condition = (np.absolute(eig) < 1 + 10e-10)
        eig = eig[condition]
        eigv = eigv[:, condition]

        # Find propagating modes
        condition = np.isclose(np.absolute(eig), 1, rtol=0, atol=1e-10)
        prop_eig = eig[condition]
        prop_eigv = eigv[:, condition]

        # Find evanescent modes
        eva_eig = np.extract(np.logical_not(condition), eig)
        eva_eigv = eigv[:, np.logical_not(condition)]

        # Find degeneracies in the collection of propagating modes
        if(prop_eig.size != 0):
            comp = self.__degeneracy_checker(prop_eig)
            prop_eig, prop_eigv = self.__velocity_checker(prop_eig, prop_eigv, comp)

        # Put together evanescent and propagating solutions
        # inside the integration circle
        eig = np.concatenate((eva_eig, prop_eig))
        eigv = np.concatenate((eva_eigv, prop_eigv), axis=1)
        return eig, eigv

    def __degeneracy_checker(self, prop_eig):
        '''
        Method checking for degenerate modes in the collection of propagating modes.
        '''
        eig_r = np.real(prop_eig)
        eig_i = np.imag(prop_eig)

        # Vertically stacked lists of eigenvalues
        Eig_r = np.repeat(eig_r[:, np.newaxis], len(eig_r), axis=1)
        Eig_i = np.repeat(eig_i[:, np.newaxis], len(eig_i), axis=1)

        # Compare every element to all other elements
        comp_r = np.isclose(eig_r, Eig_r, rtol=0, atol=1e-10)
        comp_i = np.isclose(eig_i, Eig_i, rtol=0, atol=1e-10)

        # Eigenvalues where imaginary and real part are equal
        comp = np.logical_and(comp_r, comp_i)

        if(np.count_nonzero(comp) == len(prop_eig)):
            # No degeneracie -> directly give matrix
            return comp
        else:
            # Remove duplicate rows for degeneracies
            b = np.ascontiguousarray(comp).view(np.dtype((np.void, comp.dtype.itemsize * comp.shape[1])))
            _, idx = np.unique(b, return_index=True)
            return comp[idx][::-1]

    def __velocity_checker(self, prop_eig, prop_eigv, comp):
        '''
        Method calculating the velocity of propagating modes
        '''
        eig_result = []
        eigv_result = []
        for comp_row in comp:
            # Pick all modes belonging to one specific eigenvalue
            eig_val = prop_eig[comp_row]
            eig_vec = prop_eigv[:, comp_row]

            # Velocity operator with specific eigenvalue
            V = 1j*(self.T * eig_val[0] - self.T_t * eig_val[0].conj())
            V = np.dot(V, eig_vec)
            V = np.dot(eig_vec.T.conjugate(), V)

# This velocity operator also works. Not sure why and when it should not work
#            V2 = self.T * eig_val[0]
#            V2 = np.dot(V2, eig_vec)
#            V2 = np.dot(eig_vec.T.conjugate(), V2)
#            V2 = -2*np.imag(V2)

            # For degenerate modes V is a matrix for non-degenerate ones a scalar
            if(np.shape(V) == (1, 1) and np.real(V) > 0):
                # Append right going mode
                eig_result.append(eig_val)
                eigv_result.append(eig_vec)
            elif(np.shape(V) != (1, 1)):
                # Here we need to diagonalise the degenerate subspace
                vel, subeigv = LA.eigh(V)

                # Rotate the eigenvectors to decouple them
                eig_vec = np.dot(eig_vec, subeigv)

                # Check for right going modes
                subcomp = vel > 0

                # Append right going modes
                eig_result.append(eig_val[subcomp])
                eigv_result.append(eig_vec[:, subcomp])

        return np.concatenate(eig_result), np.concatenate(eigv_result, axis=1)

    def eigendecomposition(self, E):
        '''
        Calculates the self energy of the lead via the eigendecomposition method.
        '''
        self.ene = E

        # Construct and solve the general eigenproblem
        Q_l, Q_r = self.__quad_mat_construct()
        eig, eigv = self.__general_eig_solver(Q_l, Q_r)

        # Find evanescent modes and prop. modes with positive velocity
        eig, eigv = self.__mode_filter(eig, eigv)
        eig = np.diag(eig)
        eigv_inv = LA.inv(eigv)

        # Self energy of the lead
        sigma = np.dot(np.dot(self.T, eigv), np.dot(eig, eigv_inv))
        return sigma

    def decimation(self, E, iterations, eta):
        '''
        Calculates the self energy of the lead via the decimation technique.
        '''
        self.ene = E
        mbuff = np.copy(self.M)
        mbuff.flat[::self.hsize + 1] -= E + 1j * eta
        sigma = np.zeros((self.hsize, self.hsize), dtype=complex)
        for i in range(iterations):
            green = LA.inv(-mbuff - sigma)
            sigma = np.dot(self.T, np.dot(green, self.T_t))

        return sigma

    def eigendecomposition_fast(self, E):
        sigma = np.zeros((self.hsize, self.hsize), dtype=complex)
        M = self.M
        T = self.T
        T_t = self.T_t

        if (self.M.dtype == np.complex or self.T.dtype == np.complex):
            z_eigendecomposition_fast(self.hsize, M.astype(np.complex), T.astype(np.complex),
                                      T_t.astype(np.complex), E, sigma)
        else:
            d_eigendecomposition_fast(self.hsize, M, T, T_t, E, sigma)

        return sigma

    def decimation_fast(self, E, iters, eta):
        sigma = np.empty((self.hsize, self.hsize), dtype=complex)
        M = self.M
        T = self.T
        T_t = self.T_t

        if (self.M.dtype == np.complex or self.T.dtype == np.complex):
            z_decimation_fast(self.hsize, M.astype(np.complex), T.astype(np.complex), T_t.astype(np.complex), E, iters,
                              eta, sigma)
        else:
            d_decimation_fast(self.hsize, M, T, T_t, E, iters, eta, sigma)
