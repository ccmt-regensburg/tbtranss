import numpy as np
import numpy.linalg as LA
from mpi4py import MPI
from ..multicore import MpiHelpers


class Spectrum():
    '''
    Class for solving the 3d tight binding Hamiltonian infinite in one dimension for its
    wave functions and band structures. Naturally this also includes the density of states.
    '''

    def __init__(self, Hlist, sfactor, krange):
        self.Hlist = Hlist
        self.Hsize = np.size(Hlist, axis=0)
        self.sfactor = int(sfactor)                                         # Depicts if we deal with a reduced B-zone
        krange[0] = krange[0]/self.sfactor
        krange[1] = krange[1]/self.sfactor
        self.krange = krange                                                # List with k-values one needs to calculate

        # Instance of the MpiHelper class giving access to listpartition functions
        self.multi = MpiHelpers()

    def __calc_eigv_eigvec(self, exp, exp_c):                               # Gives eigenvalues and eigenvectors of the hamiltonian
        H = self.Hlist[0] + np.einsum('ijk,i', self.Hlist[1:], exp) + np.einsum('ijk,i', np.transpose(self.Hlist[1:], axes=(0, 2, 1)).conjugate(), exp_c)
        return LA.eigh(H)

    def ham_eigv_eigvec(self):
        klist, klocal, ptuple, displace = self.multi.listpart(self.krange)
        self.multi.comm.Scatterv([klist, ptuple, displace, MPI.DOUBLE], klocal)

        local_energies = []
        local_eigvec = []

        # This allows for phase factors bigger than exp(1j)
        # meaning next nearest unit cell hopping
        expmat = []
        expmat_c = []
        for i in range(self.Hsize - 1):
            explist = np.exp(self.sfactor*(i+1) * 1j * klocal)
            expmat.append(explist)
            expmat_c.append(explist.conjugate())
        expmat = np.array(expmat)
        expmat_c = np.array(expmat_c)

        for exp, exp_c in zip(expmat.T, expmat_c.T):
            e, v = self.__calc_eigv_eigvec(exp, exp_c)                      ##Calculates eigenvalues and eigenvectors on each seperate core
            local_energies.append(e)
            local_eigvec.append(v)

        self.multi.comm.Barrier()
        if(self.multi.rank == 0):
            ksize = self.krange[2]                                          #Length of k-vector
            matdim = np.shape(self.Hlist[0])[0]                                    #Dimension of Hamiltonian
            esize = ksize*matdim                                            #Total number of energies
            eigvecsize = ksize*matdim*matdim                                #Total number of eigenvector entries
            edisplace = np.array(displace)*matdim                           #Displacement of energies in total eigenvalue vector
            eigvecdisplace = np.array(displace)*matdim*matdim               #Displacement of eigenvectors in total eigenvector vector
            eptuple = np.array(ptuple)*matdim                               #Total amount of energies per core
            eigvecptuple = np.array(ptuple)*matdim*matdim                   #Total amount of eigenvector entries per core
            ebuffer = np.empty( esize, dtype=float )                        #Receive buffer for all energies
            eigvecbuffer = np.empty( eigvecsize, dtype=complex )            #Receive buffer for all eigenvector entries
        else:
            ksize = None
            matdim = None
            esize = None
            eigvecsize = None
            edisplace = None
            eigvecdisplace = None
            eptuple = None
            eigvecptuple = None
            ebuffer = None
            eigvecbuffer = None

        self.multi.comm.Gatherv(np.array(local_energies).flatten(), [ebuffer, eptuple, edisplace, MPI.DOUBLE], root=0)
        self.multi.comm.Gatherv(np.array(local_eigvec).flatten(), [eigvecbuffer, eigvecptuple, eigvecdisplace, MPI.DOUBLE_COMPLEX], root=0)

        if(self.multi.rank == 0):
            energies = ebuffer.reshape((ksize, matdim))
            eigvecs = eigvecbuffer.reshape((ksize, matdim, matdim))
            return np.array(klist), np.array(energies), np.array(eigvecs)
        else:
            klist = None
            energies = None
            eigvecs = None
            return np.array(klist), np.array(energies), np.array(eigvecs)

    def ldos_lorenz(self, energies, eigvecs, efermi, orbitals, eps, cutoff):

        cutoff = eps*np.sqrt((1./cutoff - 1))                              #Energetic cutoff calculated from the cutoff percentage
        def lcurve(e):
            return 1/np.pi * eps/(e*e + eps*eps)                            #Definition of the Lorentz curve

        enediff = efermi - energies
        ldos = np.zeros(len(eigvecs[0][:, 0]), dtype=complex)

        for dlist, eigvlist in zip(enediff, eigvecs):
            for i, diff in enumerate(dlist):
                if( diff >= 0 and diff < cutoff ):
                    ldos += np.multiply(eigvlist[:, i], np.conjugate(eigvlist[:, i])) * lcurve(diff)

        ldos = ldos.astype(dtype = float)
        ldos = [sum(ldos[i: i+orbitals]) for i in range(0, len(ldos), orbitals)]
        return ldos
