import numpy as np
import numpy.linalg as LA


class GreenSystem():
    '''
    This class can construct the retarded and keldysh greens function of the system with two leads.
    It can also calculate the bond currents and transmission of the system.
    '''

    def __init__(self, scatter):
        '''
        Constructor intialising the hamiltonian and scattering region parameters. Also sets
        the retarded Greens function (G_r) and the keldysh Greens function to None.

        #PARAMETERS:
        scatter             Needs to be a instance of the ScattererBuilder class

        #ATTRIBUTES:
        self.H              Clean Hamiltonian of the scattering region
        self.orb            Orbitals per site for the Hamiltonian
        self.hsize          Hamiltonian dimensionality
        self.G_r            Retarded Greens function
        self.G_k            Keldysh Greens function
        '''

        self.H = scatter.hamiltonian
        self.orb = scatter.orb
        if(self.orb > 1):
            self.idx = np.transpose(np.where(scatter.idxmap))
        else:
            self.idx = None
        self.hsize = np.size(self.H, axis=0)
        self.G_r = None
        self.G_k = None

    def init_retarded(self, selfL, selfR, E):
        '''
        Constructor intialising the retarded Greens function and calculating gamma matrices for transport.

        #PARAMETERS:
        selfL               self-energy of the left (first) lead
        selfR               self-energy of the right (second) lead
        E                   Fermi energy of the system

        #ATTRIBUTES:
        self.E              Fermi energy of the system
        self.sL             self-energy of the left (first) lead
        self.sR             self-energy of the right (second) lead
        self.ssize          Dimensionality of the self-energy (therefore of the lead unit cell)
        self.Y1             Left (first) gamma matrix of self.sL
        self.Y2             Right (second) gamma matrix of self.sR
        '''
        self.E = E
        self.sL = selfL
        self.sR = selfR
        self.ssize = np.size(self.sL, axis=0)
#        self.Y1 = np.zeros((self.hsize, self.hsize), dtype=complex)
#        self.Y2 = np.zeros((self.hsize, self.hsize), dtype=complex)
#        self.Y1[:self.ssize, :self.ssize] += 1j * (self.sL - self.sL.conj().T)
#        self.Y2[-self.ssize:, -self.ssize:] += 1j * (self.sR - self.sR.conj().T)
        self.Y1 = 1j * (self.sL - self.sL.conj().T)
        self.Y2 = 1j * (self.sR - self.sR.conj().T)
        self.__retarded()

    def __retarded(self):
        '''
        Private function, constructing the retarded Greens function from the __init__ data.
        '''
        hbuff = np.copy(self.H)
        hbuff[:self.ssize, :self.ssize] += self.sL
        hbuff[-self.ssize:, -self.ssize:] += self.sR
        hbuff.flat[::self.hsize + 1] -= self.E
        self.G_r = LA.inv(-hbuff)

    def transmission(self):
        '''
        This function calculates the transmission from the retarded greens function and gamma matrices
        using the Meir-Wingreen formula. To speed up calculation we are exploiting the block structure
        of the gamma matrices.
        '''
        G_lr = self.G_r[:self.ssize, -self.ssize:]
        trans = np.dot(np.dot(self.Y1, G_lr), np.dot(self.Y2, G_lr.conj().T))
        return np.trace(np.real(trans))

    def init_keldysh(self, mu1, mu2, T):
        '''
        This function constructs the keldysh Greens function from the __init__ data and retarded Greens function.

        #PARAMETERS:
        mu1             Chemical potential of the left or first lead
        mu2             Chemical pontential of the right or second lead
        T               Temperature of the system

        #ATTRIBUTES:
        self_k          buffer for the gamma matrices multiplied by their occupation probability
                        also called Keldysh self-energy
        '''
        self_k = np.zeros((self.hsize, self.hsize), dtype=complex)
        mumin = np.min((mu1, mu2))
        mumax = np.max((mu1, mu2))
        if(T == 0 and mumin <= self.E and self.E <= mumax):
            # For zero temperature the fermi dirac distribution of only one lead is important (the one with the higher
            # chemical potential)
            if(mu1 == mumax):
                self_k[:self.ssize, :self.ssize] = 1j*self.Y1
            else:
                self_k[-self.ssize:, -self.ssize:] = 1j*self.Y2
            self.G_k = np.dot(self.G_r, np.dot(self_k, self.G_r.conj().T))
        elif(T == 0 and (mumin > self.E or mumax < self.E)):
            # Empty Keldysh Greens function when the energy is out of chemical potential bounds
            self.G_k = np.zeros((self.hsize, self.hsize), dtype=complex)
        else:
            # For finite temperature we have to consider the fermi dirac distributions of both leads
            beta = 1/T
            prob1 = 1/(np.exp((beta*(self.E-mu1)))+1)
            prob2 = 1/(np.exp((beta*(self.E-mu2)))+1)
            self_k[:self.ssize, :self.ssize] = 1j * prob1 * self.Y1
            self_k[-self.ssize:, -self.ssize:] = 1j * prob2 * self.Y2
            self.G_k = np.dot(self.G_r, np.dot(self_k, self.G_r.conj().T))

    def bond_current(self, *args):
        orb = self.orb
        dim = self.hsize//orb
        self.bcurrent = np.zeros((dim, dim), dtype=float)
        if(orb == 1):
            self.bcurrent = np.real(self.H * self.G_k.T)
        else:
            if(len(args) == 0):
                s = np.eye(orb)
            else:
                s = args[0]
            for i, n in self.idx:
                # Here we only take relevant entries where the lattice sites are actually connected
                t = self.H[orb*i:orb*(i+1), orb*n:orb*(n+1)]
                G = self.G_k[orb*n:orb*(n+1):, orb*i:orb*(i+1)]
                buff = np.dot(t, G)

                # Definition of bond current
                self.bcurrent[i, n] = np.trace(np.real(np.dot(s, buff)))

            self.bcurrent = 1/np.pi * self.bcurrent
        return self.bcurrent

    @property
    def retarded(self):
        return self.G_r

    @property
    def keldysh(self):
        return self.G_k

    def export_bcurrent(self, filestr):
        np.savez(filestr, bcurrent=self.bcurrent)
