import numpy as np                      # NOQA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # NOQA

from ..error.error import CoordinateError


class LeadBuilder():
    r"""Set docstring here.
    This class constructs the Hamiltonian onsite and hopping matrices from the geometrical information
    provided with the user lattice and translational vector.
    Returns a list of matrices with the first being the onsite Hamiltonian then hopping Hamiltonian etc.

    Parameters
    ----------
    lat: np.ndarray
        Geometrical lattice information. Needs to have dimension (*, 3)
    tvector: np.ndarray
        Translational vector of the quasi 1d wire.
    **kwargs: 'orb'; int, optional
        Specifies the orbitals per site inside the system

    Returns
    -------
    LeadBuilder:
        Returns an empty lead builder instance (apart from geometrical information).
        The Hamiltonian can be filled by calling the corresponding hopping methods.
    """

    def __init__(self, lat, tvector, **kwargs):
        '''
        #Attributes:
        self.lattice        Array saving all lattice points of the starting slice. (start lattice)
        self.tvector        Translational vector of the lead lattice.
        self.reps           Range of unit slice hopping that can be included. self.reps = 3 is the default
                            and allows for next nearest unit slice hopping. Therefore this specifies the
                            target lattice for the hoppings.
        self.tlattice       Array including the original slice and (as default) two additional ones along
                            the translational vector direction. (target lattice)
        self.index          Array closely tied to self.tlattice holding the indices of lattice points.
                            For example self.index[0] is the index of self.tlattice[0]

        self.orb            Integer defining the number of orbitals per site.
        self.latmatdim      Dimensionality of one unit slice in the hamiltonian (latmatdim, latmatdim)
        self.__hamiltonian  Internal Hamiltonian variable of the system. (as default) consists of three
                            vertically stacked (self.latmatdim, self.latmatdim) matrices filled with zeros.
                            These are the onsite, nearest unit cell and next nearest unit cell hopping matrices.
        self.htemplate      Hamiltonian template with dimensionality of self.__hamiltonian without the orbitals.
                            Is used to add hoppings to the hamiltonian.

        self.__idxmap       Boolean indexmap. Has the dimensionality (tlatsize, latsize).
                            Has a true value for each connecting hopping in the hamiltonian.
                            Everyting below the lower tridiagonal is written to.
        '''
        # Build scatterer lattice from lead information
        self.lattice = lat                                          # Calls the lattice setter storing the whole lattice
        self.tvector = tvector

        # Add a target lattice including the neighbouring two unit cells
        # The self.reps parameter defines how many neighbouring unit cells we
        # include to define a hopping.
        self.reps = int(3)
        tvectors = np.arange(self.reps)[:, np.newaxis] * self.tvector
        lat = tvectors[:, np.newaxis] + lat
        lat = np.concatenate(lat)

        self.tlattice = lat

        latsize = np.size(self.lattice, axis=0)                     # Number of lattice sites in starting lattice
        tlatsize = np.size(self.tlattice, axis=0)
        self.index = np.arange(tlatsize)                            # Array with indices of vectors in self.tlattice

        # Variables to work with while constructing the Hamiltonian matrix
        self.orb = kwargs['orb']                                            # Parameter for the orbitals per site
        self.latmatdim = self.orb*latsize                                   # Dimension of a unit slice matrix
        self.__hamiltonian = np.zeros((self.orb*tlatsize, self.latmatdim), dtype=complex)
        self.htemplate = np.zeros((tlatsize, latsize), dtype=complex)
        self.__idxmap = self.htemplate.astype(bool, copy=True)              # Empty indexmap

    def hop_all(self, vec, hpar):
        '''
        Applies a hopping to all points that can be connected by a provided vector.

        #Function parameters:
        vec                 User provided vector for the hopping direction
        hpar                Hopping parameter

        '''
        # Call the hamsetter with the full lattice and corresponding indices
        vec, hpar = self.__directionchecker(vec, hpar)
        sidx, tidx = self.__hamchecker(vec, self.lattice, self.index)
        self.__hamsetter(hpar, sidx, tidx)

    def hop_region(self, vec, hpar, boundary):
        '''
        Applies a hopping to points specified inside a region that can be connected by a provided vector.

        #Function parameters:
        vec                 User provided vector for the hopping direction
        hpar                Hopping parameter
        boundary            User provided function specifying the region in which to consider points.
                            Function needs to return True if point is inside or False if outside.
        '''
        # These two arrays go together and have to be manipulated together
        lat = self.lattice
        lidx = self.index

        # If the argument is set as an boundary function reduce the total lattice size
        if callable(boundary):
            inidx = np.apply_along_axis(boundary, 1, lat)
            if np.any(inidx) is False:
                raise CoordinateError('No lattice points lie inside the specified boundary.')

            # Take only those coordinates and indices that fullfill the boundary function
            lat = lat[inidx]
            lidx = lidx[inidx]
        else:
            raise TypeError('The given boundary is not a valid callable python function or method.')

        # Call the hamchecker with the partial lattice and corresponding indices
        vec, hpar = self.__directionchecker(vec, hpar)
        sidx, tidx = self.__hamchecker(vec, lat, lidx)
        self.__hamsetter(hpar, sidx, tidx)

    def hop_point(self, vec, hpar, point):
        '''
        Applies a hopping parameter to a point along a vector.

        #Function parameters:
        vec                 User provided vector for the hopping direction
        hpar                Hopping parameter
        point               Needs to be a numpy.ndarray with three entries.
        '''
        # These two arrays go together and have to be manipulated together
        lat = self.lattice
        lidx = self.index

        if isinstance(point, np.ndarray) and point.ndim == 1 and point.size == 3:
            inidx = np.all(np.isclose(point, self.lattice), axis=1)
            if np.any(inidx) is False:
                raise CoordinateError('The given lattice point is not part of the lattice.', point)

            # Take only those parts of the lattice that correspond to 'point'
            lat = lat[inidx]
            lidx = lidx[inidx]
        else:
            raise TypeError('The given point is not a numpy.ndarray with three entries [x, y, z].')

        # Call the hamchecker with only the point and corresponding index
        vec, hpar = self.__directionchecker(vec, hpar)
        sidx, tidx = self.__hamchecker(vec, lat, lidx)
        self.__hamsetter(hpar, sidx, tidx)

    def hop_nn(self, hpar, n):
        '''
        Applies a hopping parameter to all nearest neighbours or next nearest neighbours etc. depending on
        the input parameters.
        If a complex hopping is given the direction of the normal and conjugate hopping
        are choosen by the method.

        #Function parameters:
        hpar                Hopping parameter
        n                   Integer specifying the hopping range (n=0 onsite, n=1 next nearest ...)
        '''
        lat = self.lattice
        lidx = self.index

        # This matrix contains the distance of every point to every point
        stvecs = self.tlattice[:, np.newaxis] - lat
        dist = np.linalg.norm(stvecs, axis=2)
        mind = 0

        # If n=0 mind=0, if n=1 drop 0 out of dist and find first minimum,
        # if n=2 drop first minimum out of dist and find second minimum
        eps = 1e-12
        for i in range(n):
            mind = np.min(dist[dist > mind + eps])

        # Find which distances correspond to the minimum distance
        check = np.isclose(mind, dist)
        self.idx = np.where(check)

        # This gives what directional type of hopping we deal with
        hopdir = stvecs[self.idx]

        # Here we find the unique hopping vectors without their
        # negative (hermitian conjugate) parts.
        uqhops = []

        def finduq(hopdir):
            '''
            Function to call recursively. Gives the unique vectors in an array of vectors
            also dropping the negative (hermitian conjugate) of those vectors.
            '''
            vec = hopdir[0]

            # For a given vector check the array of all hoppings for equal or negative equal vectors
            check = np.logical_or(np.all(np.isclose(vec, hopdir), axis=1), np.all(np.isclose(-vec, hopdir), axis=1))

            # This is the break condition if this is true the next hopdir would be empty.
            if np.all(check):
                uqhops.append(vec)
                return uqhops
            # This is the continue condition calling finduq with a smaller set of hopdir
            else:
                uqhops.append(vec)
                finduq(hopdir[np.logical_not(check)])

        finduq(hopdir)

        # For each unique hopping found, do the hop_all procedure.
        for vec in uqhops:
            vec, hpar = self.__directionchecker(vec, hpar)
            sidx, tidx = self.__hamchecker(vec, lat, lidx)
            self.__hamsetter(hpar, sidx, tidx)

    def __directionchecker(self, vec, hpar):
        '''
        Private method checking wheter the specified hopping vector points in the direction of the
        translational vector or not.
        If it does not turn the vector around and treat the hermitian conjugate hopping.
        '''
        dotcheck = np.dot(vec, self.tvector)
        if dotcheck < 0 - 10e-12:
            if isinstance(hpar, np.ndarray):
                return -vec, hpar.T.conjugate()
            else:
                return -vec, hpar.conjugate()
        else:
            return vec, hpar

    def __hamchecker(self, vec, lat, lidx):
        '''
        Private method finding start and target indices of hoppings that connect to sites.
        This method relies on a provided vector to find matching connections.
        It also provides the indices to set the indexmap.
        '''
        target = lat + vec

        # Check if any of the target points is already in self.lattice and consequently \"connecting\"
        check = np.all(np.isclose(target[:, np.newaxis], self.tlattice), axis=2)

        # First array in idx describes starting point index second array target point index
        self.idx = np.where(check)
        sidx = lidx[self.idx[0]]                    # Starting point indices
        tidx = self.idx[1]

        # Write into an indexmatrix True values for the points that connect
        self.__idxmapsetter(sidx, tidx)

        return sidx, tidx

    def __hamsetter(self, hpar, sidx, tidx):
        '''
        Private method setting the hopping parameters into the right place in the hamiltonian.
        Uses the information about indices from the hamchecker method.
        '''
        # Call setter for the hopping parameter
        self.__hop = hpar

        # Here we need to copy since otherwise we would modify self.htemplate
        hbuff = np.copy(self.htemplate)
        hbuff[(tidx, sidx)] = 1
        hbuff = np.kron(hbuff, self.__hop)

        # Do not add hermitian conjugate if we deal with onsite hopping
        if np.all(sidx == tidx):
            self.__hamiltonian += hbuff
        else:
            dim = self.latmatdim
            hbuff[:dim, :dim] = hbuff[:dim, :dim] + hbuff[:dim, :dim].T.conjugate()
            self.__hamiltonian += hbuff

    def __idxmapsetter(self, sidx, tidx):
        '''
        Sets an index matrix with true for connected sites.
        the tidx is the row whereas the sidx is the column.
        The first matrix self.__idxmap[:latdim, :latdim] only has entries on the
        lower triangle.
        This matrix (as the hamiltonian) holds the infromation about hopping to the
        NEXT unit slice (in direction of the tvector).
        '''
        sidx = np.copy(sidx)
        tidx = np.copy(tidx)

        # Exchange upper diagonal entries to lower diagonal ones
        upper = sidx > tidx
        buff = sidx[upper]
        sidx[upper] = tidx[upper]
        tidx[upper] = buff

        # Write True into the relevant self.__idxmap entries.
        self.__idxmap[tidx, sidx] = True

    def remove_dangling(self):
        '''
        This function finds points that only have one (or zero) connecting site, also known as dangling bond.
        This can achieved by searching in the self.__idxmap for connections that only appear once or zero times.
        '''
        idxbuff = np.copy(self.__idxmap)
        latdim = np.size(idxbuff, axis=1)
        idxbuff[:latdim, :latdim] += idxbuff[:latdim, :latdim].T
        for i in range(self.reps-1):
            # This for loop adds the hoppings to the PREVIOUS unit slice to the idxbuff
            # The self.__idxmap only holds information about hopping to the NEXT unit slice
            idxbuff = np.vstack((idxbuff, idxbuff[(i+1)*latdim:(i+2)*latdim, :latdim].T))

        sidx = np.where(idxbuff)[1]
        count = np.bincount(sidx)

        # These are the points with only one or zero connections
        # in the starting lattice 'self.lattice'.
        # These are the ones we want to remove.
        rmvidx = np.where(np.logical_or(count == 0, count == 1))[0]
        hamrmvidx = self.orb * rmvidx
        mvidx = np.arange(self.orb).reshape(self.orb, 1)
        hamrmvidx = mvidx + hamrmvidx
        hamrmvidx = hamrmvidx.flatten()

        # Remove points from lattice, hamiltonian columns, indexmap columns
        # htemplate columns, redefine latmatdim
        self.lattice = np.delete(self.lattice, rmvidx, axis=0)
        hambuff = np.delete(self.__hamiltonian, hamrmvidx, axis=1)
        htempbuff = np.delete(self.htemplate, rmvidx, axis=1)
        self.__idxmap = np.delete(self.__idxmap, rmvidx, axis=1)
        self.latmatdim = self.orb * np.size(self.lattice, axis=0)

        # Use periodicity and calculate the points to remove in the target lattice
        # from the starting lattice points.
        mvidx = np.arange(0, self.reps).reshape(self.reps, 1) * latdim
        rmvidx = mvidx + rmvidx
        rmvidx = rmvidx.flatten()
        hamrmvidx = self.orb*mvidx + hamrmvidx
        hamrmvidx = hamrmvidx.flatten()

        # Remove points from target lattice and hamiltonian rows and indexmap rows
        self.tlattice = np.delete(self.tlattice, rmvidx, axis=0)
        self.__hamiltonian = np.delete(hambuff, hamrmvidx, axis=0)
        self.htemplate = np.delete(htempbuff, rmvidx, axis=0)
        self.__idxmap = np.delete(self.__idxmap, rmvidx, axis=0)

    @property
    def connectmap(self):
        '''
        Getter for the connectmap defining hoppings between points.
        The connectmap is calculated from the indexmap 'self.__idxmap'.
        '''
        tidx, sidx = np.where(self.__idxmap)
        spoints = self.lattice[sidx]
        tpoints = self.tlattice[tidx]

        return np.hstack((spoints, tpoints - spoints))

    @property
    def idx(self):
        '''
        Getter for the index tuple
        '''
        return self._idx

    @idx.setter
    def idx(self, index):
        '''
        Setter for the index tuple, raises error if index tuple is empty.
        '''
        if index[0].size != 0:
            self._idx = index
        else:
            raise CoordinateError('The specified hopping does not connect to any neighbouring sites.')

    @property
    def lattice(self):
        '''
        Getter for the lattice variable
        '''
        return self._lattice

    @lattice.setter
    def lattice(self, lat):
        '''
        Setter for the lattice class. Checks wheter the given variable is an instance of the Lattice class.
        Also checks if the translational lattice (trans_lattice) is initialized.
        '''
        if isinstance(lat, np.ndarray) and lat.ndim == 2 and lat.shape[1] == 3:
            self._lattice = lat
        else:
            raise ValueError(
                            'The provided lattice argument is no instance of the \"numpy.ndarray\"-class \
                            or does not have the right shape (*,3).')

    @property
    def __hop(self):
        return self.__hopparam

    @__hop.setter
    def __hop(self, hpar):
        '''
        Temporarily set the hopping parameter in this variable and check for consistency.
        '''
        isnum = isinstance(hpar, (int, float, complex)) and self.orb == 1
        isarr = isinstance(hpar, np.ndarray) and hpar.ndim == 2 and hpar.shape[0] == hpar.shape[1] \
            and hpar.shape[0] == self.orb

        if isnum or isarr:
            self.__hopparam = hpar
        else:
            raise ValueError('The given hopping is not a number or numpy.ndarray with the dimension of orbitals.')

    @property
    def __hamiltonian(self):
        '''
        Getter for the hamiltonian matrix.
        '''
        return self.__ham

    @__hamiltonian.setter
    def __hamiltonian(self, hamil):
        self.__ham = hamil
        '''
        Setter for the \"hamiltonian\" variable to input Hamiltonians.
        Checks for hermiticity and \"numpy.ndarray\" with the right dimensionality.
        '''
        check_hamil = hamil[:self.latmatdim, :self.latmatdim]
        if isinstance(hamil, np.ndarray) and hamil.ndim == 2 and check_hamil.shape[0] == check_hamil.shape[1]:
            if np.all(np.isclose(check_hamil, check_hamil.T.conj())):
                self.__ham = hamil
            else:
                raise TypeError('The \"Hamiltonian\" is the wrong type since it is not hermitian.')
        else:
            raise ValueError('The \"Hamiltonian\" is either not an \"numpy.ndarray\" or not quadratic or no matrix.')

    @property
    def hamiltonian(self):
        '''
        Returns the private hamiltonian in a more comprehensible form:
        H = np.array(H_onsite, H_nn, H_nnn ...)
        '''
        H = self.__hamiltonian
        H = H.reshape(self.reps, self.latmatdim, self.latmatdim)
        unset_rows = np.count_nonzero(H, axis=2)
        unset_matrices = np.count_nonzero(unset_rows, axis=1)
        return H[np.where(unset_matrices)]

    def visualise(self):
        fig = plt.figure()
        lat = self.lattice
        latsize = np.size(lat, axis=0)
        tlat = self.tlattice[latsize:]

        ax1 = fig.add_subplot(211, projection='3d')
        ax1.scatter(lat[:, 0], lat[:, 1], lat[:, 2], c='blue', edgecolors='none', depthshade=False)
        ax1.scatter(tlat[:, 0], tlat[:, 1], tlat[:, 2], c='red', edgecolors='none', depthshade=False)

        convec = self.connectmap
        ax1.quiver(convec[:, 0], convec[:, 1], convec[:, 2], convec[:, 3],
                   convec[:, 4], convec[:, 5], colors='green', arrow_length_ratio=0)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')

        plt.show()
