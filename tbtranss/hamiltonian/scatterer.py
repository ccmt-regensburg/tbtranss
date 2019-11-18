import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # NOQA

from ..error.error import CoordinateError


class ScattererBuilder:
    '''
    This class constructs the Hamiltonian matrix from the geometrical information in the Lattice class.
    Returns a hermitian Hamiltonian array of dimension (orbitals*sites, orbitals*sites).
    '''

    def __init__(self, lat, **kwargs):
        '''
        Function constructing the blank hamiltonian and initalising all important internal variables.

        #PARAMETERS:
        lat                 Needs to be a numpy.ndarray of dimension (>=2, *, 3) (Lattice ordered by slices).
        kwargs['orb']       Integer specifying the orbitals per site.

        #ATTRIBUTES:
        self.sdim           Number of sites in the first and last slice of the provided lattice.
                            Defines which lattice points are not allowed to be touched by remove_dangling.
        self.lattice        Array saving all lattice points.
        self.index          Array closely tied to self.lattice holding the indices of lattice points.
                            For example self.index[0] is the index of self.lattice[0]

        self.orb            Integer defining the number of orbitals per site.
        self.hamiltonian    Hamiltonian of the system, is initialised with zeros in the constructor.
        self.htemplate      Hamiltonian template with dimensionality of the lattice.
                            Is used to add hoppings to the hamiltonian.

        self.idxmap       Boolean indexmap. Has the dimensionality of the lattice (latsize, latsize).
                            Has a true value for each connecting hopping in the hamiltonian.
                            Only lower tridiagonal is written to.
        '''
       # Build scatterer lattice from lead information

        self.sdim = lat                                             # Saves the sitenumber of the first and last slice
        self.lattice = np.concatenate(lat, axis=0)                  # Stores the whole lattice

        latsize = np.size(self.lattice, axis=0)                     # Number of lattice sites
        self.latsize = latsize
        self.index = np.arange(latsize)                             # Array holding the indices of vectors in lat

        # Variables to work with while constructing the Hamiltonian matrix
        self.orb = kwargs['orb']                                              # Parameter for the orbitals per site
        self.hamiltonian = np.zeros((self.orb*latsize, self.orb*latsize), dtype=complex)
        self.htemplate = np.zeros((latsize, latsize), dtype=complex)
        self.idxmap = self.htemplate.astype(bool, copy=True)              # Empty indexmap

    def hop_all(self, vec, hpar):
        '''
        Applies a hopping to all points that can be connected by a provided vector.

        #Function parameters:
        vec                 User provided vector for the hopping direction
        hpar                Hopping parameter

        '''
        # Call the hamsetter with the full lattice and corresponding indices
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
        stvecs = lat[:, np.newaxis] - lat
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
            Function to call recursively. Gives the unique vectors in an array of vectors.
            Drops the negative (hermitian conjugate) of those vectors.
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
            sidx, tidx = self.__hamchecker(vec, lat, lidx)
            self.__hamsetter(hpar, sidx, tidx)

    def __hamchecker(self, vec, lat, lidx):
        '''
        Private method finding start and target indices of hoppings that connect to sites.
        This method relies on a provided vector to find matching connections.
        It also provides the indices to set the indexmap.
        '''
        target = lat + vec

        # Check if any of the target points is already in self.lattice and consequently \"connecting\"
        check = np.all(np.isclose(target[:, np.newaxis], self.lattice), axis=2)

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
            self.hamiltonian += hbuff
        else:
            self.hamiltonian += hbuff + hbuff.T.conjugate()

    def __idxmapsetter(self, sidx, tidx):
        '''
        Sets an index matrix with true values for connected sites.
        the target idx is the row whereas the start idx is the column.
        The idxmap only has entries on the lower triangle as it must be symmetric.
        '''
        sidx = np.copy(sidx)
        tidx = np.copy(tidx)

        # Exchange upper diagonal entries to lower diagonal ones
        upper = sidx > tidx
        buff = sidx[upper]
        sidx[upper] = tidx[upper]
        tidx[upper] = buff

        # Write True into the relevant self.idxmap entries.
        self.idxmap[tidx, sidx] = True

    def remove_dangling(self):
        '''
        This function finds points that only have one (or zero) connecting site, also known as dangling bond.
        This can achieved by searching in the self.idxmap for connections that only appear once or zero times.
        The method explicitely excludes points found in the first and last slice to assure compatibiliy with the
        leads.
        '''
        idxbuff = np.copy(self.idxmap)
        idxbuff += idxbuff.T

        sidx = np.where(idxbuff)[1]

        # Remove site indices from the first and last slice
        latsize = np.size(self.lattice, axis=0)

        count = np.bincount(sidx)

        # These are the points with only one or zero connections
        # in the starting lattice 'self.lattice'.
        # These are the ones we want to remove.
        rmvidx = np.where(np.logical_or(count == 0, count == 1))[0]
        rmvidx = rmvidx[np.logical_and(self.sdim <= rmvidx, rmvidx < latsize - self.sdim)]
        hamrmvidx = self.orb * rmvidx
        mvidx = np.arange(self.orb).reshape(self.orb, 1)
        hamrmvidx = mvidx + hamrmvidx
        hamrmvidx = hamrmvidx.flatten()

        # Remove points from lattice, hamiltonian columns, indexmap columns
        # htemplate columns, redefine latmatdim
        self.lattice = np.delete(self.lattice, rmvidx, axis=0)
        hambuff = np.delete(self.hamiltonian, hamrmvidx, axis=1)
        htempbuff = np.delete(self.htemplate, rmvidx, axis=1)
        self.idxmap = np.delete(self.idxmap, rmvidx, axis=1)
        self.latmatdim = self.orb * np.size(self.lattice, axis=0)

        # Remove points from target lattice and hamiltonian rows and indexmap rows
        self.hamiltonian = np.delete(hambuff, rmvidx, axis=0)
        self.htemplate = np.delete(htempbuff, rmvidx, axis=0)
        self.idxmap = np.delete(self.idxmap, rmvidx, axis=0)

    @property
    def connectmap(self):
        '''
        Getter for the connectmap defining hoppings between points.
        The connectmap is calculated from the indexmap 'self.idxmap'.
        '''
        tidx, sidx = np.where(self.idxmap)
        spoints = self.lattice[sidx]
        tpoints = self.lattice[tidx]

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
    def sdim(self):
        return self._sdim

    @sdim.setter
    def sdim(self, lat):
        '''
        Checks if the given lattice has enough slices. And that the size of the frst
        and last slice are equal.
        '''

        # Check if we have enough slices.
        checkslice = (2 <= np.size(lat, axis=0))
        if not checkslice:
            raise ValueError(
                        'The provided lattice argument does not have enough slices (#slices >= 2)')

        # Check if the last and first slice entries have the same amount of sites.
        checknum = np.size(lat[0]) == np.size(lat[-1])
        if not checknum:
            raise ValueError(
                        'The provided lattice does not have the same number of sites in the first and'
                        'last slice')

        self._sdim = np.size(lat[0], axis=0)

    @property
    def lattice(self):
        '''
        Getter for the lattice variable
        '''
        return self._lattice

    @lattice.setter
    def lattice(self, lat):
        '''
        Setter for the lattice array. This is the concatenated version of the lattice array that holds the information
        about slice indices, provided by the user.
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
    def hamiltonian(self):
        '''
        Getter for the hamiltonian matrix.
        '''
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, hamil):
        '''
        Setter for the \"hamiltonian\" variable to input Hamiltonians from outside.
        Checks for hermiticity and \"numpy.ndarray\" with the right dimensionality.
        '''
        if isinstance(hamil, np.ndarray) and hamil.ndim == 2 and hamil.shape[0] == hamil.shape[1]:
            if np.all(np.isclose(hamil, hamil.T.conj())):
                self._hamiltonian = hamil
            else:
                raise TypeError('The \"Hamiltonian\" is the wrong type since it is not hermitian.')
        else:
            raise ValueError('The \"Hamiltonian\" is either not an \"numpy.ndarray\" or not quadratic or no matrix.')

    def visualise(self):
        '''
        Visualises the lattice and all connections already present in the lattice.
        '''
        fig = plt.figure()
        lat = self.lattice

        ax1 = fig.add_subplot(211, projection='3d')
        ax1.scatter(lat[:, 0], lat[:, 1], lat[:, 2], c='blue', edgecolors='none', depthshade=False)

        convec = self.connectmap
        ax1.quiver(convec[:, 0], convec[:, 1], convec[:, 2], convec[:, 3], convec[:, 4], convec[:, 5], colors='green', arrow_length_ratio=0)    # NOQA
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')

        plt.show()

    def export_geometry(self, filestr):
        np.savez(filestr, idxmap=self.idxmap, conmap=self.connectmap)
