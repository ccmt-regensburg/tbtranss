import numpy as np
from fractions import gcd

# Plotting packages
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # NOQA

# Tbtrans packages
from ..error.error import CoordinateError


class LeadBuilder():
    r"""
    Construct a wire out of a regular lattice with translational symmetry.

    Parameters
    ----------
    bound : list or function
        Needs a `list` with two `double` entries.
        Function input needs to take a position list and outputs needs to be boolean.
        If list is given: creates rectangular boundary from -list.max/2 to + list.max/2 for x and y.
        If function is given: creates boundary with respect to given function.
    tvecs : np.ndarray
        Translational symmetry vectors of the lattice in the format:
        [[...], [...], [...]]
    basis: np.ndarray, optional
        If the lattice has a basis, with more than one atom set a basis vector for each additional atom
        in the same format as for tvecs.

    Returns
    -------
    LeadBuilder:
        Returns an empty LeadBuilder instance.
        Can be filled by calling the `trans_constructor` method.
    """

    def __init__(self, bound, tvecs, *basis):
        self.eps = 1e-10
        self.tvecs = tvecs
        self.basis = basis

        self.boundary = bound

        self.trans_lattice = None
        self.basis_lattice = None

    def trans_constructor(self, lcomb, **kwargs):
        r"""
        Constructs the wire with the linear combination of of translational lattice vectors given
        by the linear combination `lcomb`. The translational vector is than given in the direction
        `transvec` = `lcomb[0]` * `tvecs[0]` + `lcomb[1]` * `tvecs[1]` + `lcomb[1]` * `tvecs[2]`

        Parameters
        ----------
        lcomb: list
            Needs as many entries as translational lattice vectors. Only accepts integer entries.
            This defines the infinite direction of the quasi 1d wire.
        **kwargs: 'otvector'; np.ndarray and 'start'; np.ndarray, optional
            The orthogonal translational vector `otvector` is an optional vector that defines the
            coordinate system that is used to set the wire boundaries via a function or a rectangle.
            This can be used to rotate the crossection of you wire.
            The `start` point is used to define the starting point of the lattice construction.

        Returns
        -------
        complete_lattice: np.ndarray
            An array with as many rows as there are lattice points and 3 columns for x, y and z.

        """
        self.lcomb = lcomb
        self.mtvector = self.lcomb

        # Set the first orthogonal basis vector if not set: automatically calculate
        if 'ovector' in kwargs:
            self.otvector = kwargs['otvector']
        else:
            self.otvector = None

        # Set the starting point if specified if not set to (0, 0, 0)
        if 'start' in kwargs:
            self.start = kwargs['start']
        else:
            self.start = np.array([0, 0, 0])

        # All basis vectors of the orthogonal, local wire basis system spanning the space around mtvector.
        # This is also the transformation matrix to write the lattice vectors in terms of the new coordinate
        # system.
        S = np.hstack((self.otvector.T, self.mtvector[:, np.newaxis]))

        # Overwrite the old translational vectors with the ones in the new basis
        ntvecs = np.linalg.solve(S, self.tvecs.T).T
        if self.basis is not None:
            nbasis = np.linalg.solve(S, self.basis.T).T
        else:
            nbasis = None
        start = np.linalg.solve(S, self.start).T
        trans_lattice, basis_lattice = self.__fill(start, ntvecs, nbasis)

        self.trans_lattice = self.__repeat_slice(np.dot(S, trans_lattice.T).T)
        complete_lattice = self.trans_lattice
        if self.basis is not None:
            self.basis_lattice = self.__repeat_slice(np.dot(S, basis_lattice.T).T)
            complete_lattice = np.vstack((complete_lattice, self.basis_lattice))
        else:
            self.basis_lattice = None

        return complete_lattice

    def __fill(self, start, tvecs, basis):
        '''
        Gradually fills the boundary volume with pre-defined lattice + basis sites.
        '''

        tvecs = np.vstack((tvecs, -tvecs))
        check = [start]                                                     # Holding all checked & to check vectors
        trans_in = []                                                       # Container if trans-symmetry point is in
        basis_lattice = []                                                  # Container for basis lattice positions

        for pos in check:
            if self.boundary(pos):
                trans_in.append(True)
                for npos in (pos + tvecs):
                    if not np.isclose(npos, np.array(check)).all(axis=1).any():
                        check.append(npos)
            else:
                trans_in.append(False)

            if basis is not None:                                      # Only execute if basis is not empty
                basis_pos = pos + basis
                for bpos in basis_pos:
                    if self.boundary(bpos):
                        basis_lattice.append(bpos)

        # Only make self.basis_lattice equal to basis_lattice if we appended something to basis_lattice
        # Here we also remove points longer than the intended tvector.
        # To understand this step see the docstring of the boundwrapper function in the boundary setter.
        if self.basis is not None:
            basis_lattice = np.array(basis_lattice)
            basis_in = basis_lattice[:, 2] < 1 - self.eps
            basis_lattice = basis_lattice[basis_in]
        else:
            basis_lattice = None

        trans_in = np.array(trans_in, dtype=bool)
        trans_lattice = np.array(check)[trans_in]

        # Here also remove points longer or equal to intended mtvector
        trans_in = trans_lattice[:, 2] < 1 - self.eps
        trans_lattice = trans_lattice[trans_in]

        return trans_lattice, basis_lattice

    def __repeat_slice(self, lat):
        '''
        This function takes a lattice (minimal slice) as a parameter and increases it
        to the size specified by the user.
        '''
        latbuf = np.copy(lat)
        # This loops increases the slice size by the self.__lcombdiv factor.
        # This factor is derived in the lcomb setter
        for i in range(self.__lcombdiv - 1):
            latbuf += self.mtvector
            lat = np.vstack((lat, latbuf))

        return lat

# Translational lattice vectors and basis, setter and getter
    @property
    def basis(self):
        '''
        Getter for the basis set vectors.
        '''
        return self._basis

    @basis.setter
    def basis(self, args):
        '''
        Setter checks if basis argument is empty or set.
        Also increases ndarray.ndim from 1 to 2 if basis is only a vector.
        '''
        if args:
            if args[0].ndim == 1:
                self._basis = args[0][np.newaxis]                           # Make basis a matrix if its only a vector
            else:
                self._basis = args[0]
        else:
            self._basis = None

# Translational wire vector and corresponding linear combination of translational lattice vectors, setter and getter
    @property
    def lcomb(self):
        return self._lcomb

    @lcomb.setter
    def lcomb(self, lcomb):
        '''
        Sets the prefactors of the linear combination of translational vectors defining the final translational\
        vector of the wire.
        '''
        if np.size(lcomb) == np.size(self.tvecs, axis=0) and lcomb.dtype == np.int64:
            # Here we find the greatest common divisor and use it to divide the linear combination prefactors
            if lcomb.size == 3:
                div = gcd(lcomb[0], gcd(lcomb[1], lcomb[2]))
            if lcomb.size == 2:
                div = gcd(lcomb[0], lcomb[1])
            if lcomb.size == 1:
                div = lcomb[0]
            self.__lcombdiv = div                        # Divisor to find the minimal unique set of integers
            self._lcomb = lcomb // div
        else:
            raise ValueError('You did not give as much prefactors as there are translational lattice vectors or the prefactors are\
                    not integers.')

    @property
    def tvector(self):
        '''
        Return the user specified translational wire vector
        '''
        return self.mtvector * self.__lcombdiv

    @property
    def stretchfactor(self):
        '''
        Return the integer specifying how often the minimal tvector (self.mtvector)
        fits into the user specified tvector.
        '''
        return self.__lcombdiv

    @property
    def mtvector(self):
        '''
        Return the minimal translational wire vector.
        '''
        return self._mtvector

    @mtvector.setter
    def mtvector(self, lcomb):
        '''
        Sets the translational lattice vector of the lattice. It is constructed from the given linear combination given\
        as first argument of the trans_constructor.
        '''
        # Multiply every row by the corresponding prefactor for the linear combination
        # And add the vectors for the linear combination
        self._mtvector = (self.lcomb * self.tvecs.T).T
        self._mtvector = np.sum(self.mtvector, axis=0)

# Orthogonal basis vector to translational wire vector, setter and getter
    @property
    def otvector(self):
        return np.array([self._otvec1, self._otvec2])

    @otvector.setter
    def otvector(self, args):
        '''
        This sets the orthogonal vectors to self.mtvector in order to have a orthonormal basis system.
        '''
        if args is None:
            # Here we set the perpendicular vector to the translational vector in order to construct a local vector
            # space around the wire. This is called because no user vector was given.
            buff = np.cross(self.mtvector, self.tvecs[0])

            # If mtvector already points into the direction of the first translational lattice vector do
            # the cross product with the second one
            if np.all(np.isclose(buff, 0)):
                buff = np.cross(self.mtvector, self.tvecs[1])

            self._otvec2 = buff/np.linalg.norm(buff)                    # Orthogonal to mtvector of wire and normalised
            buff = np.cross(self._otvec2, self.mtvector)
            self._otvec1 = buff/np.linalg.norm(buff)

        else:
            if args[0].size == 3 and np.isclose(np.dot(args[0], self.mtvector), 0):
                self._otvec1 = args[0]
                buff = np.cross(self._otvec1, self.mtvector)
                self._otvec2 = buff/np.linalg.norm(buff)
            else:
                raise ValueError('Your perpendicular vector does not have the right dimension or is not perpendicular to the\
                    translational vector of the wire.')

# Boundary of the system, either specified by a function or two numbers
    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, boundfunc):
        if not callable(boundfunc):
            # If boundfunc consists only of two numbers create a function describing a square
            W = boundfunc[0]
            L = boundfunc[1]

            def square(pos):
                return -W/2 <= pos[0] <= +W/2 and -L/2 <= pos[1] <= L/2
            boundfunc = square

        # Overwrite the old boundary function by a decorator adding the mtvector direction

        def boundwrapper(pos):
            '''
            Decorator for the boundfunc given by the user. Extends the defined boundary to the dimension
            along the transversal symmetry of the wire.
            In order to make the fill_constructor work we have to give a region that is larger than one unit
            cell. This is why we add 10e-10 to the upper bond comparison with pos[2] and subtract 10e-10 from
            the lower bond comparison of pos[2].
            '''

            return boundfunc(pos[:2]) and 0 - self.eps <= pos[2] <= 1 + self.eps

        self._boundary = boundwrapper

# Starting point used in the _fill routine
    @property
    def start(self):
        '''
        Getter for the starting point.
        '''
        return self._start

    @start.setter
    def start(self, pos):
        '''
        Setter checks if the starting point is inside the boundary, else an error is raised.
        '''
        if not self.boundary(pos):
            raise CoordinateError('The specified vector is not inside the boundary', pos)
        self._start = pos

    def visualise(self):
        fig = plt.figure()
        lat = self.trans_lattice
        lat2 = self.trans_lattice + self.tvector
        lat2 = np.vstack((lat2, self.trans_lattice + 2*self.tvector))

        ax1 = fig.add_subplot(211, projection='3d')
        ax1.scatter(lat[:, 0], lat[:, 1], lat[:, 2], c='blue', edgecolors='none', depthshade=False)
        ax1.scatter(lat2[:, 0], lat2[:, 1], lat2[:, 2], c='red', edgecolors='none', depthshade=False)

        bas = self.basis_lattice
        if bas is not None:
            bas2 = self.basis_lattice + self.tvector
            bas2 = np.vstack((bas2, self.basis_lattice + 2*self.tvector))
            ax1.scatter(bas[:, 0], bas[:, 1], bas[:, 2], c='green', edgecolors='none', depthshade=False)
            ax1.scatter(bas2[:, 0], bas2[:, 1], bas2[:, 2], c='red', edgecolors='none', depthshade=False)

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')

        plt.show()
