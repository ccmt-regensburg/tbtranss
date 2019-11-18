import numpy as np

import matplotlib.pyplot as plt                 # NOQA
from mpl_toolkits.mplot3d import Axes3D         # NOQA

from ..error.error import CoordinateError       # NOQA


class ScattererCarver:
    '''
    This class thakes a lead lattice with translational symmetry and creates a
    carvable scattering region out of it.

    self.reps       Number of slices in the system (or repetitions of the unit slice)
    '''

    def __init__(self, lat, tvector, **kwargs):
        # Build scatterer lattice from lead information
        if 'reps' in kwargs:
            self.reps = int(kwargs['reps'])
        else:
            self.reps = 0

        tvectors = np.arange(self.reps)[:, np.newaxis] * tvector

        # This implicitely sets a slice index
        lat = lat + tvectors[:, np.newaxis]
        self.lattice = lat

        self.slicedim = np.size(self.lattice, axis=0)                     # Number of sites in a slice
        self.index = np.arange(self.slicedim)                             # Array holding the indices in a slice

    @property
    def reps(self):
        return self._reps

    @reps.setter
    def reps(self, r):
        '''
        Sets the number of unit slice repetitions and is therefore equal to the number of slices
        in the system.
        '''

        if r > 1:
            self._reps = int(r)
        elif 0 < r <= 1:
            raise ValueError('The parameter giving the number of slices needs to be >= 2, '
                    'since the last and first slice are coupled to the leads.')
        elif r == 0:
            # Make sure that the number of slices is at least 2.
            print('The number of slices was not set. Therefore, it will be set to 2.')
            self._reps = 2
        else:
            raise ValueError('The given parameter is no number or negative.')

    @property
    def lattice(self):
        '''
        Getter for the lattice variable
        '''
        return self._lat

    @lattice.setter
    def lattice(self, lat):
        '''
        Setter for the lattice array. Checks if a 3d lattice is given.
        This is the internal lattice used for calculations and checks. It is an array with three indices.
        First index is the slice index. next index (row) is the point in space, next index (column)
        is the coordinate (x, y or z).
        '''
        if lat.shape[2] == 3:
            self._lat = lat
        else:
            raise ValueError('The provided lattice does not have the right shape (*,3).')
