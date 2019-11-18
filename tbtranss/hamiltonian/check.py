import numpy as np

from ..error.error import CoordinateError


class SystemCheck:
    '''
    This class takes an instance of the hamiltonian.ScattererBuilder and hamiltonian.LeadBuilder class
    and finalises the system. It checks if the leads fit the scattering region geometrically.
    It also checks if the orbitals per site are equal and the Hamiltonian blocks of a lead slice and
    scatterer slice fit together.
    '''

    def __init__(self, ScaB, *args):
        '''
        Stores the relevant information of the ScaB and LeadB instances for further processing
        in the green.System class.
        Also calls the relevant functions to check for consistency between lead and scatterer
        geometry and Hamiltonian.

        # Function parameters:
        ScaB                Instance of the hamiltonian.ScattererBuilder class.
        *args               Tuple of hamiltonian.LeadBuilder instances (max. 2)
                            If two leads are given the first one is attached in direction
                            of the translational lead vector.
        '''
        # Before saving relevant data we first check the lattices and Hamiltonian
        # for consistency
        # Check lattices
        LeadB = args
        if len(args) >= 2:
            # If we have two individual leads
            self.__check_geometry(LeadB[0].lattice, LeadB[1].lattice)

        ScaBslice = ScaB.lattice[:ScaB.sdim]
        self.__check_geometry(ScaBslice, LeadB[0].lattice)
        ScaBslice = ScaB.lattice[-ScaB.sdim:]
        self.__check_geometry(ScaBslice, LeadB[0].lattice)

        # Check Hamiltonian
        # When the lattice checker was succesful we only need to check for orbitals per site
        if len(args) >= 2:
            self.__check_orbitals(LeadB[0].orb, LeadB[1].orb)
        else:
            LeadB = LeadB + LeadB

        self.__check_orbitals(ScaB.orb, LeadB[0].orb)

        for Lead in LeadB:
            self.__check_hoprange(Lead.hamiltonian)

        print("Leads and system fit.")
#        # Now save the important data for transport calculations
#        self.ScaHam = ScaB.hamiltonian
#        self.LeadHam = []
#        for Lead in LeadB:
#            self.LeadHam.append(Lead.hamiltonian)

    def __check_geometry(self, lat1, lat2):
        '''
        Checks wheter two given lattice arrays fit together.
        The try-except block checks for equal number of sites.
        The if checks if all points are connected by the same vector (translational vector)
        '''
        try:
            diff = lat2 - lat1
        except ValueError:
            raise ValueError('Atleast one of the given leads does not have the same amount of sites'
                             ' as the connecting slice in the scatterer.')

        check = np.all(np.isclose(diff[0], diff))
        if not check:
            raise CoordinateError('The given lead does not fit together with the connecting slice of the'
                                  ' scatterer.')

    def __check_orbitals(self, orb1, orb2):
        '''
        Checks wheter the orbitals per site of the given leads is equal to the orbitals per site of the
        scatterer.
        '''
        if orb1 != orb2:
            raise ValueError('Either the scattering region or the leads do not have the same number'
                             ' of orbitals per site.')

    def __check_hoprange(self, hlist):
        '''
        Checks if the leads only use nearest slice hopping. This is necessary for the self-energy algorithms.
        '''
        hoprange = np.size(hlist, axis=0) - 1
        if hoprange > 1:
            raise AttributeError('The range of the unit slice hopping is not allowed to exceed nearest unit'
                                 ' slice hopping for transport calculations.')
