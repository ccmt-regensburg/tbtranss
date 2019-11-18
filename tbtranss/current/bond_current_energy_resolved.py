import os
import errno
import sys
import numpy as np
from ..green import GreenSystem
from ..green import SelfEnergy
from ..multicore import MpiHelpers as Multi
from mpi4py import MPI


def bond_current_energy_resolved(ScaHamB, WirHam, mu1, mu2, mudiv, T, divisor, p, **kwargs):

    mult = Multi()
    comm = mult.comm
    rank = mult.rank

    mu_min = np.min((mu1, mu2))
    mu_max = np.max((mu1, mu2))

    if(rank == 0):
        # Analytical calculation deriving the integration boundaries
        # from the cutoff percentage p of p = f(E-mu_max) - f(E-mu_min)
        E_min, E_max = boundaries(mu_min, mu_max, T, p)
        totaldiv = (mudiv - 1) * divisor + 1
    else:
        E_min = None
        E_max = None
        totaldiv = None

    E_min = comm.bcast(E_min, root=0)
    E_max = comm.bcast(E_max, root=0)

    # Only returns the listpartition for the root process
    elist, elocal, ptuple, displace = mult.listpart([E_min, E_max, totaldiv], "rand")
    comm.Scatterv([elist, ptuple, displace, MPI.DOUBLE], elocal)

    LeadL = SelfEnergy(WirHam, 0)
    LeadR = SelfEnergy(WirHam, 1)
    GreenS = GreenSystem(ScaHamB)

    voltages = np.linspace(E_min, E_max, mudiv)

    if(rank == 0):
        # Create empty folders for every voltage
        for v in voltages:
            vname = '{:.7f}'.format(v)
            try:
                os.makedirs(kwargs['path'] + vname)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    elocallen = np.size(elocal)
    count = 0

    # Calculate bond current 'locally' on every core available
    for E in elocal:
        selfLe = LeadL.eigendecomposition(E)
        selfRe = LeadR.eigendecomposition(E)
        GreenS.init_retarded(selfLe, selfRe, E)
        voltindices = np.nonzero(-1e-10 <= voltages - E)[0]
        for i in voltindices:
            v = voltages[i]
            GreenS.init_keldysh(mu_min, v, T)
            vname = '{:.7f}'.format(v)
            outfile = str(kwargs["prefix"]) + '_T' + str(T) + '_' + \
                "{:.7f}".format(mu_min) + '_' + vname + '_E_' + '{:.7f}'.format(E) + '_bcurrent'
            np.savez(kwargs["path"] + vname + '/' + outfile, mu=[mu_min, v], E=E, bcurrent=GreenS.bond_current())
        count += 1
        sys.stdout.write('Rank: ' + str(rank) + ' is ' + '{:.2f}'.format((count/elocallen)*100) + '% done' + '\n')


def boundaries(mu_min, mu_max, T, p):

    if(T == 0):
        E_min = mu_min
        E_max = mu_max
    else:
        beta = 1/T
        r = np.exp(beta*(mu_max - mu_min))
        diskrim = np.sqrt((r*p+p-r+1)**2 - 4*p*p*r)
        x_max = (-(r*p+p-r+1)+diskrim)/(2*p*r)
        E_max = np.log(x_max)/beta+mu_max
        E_min = mu_min - (E_max - mu_max)
    return E_min, E_max
