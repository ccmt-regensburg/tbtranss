import numpy as np
from ..green import GreenSystem
from ..green import SelfEnergy
from ..multicore import MpiHelpers as Multi
from mpi4py import MPI


def bond_current_integration(ScaHamB, WirHam, mu1, mu2, mudiv, T, divisor, p, **kwargs):

    mult = Multi()
    comm = mult.comm
    size = mult.size
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

    dim = np.prod(ScaHamB.latsize)
    bcurrentmat = np.zeros((mudiv, dim, dim))
    voltages = np.linspace(E_min, E_max, mudiv)
    elocallen = np.size(elocal)
    count = 0
    for E in elocal:
        selfLd = LeadL.eigendecomposition(E)
        selfRd = LeadR.eigendecomposition(E)
        GreenS.init_retarded(selfLd, selfRd, E)
        voltindices = np.nonzero(-1e-10 <= voltages - E)[0]
        for i in voltindices:
            GreenS.init_keldysh(mu_min, voltages[i], T)
            bcurrentmat[i] += GreenS.bond_current()
        count += 1
        print("Rank: ", rank, " is ", "{:.2f}".format((count/elocallen)*100), "% done")

    comm.Barrier()
    if(rank == 0):
        bsize = np.size(bcurrentmat)
        bcurrentdisplace = np.arange(size) * bsize
        bcurrenttuple = (bsize,) * size
        bcurrentbuffer = np.empty(bsize * size, dtype=float)
    else:
        bsize = None
        bcurrentdisplace = None
        bcurrenttuple = None
        bcurrentbuffer = None

    comm.Gatherv(np.array(bcurrentmat).flatten(), [bcurrentbuffer, bcurrenttuple, bcurrentdisplace, MPI.DOUBLE], root=0)

    if(mult.rank == 0):
        buffermat = bcurrentbuffer.reshape((size, mudiv, dim, dim))
        bcurrent = np.zeros(np.shape(buffermat[0]))
        for i in range(size):
            bcurrent += buffermat[i]
        currentdiv = 0
        print("Start writing")
        E_delta = (E_max - E_min)/(totaldiv-1)
        for i, v in enumerate(voltages):
            if(i == 0):
                bcurrent[i] = bcurrent[i]*0
            else:
                bcurrent[i] = bcurrent[i]*E_delta
            currentdiv = currentdiv + divisor
            outfile = str(kwargs["prefix"]) + '_T' + str(T) + '_' + \
                "{:.2f}".format(mu_min) + '_' + "{:.3f}".format(v) + '_bcurrent'
            np.savez(kwargs["path"] + outfile, mu=[mu1, v], T=T, part=currentdiv, bcurrent=bcurrent[i])
        print("Done")


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
