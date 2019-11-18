import numpy as np
import matplotlib.pyplot as plt                 # NOQA
from mpl_toolkits.mplot3d import Axes3D         # NOQA

import tbtranss.lattice as lat
import tbtranss.hamiltonian as ham
import tbtranss.green as gre

plt.rcParams["text.usetex"] = True


def main():

    # Translational vectors in a 3d cubic lattice
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])

    tvecs = np.array([ex, ey, ez])

    L = 8

    def rect(pos):
        return 0 <= pos[0] <= L-1 and 0 <= pos[1] <= L-1

    # Lead lattice construction calls
    lcomb = np.array([0, 0, 1])
    WirB = lat.LeadBuilder(rect, tvecs)
    ltlat = WirB.trans_constructor(lcomb)

    # Lead Hamiltonian construction calls
    t = 1
    s1 = np.array([[0, 1], [1, 0]])
    s2 = np.array([[0, -1j], [1j, 0]])
    s3 = np.array([[1, 0], [0, -1]])

    on = 2*t*s3
    hopx = 1j*t/2*s1 - t/2*s3
    hopy = 1j*t/2*s2 - t/2*s3
    hopz = t/2*s3

    WirHamB = ham.LeadBuilder(ltlat, WirB.tvector, orb=2)
    WirHamB.hop_all([0, 0, 0], on)
    WirHamB.hop_all(ex, hopx)
    WirHamB.hop_all(ey, hopy)
    WirHamB.hop_all(ez, hopz)
    WirHamB.visualise()

    # Here we remove dangling bonds to avoid flat bands
    WirHamB.connectmap
    ltlat = WirHamB.lattice
    WirHam = WirHamB.hamiltonian

    # Scatterer lattice construction calls
    ScaB = lat.ScattererCarver(ltlat, WirB.tvector, reps=3)
    stlat = ScaB.lattice

    # Scatterer Hamiltonian construction calls
    ScaHamB = ham.ScattererBuilder(stlat, orb=2)
    ScaHamB.hop_all([0, 0, 0], on)
    ScaHamB.hop_all(ex, hopx)
    ScaHamB.hop_all(ey, hopy)
    ScaHamB.hop_all(ez, hopz)
    ScaHamB.visualise()

    # Lead band structure calculation
    Solver = ham.Spectrum(WirHam, WirB.stretchfactor, [-np.pi, np.pi, 100])
    klist, spectrum, evlist = Solver.ham_eigv_eigvec()

    # Check the system for consistency with the leads
    ham.SystemCheck(ScaHamB, WirHamB)

    # Use the system in transport calculations
    Te = []
    elist = np.linspace(0, 1.0, 100)
    LeadL = gre.SelfEnergy(WirHam, 0)
    GreenS = gre.GreenSystem(ScaHamB)
    for e in elist:
        selfLe = LeadL.eigendecomposition(e)
        GreenS.init_retarded(selfLe, selfLe, e)
        Te.append(GreenS.transmission())

    # Plot band structure with transmission results
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(klist, spectrum)
    ax1.set_xlabel(r"k $[\frac{1}{a}]$")
    ax1.set_ylabel(r"E $[t]$")
    line1, = ax2.plot(Te, elist)
    ax2.set_xlabel(r"T")

    plt.show()

    # Integrated bond currents.
    ScaHamB.export_geometry("weyl_8x8x3_geo")

    mumin = 0
    mumax = 0.201
    N = 150
    elist = np.linspace(mumin, mumax, N)
    bcurrent = np.zeros((192, 192))

    LeadL = gre.SelfEnergy(WirHam, 0)
    GreenS = gre.GreenSystem(ScaHamB)
    for e in elist:
        selfLe = LeadL.eigendecomposition(e)
        GreenS.init_retarded(selfLe, selfLe, e)
        GreenS.init_keldysh(mumin, mumax, 0.0)
        bcurrent += GreenS.bond_current()

    dE = (mumax - mumin)/(N-1)
    bcurrent = bcurrent*dE

    np.savez("weyl_8x8x3_ibc", bcurrent=bcurrent)


if __name__ == "__main__":
    main()
