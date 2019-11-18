import numpy as np
import matplotlib.pyplot as plt                 # NOQA
from mpl_toolkits.mplot3d import Axes3D         # NOQA

import tbtranss.lattice as lat
import tbtranss.hamiltonian as ham
import tbtranss.green as gre

plt.rcParams["text.usetex"] = True


def main():
    # Definitions for 2D Graphene
    base = np.array([[1/np.sqrt(3), 0, 0]])
    tvecs = np.array([[np.sqrt(3)/2, 0.5, 0], [np.sqrt(3)/2, -0.5, 0], [0, 0, 0]])

    def rect(pos):
        return -3.5 <= pos[0] <= 3 and 0 <= pos[1] <= 3

    # Lead lattice construction calls
    lcomb = np.array([1, 1, 0])
    WirB = lat.LeadBuilder(rect, tvecs, base)
    ltlat = WirB.trans_constructor(lcomb)

    # Lead Hamiltonian construction calls
    t = 1

    WirHamB = ham.LeadBuilder(ltlat, WirB.tvector, orb=1)
    WirHamB.hop_nn(t, 1)

    # Here we remove dangling bonds to avoid flat bands
    WirHamB.remove_dangling()
    WirHamB.connectmap
    ltlat = WirHamB.lattice
    WirHam = WirHamB.hamiltonian

    # Scatterer lattice construction calls
    ScaB = lat.ScattererCarver(ltlat, WirB.tvector, reps=10)
    stlat = ScaB.lattice

    # Scatterer Hamiltonian construction calls
    ScaHamB = ham.ScattererBuilder(stlat, orb=1)
    ScaHamB.hop_nn(t, 1)
    ScaHamB.visualise()

    # Lead band structure calculation
    Solver = ham.Spectrum(WirHam, WirB.stretchfactor, [-np.pi, np.pi, 100])
    klist, spectrum, evlist = Solver.ham_eigv_eigvec()

    # Check the system for consistency with the leads
    ham.SystemCheck(ScaHamB, WirHamB)

    # Use the system in transport calculations
    Td = []
    elist = np.linspace(0, 0.9, 100)
    LeadL = gre.SelfEnergy(WirHam, 0)
    LeadR = gre.SelfEnergy(WirHam, 1)
    GreenS = gre.GreenSystem(ScaHamB)
    for e in elist:
        selfLd = LeadL.decimation(e, 3000, 0.01)
        selfRd = LeadR.decimation(e, 3000, 0.01)
        GreenS.init_retarded(selfLd, selfRd, e)
        Td.append(GreenS.transmission())

    # Plot band structure with transmission results
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(klist, spectrum)
    ax1.set_xlabel(r"k $[\frac{1}{a}]$")
    ax1.set_ylabel(r"E $[t]$")
    line1, = ax2.plot(Td, elist)
    ax2.set_xlabel(r"T")

    plt.show()

    e = 0.20

    selfLd = LeadL.decimation(e, 20000, 0.01)
    selfRd = LeadR.decimation(e, 20000, 0.01)

    mumin = 0.19
    mumax = 0.21
    GreenS.init_retarded(selfLd, selfRd, e)
    GreenS.init_keldysh(mumin, mumax, 0.0)
    GreenS.bond_current()

    ScaHamB.export_geometry("graphene_w14_geo")
    GreenS.export_bcurrent("graphene_w14_t1")


if __name__ == "__main__":
    main()
