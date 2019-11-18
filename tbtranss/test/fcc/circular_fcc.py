import numpy as np
import matplotlib.pyplot as plt

import tbtranss.lattice as lat
import tbtranss.hamiltonian as ham
import tbtranss.green as gre


def main():

    # Definitions for 3D fcc lattice
    base = np.array([[0.5, 0.5, 0.5]])
    tvecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    start = np.array([0, 0, 0])

    def disc(pos):
        x = pos[0]
        y = pos[1]

        r = x**2 + y**2
        return r < 20

    def square(pos):
        x = pos[0]
        y = pos[1]

        return 0 <= x <= 3 and 0 <= y <= 3

    # Lead construction calls

    lcomb = np.array([1, 0, 0])
    WirB = lat.LeadBuilder(disc, tvecs, base)
    tlat = WirB.trans_constructor(lcomb, start=start)
    WirB.visualise()

    # Scatterer Lattice construction calls

    ScaB = lat.ScattererCarver(tlat, WirB.tvector, reps=2)
    stlat = ScaB.lattice

    # Scatterer Hamiltonian construction calls

    t = 1
    ScaHamB = ham.ScattererBuilder(stlat, orb=1)
    ScaHamB.hop_nn(t, 1)
    ScaHamB.visualise()

    # Lead Hamiltonian construction calls

    WirHamB = ham.LeadBuilder(tlat, WirB.tvector, orb=1)
    WirHamB.hop_nn(t, 1)
    WirHamB.visualise()
    WirHam = WirHamB.hamiltonian

    # Lead band structure calculation

    Solver = ham.Spectrum(WirHam, WirB.stretchfactor, [-np.pi, np.pi, 100])
    klist, spectrum, evlist = Solver.ham_eigv_eigvec()

    # Check the system for consistency with the leads

    ham.SystemCheck(ScaHamB, WirHamB)

    # Use the system in transport calculations
    # with decimation technique

    Td = []

    elist = np.linspace(-2, 2, 100)
    LeadL = gre.SelfEnergy(WirHam, 0)
    LeadR = gre.SelfEnergy(WirHam, 1)
    GreenS = gre.GreenSystem(ScaHamB)
    i = 0
    for e in elist:
        print("Round ", i)
        i += 1
        selfLd = LeadL.decimation(e, 500, 0.01)
        selfRd = LeadR.decimation(e, 500, 0.01)
        GreenS.init_retarded(selfLd, selfRd, e)
        Td.append(GreenS.transmission())

    # Plot band structure with transmission results
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(klist, spectrum)
    ax1.set_xlabel(r"k $[\frac{1}{a}]$")
    ax1.set_ylabel(r"E $[t]$")
    ax2.plot(Td, elist)
    ax2.set_xlabel(r"T")

    plt.show()


if __name__ == "__main__":
    main()
