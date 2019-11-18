import argparse
import numpy as np
import scipy.integrate as spi


def parsefile():
    parser = argparse.ArgumentParser(description="Bond current integrator using the trapezodial integration rule."
                                     "Is used in conjunction with the bond_current_energy_resolved script.")

    parser.add_argument("-b", "--bcurrent", type=str, nargs='+', help="File path of the required bond current files"
                        "produced by GreenSystem.")
    parser.add_argument("-g", "--geometry", type=str, nargs=1, help="File path of the required geometry file produced"
                        "by ScattererBuilder.")
    parser.add_argument("-t", "--title", type=str, nargs=1, help="Path and title of the integrated currents.")
    parser.add_argument("-m", "--mu", type=float, nargs=3, help="Start voltage, end voltage, number of voltages")

    args = parser.parse_args()
    return vars(args)


def integrate(param):
    # Load the first file to get the shape of the bcurrent matrix
    file1 = np.load(param['bcurrent'][0])
    bcurrentsum = np.zeros(np.shape(file1['bcurrent']))

    # Grab indices where there is a connection
    geofile = np.load(param['geometry'][0])
    idxmap = np.where(geofile['idxmap'])

    # Load every bcurrent matrix into memory
    # Only store non-zero entries
    etotal = []
    currtotal = []
    for files in param['bcurrent']:
        currentfile = np.load(files)
        etotal.append(currentfile['E'])
        currtotal.append(currentfile['bcurrent'][idxmap])

    etotal = np.array(etotal)
    currtotal = np.array(currtotal)

    # Find the integration width de
    etemp = np.sort(etotal)
    de = etemp[1] - etemp[0]

    # Integrate according to voltages
    mumin = param['mu'][0]
    mumax = param['mu'][1]
    mudiv = param['mu'][2]
    voltages = np.linspace(mumin, mumax, mudiv)
    emin = np.min(etotal)
    for v in voltages:
        if (emin >= v):
            bcurrentsum.fill(0)
        else:
            eidx = etotal <= v
            curr = currtotal[eidx]

            # Integrate every bond current along the energy axis
            for n, (i, j) in enumerate(zip(idxmap[0], idxmap[1])):
                bcurrentsum[i, j] = spi.trapz(curr[:, n], dx=de)

        title = param['title'][0]

        # Save the integrated bond current
        np.savez(title + '_' + '{:.3f}'.format(mumin) + '_' + '{:.3f}'.format(v), bcurrent=bcurrentsum)


if __name__ == '__main__':
    param = parsefile()
    integrate(param)
