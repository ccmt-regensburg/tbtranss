import numpy as np

import tbtranss.green as gre


def main():
    N = 4

    M = 1*np.eye(N, k=1) + 1*np.eye(N, k=-1)
    T = np.eye(N)

    Lead = gre.SelfEnergy([M, T, T], 0)
    Lead.eigendecomposition(1)
    Lead.eigendecomposition_fast(1)


if __name__ == "__main__":
    main()
