---
title: Comments on self-energy methods (self_energy.py)
---

# Eigendecomposition

## Crash due to non-quadratic matrix
Two modes that are non-degenerate are considered degenerate due to numerical inaccuracies.
This means that the comparison fails due to a epsilon bound that is too big.
Diagonalising the subspace gives wrong results and removes solutions.
This will lead to a non-quadratic matrix that will crash at the inversion step.

### Solution:
Play around with the value for epsilon in the comparison of the real and imaginary parts inside
the private method `self.__degeneracy_checker`{.Python}.

## Crash due to singular matrix
At some special energies the matrix of evanescent and propagating solutions is singular.
The code will crash with the error code of a singular matrix that is non-invertible.

### Solution:
Shift the energy a little bit away from the numerically unstable region.
