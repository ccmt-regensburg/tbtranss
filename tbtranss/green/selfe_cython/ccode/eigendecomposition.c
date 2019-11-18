#include "selfalgorithm.h"
#define EPS 10e-8		// Accuracy of absolute value of eigenvalues
#define CEPS 10e-10		// Accuracy of real and imaginary part of eigenvalues

static void
d_quad_mat_construct(const size_t N, double (*Q_l)[N], double (*Q_r)[N], const size_t n, double (*M)[n], double (*T)[n], double (*T_t)[n], double E);
static void
z_quad_mat_construct(const size_t N, double complex (*Q_l)[N], double complex (*Q_r)[N], const size_t n, double complex (*M)[n], double complex (*T)[n], double complex (*T_t)[n], double E);
static void
d_general_eig_solver(const size_t N, double (*Q_l)[N], double (*Q_r)[N], double complex *alpha, double complex *beta, double complex (*V_r)[N]);
static void
z_general_eig_solver(const size_t N, double complex (*Q_l)[N], double complex (*Q_r)[N], double complex *alpha, double complex *beta, double complex (*V_r)[N]);
static void
mode_filter(const size_t N, double complex (*V_r)[N], double complex *alpha, double complex *beta, double complex (*T)[N/2], double complex (*T_t)[N/2], double complex *ieig, double complex (*V_i)[N/2]);
static void
iregion_checker(const size_t N, double complex *alpha, double complex *beta, int *in, int *prop);
static size_t
degeneracy_checker(const size_t npidx, int *pidx, double complex *peig, int ***degen);
static void
velocity_checker(const size_t npidx, int *pidx, double complex *peig, const size_t N, double complex (*V_r)[N], const size_t nunique, int **dbool, double complex (*T)[N/2], double complex (*T_t)[N/2], int *ridx);

int
d_eigendecomposition(const size_t n, void *Mp, void *Tp, void *T_tp, double E, void *sigp)
{
//  Calculate self-energy of the given system via the eigendecomposition method. This method is used for real input
//  Hamiltonians
//
//  PARAMETER:
//  	n		- number of columns (quadratic matrices)
//  	(*M)[n]		- hermitian unit cell hamiltonian
//  	(*T)[n]		- hopping matrix between unit cells
//  	(*T_t) [n]	- daggered hopping matrix between unit cells
//  	E		- Energy of the system

	double (*M)[n] = Mp;
	double (*T)[n] = Tp;
	double (*T_t)[n] = T_tp;
	double complex (*V_i)[n] = sigp;

	const size_t N = 2*n;
	double (*Q_l)[N] = d_matcal(N, N);
	double (*Q_r)[N] = d_matcal(N, N);
	d_quad_mat_construct(N, Q_l, Q_r, n, M, T, T_t, E);

	// Solve the generalised eigenproblem defined via the Q_l and Q_r matrix
	double complex *alpha = malloc(sizeof(double complex[N]));
	double complex *beta = malloc(sizeof(double complex[N]));
	double complex (*V_r)[N] = z_matmal(N, N);

	// Eigenvalue and eigenvector solutions to the general eigenproblem
	// Remember to only work with the top half rows of V_r since they
	// are the only ones relevant and normalised.
	d_general_eig_solver(N, Q_l, Q_r, alpha, beta, V_r);
	free(Q_l);
	free(Q_r);

	double complex *ieig = malloc(sizeof(double complex[n]));

	// Copy T and T_t into complex matrices to reuse mode_filter
	double complex (*Tz)[n] = z_matmal(n, n);
	double complex (*Tz_t)[n] = z_matmal(n, n);
	#pragma omp parallel for simd collapse(2)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			Tz[i][j] = T[i][j];
			Tz_t[i][j] = T_t[i][j];
		}
	}


	mode_filter(N, V_r, alpha, beta, Tz, Tz_t, ieig, V_i);
	free(alpha);
	free(beta);
	free(V_r);

	double complex (*V_i_invers)[n] = z_matmal(n, n);

	// Copy the eigenvector matrix to calculate the inverse later on
	memcpy(*V_i_invers, *V_i, sizeof(complex double[n][n]));

	// Do the left handside multiplication for the self-energy
	// V_i * Lambda, overwrites V_i
	for (int i = 0; i < n; i++) {
		cblas_zscal(n, ieig + i, (*V_i) + i, n);
	}

	// Matrix product T * V_i * Lambda overwrites V_i
	double complex asca = 1;
	double complex bsca = 0;

	// Buffer variable for the matrix product
	double complex (*buf)[n] = z_matmal(n, n);

	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, &asca, *Tz, n, *V_i, n, &bsca, *buf, n);
	free(Tz);
	free(Tz_t);

	// Calculate the inverse of V_i
	int *ipiv = malloc(sizeof(int[n]));
	LAPACKE_zgetrf(LAPACK_ROW_MAJOR, n, n, *V_i_invers, n, ipiv);
	LAPACKE_zgetri(LAPACK_ROW_MAJOR, n, *V_i_invers, n, ipiv);
	free(ipiv);

	// Calculate T * V_i * Lambda * V_i_invers
	// V_i stores the total result
	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, &asca, *buf, n, *V_i_invers, n, &bsca, *V_i, n);
	free(V_i_invers);
	free(buf);

	return 1;
}

int
z_eigendecomposition(const size_t n, void *Mp, void *Tp, void *T_tp, double E, void *sigp)
{
//  Calculate self-energy of the given system via the eigendecomposition method. This method is used for complex input
//  Hamiltonians
//
//  PARAMETER:
//  	n		- number of columns (quadratic matrices)
//  	(*M)[n]		- hermitian unit cell hamiltonian
//  	(*T)[n]		- hopping matrix between unit cells
//  	(*T_t) [n]	- daggered hopping matrix between unit cells
//  	E		- Energy of the system

	double complex (*M)[n] = Mp;
	double complex (*T)[n] = Tp;
	double complex (*T_t)[n] = T_tp;
	double complex (*sigma)[n] = sigp;

	const size_t N = 2*n;
	double complex (*Q_l)[N] = z_matcal(N, N);
	double complex (*Q_r)[N] = z_matcal(N, N);
	z_quad_mat_construct(N, Q_l, Q_r, n, M, T, T_t, E);

	// Solve the generalised eigenproblem defined via the Q_l and Q_r matrix
	double complex *alpha = malloc(sizeof(double complex[N]));
	double complex *beta = malloc(sizeof(double complex[N]));
	double complex (*V_r)[N] = z_matmal(N, N);

	// Eigenvalue and eigenvector solutions to the general eigenproblem
	// Remember to only work with the top half rows of V_r since they
	// are the only ones relevant and normalised.
	z_general_eig_solver(N, Q_l, Q_r, alpha, beta, V_r);
	free(Q_l);
	free(Q_r);

	// Allocate memory for the eigenvalues inside integration region
	// Assign sigma as V_i since V_i will be the self-energy in the end
	double complex *ieig = malloc(sizeof(double complex[n]));
	double complex (*V_i)[n] = sigma;
	mode_filter(N, V_r, alpha, beta, T, T_t, ieig, V_i);
	free(alpha);
	free(beta);
	free(V_r);

	double complex (*V_i_invers)[n] = z_matmal(n, n);

	// Copy the eigenvector matrix to calculate the inverse later on
	memcpy(*V_i_invers, *V_i, sizeof(complex double[n][n]));

	// Do the left handside multiplication for the self-energy
	// V_i * Lambda, overwrites V_i
	for (int i = 0; i < n; i++) {
		cblas_zscal(n, ieig + i, (*V_i) + i, n);
	}

	// Matrix product T * V_i * Lambda overwrites V_i
	double complex asca = 1;
	double complex bsca = 0;

	// Buffer variable for the matrix product
	double complex (*buf)[n] = z_matmal(n, n);

	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, &asca, *T, n, *V_i, n, &bsca, *buf, n);

	// Calculate the inverse of V_i
	int *ipiv = malloc(sizeof(int[n]));
	LAPACKE_zgetrf(LAPACK_ROW_MAJOR, n, n, *V_i_invers, n, ipiv);
	LAPACKE_zgetri(LAPACK_ROW_MAJOR, n, *V_i_invers, n, ipiv);
	free(ipiv);

	// Calculate T * V_i * Lambda * V_i_invers
	// V_i stores the total result
	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, &asca, *buf, n, *V_i_invers, n, &bsca, *V_i, n);
	free(V_i_invers);
	free(buf);

	// V_i contains the final self-energy
	return 1;
}

void
d_quad_mat_construct(const size_t N, double (*Q_l)[N], double (*Q_r)[N], const size_t n, double (*M)[n], double (*T)[n], double (*T_t)[n], double E)
{
//  Construct the two 2*n x 2*n (N x N) matrices required for the generalised eigenproblem.
//  PARAMETER:
//  	N		- (2*n) number of columns (quadratic matrices)
//	(*Q_l)[N]	- Left handside matrix in the general eigenproblem
//	(*Q_r)[N]	- Right handside matrix in the general eigenproblem
//	n 		- number of columns of the hamiltonian
//  	(*M)[n]		- hermitian unit cell hamiltonian
//  	(*T)[n]		- hopping matrix between unit cells
//  	(*T_t) [n]	- daggered hopping matrix between unit cells
//  	E		- Energy of the system


	// Complete the top half of the left matrix
	// Right upper submatrix
	dqu_eye(N, Q_l, n, 1);

	// Complete the bottom half
	// Left lower submatrix
	blkview Qv = {.sr=n, .er=N, .sc=0, .ec=n};
	dqu_blkassign(N, &Qv, Q_l, n, n, T_t);
	dqu_sca_blkprod(N, &Qv, Q_l, -1);

	// Right lower submatrix
	Qv = (blkview){.sr=n, .er=N, .sc=n, .ec=N};
	dqu_blkeye(N, &Qv, Q_l, 0, -E);
	dqu_blksum(N, &Qv, Q_l, n, M);
	dqu_sca_blkprod(N, &Qv, Q_l, -1);

	// Complete the top half of the right matrix
	// Left upper submatrix
	Qv = (blkview){.sr=0, .er=n, .sc=0, .ec=n};
	dqu_blkeye(N, &Qv, Q_r, 0, 1);

	// Right bottom submatrix
	Qv = (blkview){.sr=n, .er=N, .sc=n, .ec=N};
	dqu_blkassign(N, &Qv, Q_r, n, n, T);
}

void
z_quad_mat_construct(const size_t N, double complex (*Q_l)[N], double complex (*Q_r)[N], const size_t n, double complex (*M)[n], double complex (*T)[n], double complex (*T_t)[n], double E)
{
//  Construct the two 2*n x 2*n (N x N) matrices required for the generalised eigenproblem.
//  PARAMETER:
//  	N		- (2*n) number of columns (quadratic matrices)
//	(*Q_l)[N]	- Left handside matrix in the general eigenproblem
//	(*Q_r)[N]	- Right handside matrix in the general eigenproblem
//	n 		- number of columns of the hamiltonian
//  	(*M)[n]		- hermitian unit cell hamiltonian
//  	(*T)[n]		- hopping matrix between unit cells
//  	(*T_t) [n]	- daggered hopping matrix between unit cells
//  	E		- Energy of the system


	// Complete the top half of the left matrix
	// Right upper submatrix
	zqu_eye(N, Q_l, n, 1);

	// Complete the bottom half
	// Left lower submatrix
	struct blkview Qv = {.sr=n, .er=N, .sc=0, .ec=n};
	zqu_blkassign(N, &Qv, Q_l, n, n, T_t);
	zqu_sca_blkprod(N, &Qv, Q_l, -1);

	// Right lower submatrix
	Qv = (struct blkview){.sr=n, .er=N, .sc=n, .ec=N};
	zqu_blkeye(N, &Qv, Q_l, 0, (double complex) -E);
	zqu_blksum(N, &Qv, Q_l, n, M);
	zqu_sca_blkprod(N, &Qv, Q_l, -1);

	// Complete the top half of the right matrix
	// Left upper submatrix
	Qv = (struct blkview){.sr=0, .er=n, .sc=0, .ec=n};
	zqu_blkeye(N, &Qv, Q_r, 0, 1);

	// Right bottom submatrix
	Qv = (struct blkview){.sr=n, .er=N, .sc=n, .ec=N};
	zqu_blkassign(N, &Qv, Q_r, n, n, T);
}

void
d_general_eig_solver(const size_t N, double (*Q_l)[N], double (*Q_r)[N], double complex *alpha, double complex *beta, double complex (*V_r)[N])
{
//  Wrapper for the zggev function. This returns eigenvalues and normalised right eigenvectors.
//  Also the matrix V_r gets split in half as the algorithm does not need the bottom rows.
//  PARAMETER:
//  	N		- (2*n) number of columns (quadratic matrices)
//	(*Q_l)[N]	- Left handside matrix in the general eigenproblem
//	(*Q_r)[N]	- Right handside matrix in the general eigenproblem
//	*alpha		- List for the numerator of eigenvalues (e[i] = alpha[i]/beta[i])
//	*beta		- List for the denumerator of eigenvalues
//  	(*V_r)[N]	- Matrix storing right eigenvectors column wise

	double *alphar = malloc(sizeof(double[N]));
	double *alphai = malloc(sizeof(double[N]));
	double *betar = malloc(sizeof(double[N]));
	double (*V_rr)[N] = d_matmal(N, N);

	LAPACKE_dggev(LAPACK_ROW_MAJOR, 'N', 'V', N, *Q_l, N, *Q_r, N, alphar, alphai, betar, NULL, N, *V_rr, N);

	int i;

	for (i = 0; i < N; i++) {
		alpha[i] = alphar[i] + I*alphai[i];
		beta[i] = betar[i];
		if (alphai[i] > 0) {
			for (int j = 0; j < N; j++) {
				V_r[j][i] = V_rr[j][i] + I*V_rr[j][i+1];
				V_r[j][i+1] = conj(V_r[j][i]);
			}
		} else if (alphai[i] == 0) {
			for (int j = 0; j < N; j++) {
				V_r[j][i] = V_rr[j][i];
			}
		}
	}
	free(alphar);
	free(alphai);
	free(betar);
	free(V_rr);

	// Normalise every column to the first half of rows
	#pragma omp parallel for private(i)
	for (i = 0; i < N; i++) {
		double norm = cblas_dznrm2(N/2, *V_r + i, N);
		cblas_zdscal(N/2, 1/norm, *V_r + i, N);
	}
}

void
z_general_eig_solver(const size_t N, double complex (*Q_l)[N], double complex (*Q_r)[N], double complex *alpha, double complex *beta, double complex (*V_r)[N])
{
//  Wrapper for the zggev function. This returns eigenvalues and normalised right eigenvectors.
//  Also the matrix V_r gets split in half as the algorithm does not need the bottom rows.
//  PARAMETER:
//  	N		- (2*n) number of columns (quadratic matrices)
//	(*Q_l)[N]	- Left handside matrix in the general eigenproblem
//	(*Q_r)[N]	- Right handside matrix in the general eigenproblem
//	*alpha		- List for the numerator of eigenvalues (e[i] = alpha[i]/beta[i])
//	*beta		- List for the denumerator of eigenvalues
//  	(*V_r)[N]	- Matrix storing right eigenvectors column wise

	LAPACKE_zggev(LAPACK_ROW_MAJOR, 'N', 'V', N, *Q_l, N, *Q_r, N, alpha, beta, NULL, N, *V_r, N);

	// Normalise every column to the first half of rows
	int i;
	#pragma omp parallel for private(i)
	for (i = 0; i < N; i++) {
		double norm = cblas_dznrm2(N/2, *V_r + i, N);
		cblas_zdscal(N/2, 1/norm, *V_r + i, N);
	}
}

void
mode_filter(const size_t N, double complex (*V_r)[N], double complex *alpha, double complex *beta, double complex (*T)[N/2], double complex (*T_t)[N/2], double complex *ieig, double complex (*V_i)[N/2])
{
//  Find all eigenvectors with eigenvalues inside the integration circle.
//  ebool (evan.), pbool (prop.) will have a 1 or 0 if the mode is evanescent or propagating.

	int* ebool = calloc(N, sizeof(int));
	int* pbool = calloc(N, sizeof(int));
	iregion_checker(N, alpha, beta, ebool, pbool);

	// Index list of all energies with evanescent modes
	int *eidx;
	size_t neidx = ib_nonzero(N, ebool, &eidx);

	// Write evanescent modes into the eigval and eigvec result array
	for (int i = 0; i < neidx; i++) {
		ieig[i] = alpha[eidx[i]]/beta[eidx[i]];
		cblas_zcopy(N/2, *V_r + eidx[i], N, *V_i + i, N/2);
	}

	// Index list of all energies with propagating modes
	int *pidx;
	size_t npidx = ib_nonzero(N, pbool, &pidx);

	// First find degenerate modes if there are propagating ones
	// every dbool row is an 1d array with 0 or 1. 1s stand for degenerate modes
	int **dbool;

	// These are the indices of the modes that go into one direction (right) and
	// have positive velocity. They form the self-energy of the system.
	// These indices are defined relative to the indexarray of all prop. modes pidx.
	int *ridx = malloc(sizeof(int [npidx/2]));
	if (npidx > 0) {
		// Create a temporary list of the eigvals of all propagating modes
		double complex *peig = malloc(sizeof(double complex [npidx]));
		for (int i = 0; i < npidx; i++) {
			peig[i] = alpha[pidx[i]]/beta[pidx[i]];
		}
		// nunique determines the rows of dbool
		size_t nunique = degeneracy_checker(npidx, pidx, peig, &dbool);

		// Check for right and left going velocities and put the index
		// of right-going velocities into rpidx
		velocity_checker(npidx, pidx, peig, N, V_r, nunique, dbool, T, T_t, ridx);
		for (int i = 0; i < npidx/2; i++) {
			// Since we already filled in the evanescent modes we have to
			// fill up from the neidx index.
			ieig[neidx + i] = peig[ridx[i]];
			cblas_zcopy(N/2, *V_r + pidx[ridx[i]], N, *V_i + neidx + i, N/2);
		}
	}
}

void
iregion_checker(const size_t N, double complex *alpha, double complex *beta, int *evan, int *prop)
{
// Find all eigenvalues inside the integration circle and all eigenvalues corresponding to propagating solutions
//	N		- (2*n) number of columns (quadratic matrices)
//	*alpha		- List for the numerator of eigenvalues (e[i] = alpha[i]/beta[i])
//	*beta		- List for the denumerator of eigenvalues
// 	*in		- List displaying eigvals in- (1) or outside (0) the i.c.
// 	*prop		- List displaying eigvals "propagating" (1) or not (0)

	for (int i = 0; i < N; i++) {
	// Populate the in and prop list.
		double a = cabs(alpha[i]);
		double b = cabs(beta[i]);
		if (a < (1+EPS) * b) {
		// Condition for modes inside integration circle
			if (a > (1-EPS) * b) {
			// Condition for propagating modes
				prop[i] = 1;
			}
			else {
			// The rest are evanescent modes
				evan[i] = 1;
			}
		}
	}
}
size_t
degeneracy_checker(const size_t npidx, int *pidx, double complex *peig, int ***dbool)
{
// Find degenerate eigenvalues in the set of propagating modes.
// PARAMETERS:
//	npidx		- number of propagating modes
//	*pidx		- array with indices of propagating modes
//	*peig		- eigenvalues of propagating modes
//	***dbool	- boolean array with information on degenerate modes
//
// RETURNS:
// 	nunique		- number of eigvals only counted once
// 	***dbool	- information on degeneracies are stored in this array.

	// Every row is a boolean array showing which modes are degenerate
	// Do NOT free sdbool, the memory is declared in here for functions outside
	int **sdbool = malloc(sizeof(int*[npidx]));
	int **mvdbool = sdbool;

	// Check for degeneracies
	// to_check marks values that are already checked
	int *to_check = calloc(npidx, sizeof(int));

	for (int i = 0; i < npidx; i++) {
		// 0 in to_check means that the value is to check 1 means it is already checked.
		if (to_check[i] == 0) {
			int *degen = malloc(sizeof(int[npidx]));
			iz_sca_equal(npidx, peig, peig[i], CEPS, degen);

			// Do not repeat comparison for degenerate modes
			b_logical_or(npidx, degen, to_check);

			// Save the pointer to the degen array in an entry in deg.
			*mvdbool = degen;
			mvdbool++;
		}
	}
	free(to_check);
	// Delete unnecessary entries in the sdeg array if degeneracies occured.
	size_t nunique = mvdbool - sdbool;
	if(nunique != npidx) {
		// Only reallocate if sdeg has too many entries.
		sdbool = realloc(sdbool, sizeof(int*[nunique]));
	}
	*dbool = sdbool;
	return nunique;
}

void
velocity_checker(const size_t npidx, int *pidx, double complex *peig, const size_t N, double complex (*V_r)[N], const size_t nunique, int **dbool, double complex (*T)[N/2], double complex (*T_t)[N/2], int *ridx)
{
//  Calculate the velocities of every non-degnerate and degenerate propagating mode to determine
//  if they are inside or outside the integration circle.
//  PARAMETERS:
//  npidx		- Total number of propagating modes
//  *pidx		- Indices of all propagating modes (in the full eigvec matrix)
//  *peig		- Eigenvalues of all propagating modes
//  N			- Total number of eigenstates in V_r
//  V_r			- All right eigenstates including the evanescent ones
//  nunique		- Total number of propagating eigenvalues ignoring degeneracies
//  **dbool		- Row-wise highlights degeneracies with 0 or 1's
//  *ridx		- Indices of right going eigenstates relevant for the self-energy
//
//  RETURNS:
//  *ridx		- The indices are written into ridx

	int **mvbool = dbool;
	for (int i = 0; mvbool - dbool < nunique; mvbool++, i++) {
		// We loop over every row in the boolean matrix showing the degeneracies
		int *degidx;
		size_t ndeg = ib_nonzero(npidx, *mvbool, &degidx);

		// For every degenerate mode in a row of dbool
		if (ndeg == 1) {
			// Buffer array for the result of V * eigv
			double complex *y = malloc(sizeof(double complex [N/2]));

			double complex b = 0;
			double complex *a = &b;
			// V * eigv
			cblas_zgemv(CblasRowMajor, CblasNoTrans, N/2, N/2, peig + degidx[0], *T, N/2, (*V_r) + pidx[degidx[0]], N, a, y, 1);

			double complex V;
			double complex *Vp = &V;
			// eigv.T.conj * V * eigv
			cblas_zdotc_sub(N/2, V_r[0] + pidx[degidx[0]], N, y, 1, Vp);
			if (-2*cimag(V) > 0) {
				// If velocity is greater than zero (V_tot = -2*imag(V))
				// Safe index of right-going eigenvector in ridx
				*ridx = degidx[0];
				ridx++;
			}
			free(y);
		}
	}
}
