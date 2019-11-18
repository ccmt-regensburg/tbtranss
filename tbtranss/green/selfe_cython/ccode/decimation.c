#include "selfalgorithm.h"

int
z_decimation(const size_t n,  void *Mp, void *Tp, void *T_tp, double E, int iter, double eps, void *sigp)
{

	double complex (*M)[n] = Mp;
	double complex (*T)[n] = Tp;
	double complex (*T_t)[n] = T_tp;
	double complex (*sigma)[n] = sigp;

	double complex (*ME)[n] = z_matmal(n, n);
	memcpy(*ME, *M, sizeof(double complex [n][n]));

	// Write the energy on the diagonal of ME
	#pragma omp parallel for simd
	for (int i = 0; i < n; i++) {
		(*ME)[i * (n + 1)] -= E + I * eps;
	}

	double complex (*green)[n] = z_matmal(n, n);

	int *ipiv = malloc(sizeof(int[n]));
	double complex asca = 1;
	double complex bsca = 0;

	for (int i = 0; i < iter; i++) {

		#pragma omp parallel for simd
		for (int j = 0; j < n*n; j++) {
			(*green)[j] = - (*ME)[j] - (*sigma)[j];
		}

		LAPACKE_zgetrf(LAPACK_ROW_MAJOR, n, n, *green, n, ipiv);
		LAPACKE_zgetri(LAPACK_ROW_MAJOR, n, *green, n, ipiv);

		// Calculate green * T_t save in sigma
		cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, &asca, *green, n, *T_t, n, &bsca, *sigma, n);

		// Calculate T * green * T_t save in sigma
		cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, &asca, *T, n, *green, n, &bsca, *sigma, n);
	}

	return 1;
}

int
d_decimation(const size_t n,  void *Mp, void *Tp, void *T_tp, double E, int iter, double eps, void *sigp)
{

	double (*M)[n] = Mp;
	double (*T)[n] = Tp;
	double (*T_t)[n] = T_tp;
	double complex (*sigma)[n] = sigp;

	double (*ME)[n] = d_matmal(n, n);
	memcpy(*ME, *M, sizeof(double [n][n]));

	// Write the energy on the diagonal of ME
	#pragma omp parallel for simd
	for (int i = 0; i < n; i++) {
		(*ME)[i * (n + 1)] -= E + I * eps;
	}

	double complex (*green)[n] = z_matmal(n, n);

	int *ipiv = malloc(sizeof(int[n]));
	double complex asca = 1;
	double complex bsca = 0;

	for (int i = 0; i < iter; i++) {

		#pragma omp parallel for simd
		for (int j = 0; j < n*n; j++) {
			(*green)[j] = - (*ME)[j] - (*sigma)[j];
		}

		LAPACKE_zgetrf(LAPACK_ROW_MAJOR, n, n, *green, n, ipiv);
		LAPACKE_zgetri(LAPACK_ROW_MAJOR, n, *green, n, ipiv);

		// Calculate green * T_t save in sigma
		cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, &asca, *green, n, *T_t, n, &bsca, *sigma, n);

		// Calculate T * green * T_t save in sigma
		cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, &asca, *T, n, *green, n, &bsca, *sigma, n);
	}

	return 1;
}
