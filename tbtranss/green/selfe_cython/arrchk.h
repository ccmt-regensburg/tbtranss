#ifndef _ARRCHK_H_
#define _ARRCHK_H_
	#include <stdlib.h>
	#include <complex.h>
	#include <math.h>

// This header includes functions that check given arrays or matrices for certain properties
// and returns truth values or indices of the entry.

// FUNCTION Naming scheme:
// First
// b, i 	- boolean (in our case int either 0 or 1), integer (or size_t)
// s, d, c, z 	- float, double, float complex, double complex (return value)
//  		- no specifier -> list or general matrix
// qu 		- quadratic matrix (equal rows and columns)


// VARIABLE Naming scheme:
// A 		- list or matrix (capital letter)
// ma 		- number of rows (matrix)
// na		- number of elements (list) number of columns (matrix)
// val		- value to be compared to list or matrix
// k		- super or subdiagonal of given matrix
//

	// Compare two arrays for equality with threshold eps
	extern int bs_equal(size_t const na, float const *A, size_t const nb, float const *B, float const eps);
	extern int bd_equal(size_t const na, double const *A, size_t const nb, double const *B, double const eps);
	extern int bc_equal(size_t const na, float complex const *A, size_t const nb, float complex const *B, float const eps);
	extern int bz_equal(size_t const na, double complex const *A, size_t const nb, double complex const *B, double const eps);

	// Compare one array with a scalar using threshold eps
	extern int is_sca_equal(size_t const na, float const *A, float const val, float const eps, int *res);
	extern int id_sca_equal(size_t const na, double const *A, double const val, double const eps, int *res);
	extern int ic_sca_equal(size_t const na, float complex const *A, float complex const val, float const eps, int *res);
	extern int iz_sca_equal(size_t const na, double complex const *A, double complex const val, double const eps, int *res);

	// Find non-zero entries of a boolean (int) array
	extern size_t ib_nonzero(size_t const na, int const *A, int **idx);

	// Perform logical_and or logical_or operation on two arrays
	extern void b_logical_or(size_t const na, int const *A, int *B);
	extern void b_logical_and(size_t const na, int const *A, int *B);


#endif
