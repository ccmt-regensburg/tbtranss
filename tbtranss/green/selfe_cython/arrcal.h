#ifndef _ARRCAL_H_
#define _ARRCAL_H_
	#include <stdlib.h>
	#include <complex.h>
	#include "arrman.h"

// Due to a "bug" in C: float (*p)[N] does not implicitly convert to float const (*p)[N]. We cannot use the const
// modifier with arrays that will NOT get modified. This would impose ugly casts that we want to avoid when using
// this library. "See https://stackoverflow.com/questions/28062095/pass-a-two-dimensional-array-to-a-function-of-constant-parameter" for
// more information.
// This header includes functions that manipulate values of a given matrix or prints them.

// FUNCTION Naming scheme:
// s, d, c 	- float, double, complex (return value)
//  		- no specifier -> list or general matrix
// qu 		- quadratic matrix (equal rows and columns)
// blk		- functions with this prefix do something blockwise (blk)
// sca		- (scalar) functions operate on a matrix via a scalar


// VARIABLE Naming scheme:
// A 		- list or matrix (capital letter)
// ma 		- number of rows (matrix)
// na		- number of elements (list) number of columns (matrix)
// val		- value to do an operation in the matrix
// k		- super or subdiagonal of given matrix

	//Multiply or add scalars to matrix blocks
	extern void squ_sca_blksum(size_t const na, struct blkview const *Ai, float (*A)[na], float const val);
	extern void dqu_sca_blksum(size_t const na, struct blkview const *Ai, double (*A)[na], double const val);
	extern void cqu_sca_blksum(size_t const na, struct blkview const *Ai, float complex (*A)[na], float complex const val);
	extern void zqu_sca_blksum(size_t const na, struct blkview const *Ai, double complex (*A)[na], double complex const val);


	extern void squ_sca_blkprod(size_t const na, struct blkview const *Ai, float (*A)[na], float const val);
	extern void dqu_sca_blkprod(size_t const na, struct blkview const *Ai, double (*A)[na], double const val);
	extern void cqu_sca_blkprod(size_t const na, struct blkview const *Ai, float complex (*A)[na], float complex const val);
	extern void zqu_sca_blkprod(size_t const na, struct blkview const *Ai, double complex (*A)[na], double complex const val);

	//Multiply or add matrices to matrix blocks
	extern void squ_blksum(size_t const na, struct blkview const *Ai, float (*A)[na], size_t const nb, float (*B)[nb]);
	extern void dqu_blksum(size_t const na, struct blkview const *Ai, double (*A)[na], size_t const nb, double (*B)[nb]);
	extern void cqu_blksum(size_t const na, struct blkview const *Ai, float complex (*A)[na], size_t const nb, float complex (*B)[nb]);
	extern void zqu_blksum(size_t const na, struct blkview const *Ai, double complex (*A)[na], size_t const nb, double complex (*B)[nb]);

	//Trace
	extern float squ_trace(size_t const na, float (*A)[na]);
	extern double dqu_trace(size_t const na, double (*A)[na]);
	extern float complex cqu_trace(size_t const na, float complex (*A)[na]);
	extern double complex zqu_trace(size_t const na, double complex (*A)[na]);



#endif
