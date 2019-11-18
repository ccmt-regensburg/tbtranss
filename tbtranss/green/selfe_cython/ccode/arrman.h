#ifndef _ARRMAN_H_
#define _ARRMAN_H_
	#include <stdio.h>
	#include <stdlib.h>
	#include <string.h>
	#include <complex.h>

// This header includes functions that manipulate values of a given matrix or prints them.

// FUNCTION Naming scheme:
// s, d, c 	- float, double, float complex (return value)
//  		- no specifier -> list or general matrix
// qu 		- quadratic matrix (equal rows and columns)
// blk		- functions with this prefix do something blockwise (blk)


// VARIABLE Naming scheme:
// A 		- list or matrix (capital letter)
// ma 		- number of rows (matrix)
// na		- number of elements (list) number of columns (matrix)
// val		- value to be saved in list or matrix
// k		- super or subdiagonal of given matrix
//

	// Structs used in the code
	typedef struct blkview {
	// VARIABLES:
	// sr		- starting row index
	// er		- ending row index
	// sc		- starting column index
	// ec		- ending column index
		size_t sr;
		size_t er;
		size_t sc;
		size_t ec;
	} blkview;

	// Functions creating contigouus matrices in memory
	extern void* s_matmal(size_t const ma, size_t const na);
	extern void* d_matmal(size_t const ma, size_t const na);
	extern void* c_matmal(size_t const ma, size_t const na);
	extern void* z_matmal(size_t const ma, size_t const na);


	extern void* s_matcal(size_t const ma, size_t const na);
	extern void* d_matcal(size_t const ma, size_t const na);
	extern void* c_matcal(size_t const ma, size_t const na);
	extern void* z_matcal(size_t const ma, size_t const na);

	// Functions working on 1d and 2d arrays.
	extern void s_fill(size_t const na, float *A, float const val);
	extern void d_fill(size_t const na, double *A, double const val);
	extern void c_fill(size_t const na, float complex *A, float complex const val);
	extern void z_fill(size_t const na, double complex *A, double complex const val);

	extern void s_linspace(size_t const na, float *A, float const start, float const stop);
	extern void d_linspace(size_t const na, double *A, double const start, double const stop);
	extern void s_arange(size_t const na, float *A, float const start, float const step);
	extern void d_arange(size_t const na, double *A, double const start, double const step);

	// Functions working on quadratic 2d arrays.
	extern void squ_eye(size_t const na, float (*A)[na], int const k, float const val);
	extern void dqu_eye(size_t const na, double (*A)[na], int const k, double const val);
	extern void cqu_eye(size_t const na, float complex (*A)[na], int const k, float complex const val);
	extern void zqu_eye(size_t const na, double complex (*A)[na], int const k, double complex const val);

	extern void squ_blkeye(size_t const na, blkview const *Ai, float (*A)[na], int const k, float const val);
	extern void dqu_blkeye(size_t const na, blkview const *Ai, double (*A)[na], int const k, double const val);
	extern void cqu_blkeye(size_t const na, blkview const *Ai, float complex (*A)[na], int const k, float complex const val);
	extern void zqu_blkeye(size_t const na, blkview const *Ai, double complex (*A)[na], int const k, double complex const val);

	extern void squ_blkassign(size_t const na, blkview const *Ai, float (*A)[na], size_t const mb, size_t const nb, float (*B)[nb]);
	extern void dqu_blkassign(size_t const na, blkview const *Ai, double (*A)[na], size_t const mb, size_t const nb, double (*B)[nb]);
	extern void cqu_blkassign(size_t const na, blkview const *Ai, float complex (*A)[na], size_t const mb, size_t const nb, float complex (*B)[nb]);
	extern void zqu_blkassign(size_t const na, blkview const *Ai, double complex (*A)[na], size_t const mb, size_t const nb, double complex (*B)[nb]);

	// Functions for printing.
	extern void i_print_list(size_t const na, int const *A);
	extern void s_print_list(size_t const na, float const *A);
	extern void d_print_list(size_t const na, double const *A);
	extern void c_print_list(size_t const na, float complex const *A);
	extern void z_print_list(size_t const na, double complex const *A);

	extern void s_print_mat(size_t const ma, size_t const na, float (*A)[na]);
	extern void d_print_mat(size_t const ma, size_t const na, double (*A)[na]);
	extern void c_print_mat(size_t const ma, size_t const na, float complex (*A)[na]);
	extern void z_print_mat(size_t const ma, size_t const na, double complex (*A)[na]);

#endif
