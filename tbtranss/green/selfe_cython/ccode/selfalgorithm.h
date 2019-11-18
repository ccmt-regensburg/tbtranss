#ifndef _SELFALGORITHM_H_
#define _SELFALGORITHM_H_
	#include <stdlib.h>
	#include <complex.h>
	#include "arrman.h"
	#include "arrcal.h"
	#include "arrchk.h"

	#define MKL_Complex16 double complex
	#include <mkl.h>

	//Functions to be forwared to cython
	extern int d_eigendecomposition(const size_t n, void *M, void *T, void *T_t, double E, void *sigma);
	extern int z_eigendecomposition(const size_t n, void *M, void *T, void *T_t, double E, void *sigma);

	extern int z_decimation(const size_t n, void *M, void *T, void *T_t, double E, int iter, double eps, void *sigma);
	extern int d_decimation(const size_t n, void *M, void *T, void *T_t, double E, int iter, double eps, void *sigma);


#endif
