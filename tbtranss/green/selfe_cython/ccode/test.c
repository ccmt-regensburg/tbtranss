#include "selfalgorithm.h"
#define NUM 4

int
main()
{
	double t = 1;
	double complex (*M)[NUM] = z_matcal(NUM, NUM);

	zqu_eye(NUM, M, 1, t);
	zqu_eye(NUM, M, -1, t);

	double complex (*T)[NUM] = z_matcal(NUM, NUM);

	zqu_eye(NUM, T, 0, t);

	double complex (*sigma)[NUM] = z_matcal(NUM, NUM);
	z_eigendecomposition(NUM, M, T, T, 1, sigma);
	z_print_mat(NUM, NUM, sigma);

	return 1;
}


