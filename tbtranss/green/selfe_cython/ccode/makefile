CC=icc
OBJ=arrman.o arrcal.o arrchk.o

ifeq ($(CC), icc)
	CPFLAGS = -g -Wall -O3 -std=c11 -xHost -fPIC -qopenmp -m64 -I${MKLROOT}/include -D_Float128=__float128
	LDPFLAGS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

	CSFLAGS = -g -Wall -O3 -std=c11 -march=native -fPIC -I${MKLROOT}/include
	LDSFLAGS = -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

else
	CPFLAGS = -g -Wall -O3 -std=c11 -xHost -fPIC -fopenmp -I${MKLROOT}/include -D_Float128=__float128
	LDPFLAGS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

	CSFLAGS = -g -Wall -O3 -std=c11 -march=native -fopenmp -D_Float128=__float128
	LDSFLAGS = -lcblas -llapacke -lm
endif

parallel:
	$(CC) $(CPFLAGS) -c eigendecomposition.c
	$(CC) $(CPFLAGS) -c decimation.c
	cp eigendecomposition.o ..
	cp decimation.o ..

sequential:
	sed '/pragma/s/^/\/\//g' eigendecomposition.c > eigendecomposition_seq.c
	$(CC) $(CSFLAGS) -c eigendecomposition_seq.c
	rm eigendecomposition_seq.c

	sed '/pragma/s/^/\/\//g' decimation.c > decimation_seq.c
	$(CC) $(CSFLAGS) -c decimation_seq.c
	rm decimation_seq.c

test: test.o test.c parallel
	$(CC) $(CPFLAGS) -o test test.o $(OBJ) eigendecomposition.o decimation.o $(LDPFLAGS)

test.o:
	$(CC) $(CPFLAGS) -c test.c

clean:
	rm *.o
