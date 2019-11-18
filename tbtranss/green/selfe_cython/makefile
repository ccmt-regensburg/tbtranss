CFLAGS = -g -O3 -xHost -fPIC
LDFLAGS =

CPYFLAGS = -I/opt/miniconda3/envs/idp/include/python3.6m -I
LDPYFLAGS = -lpython3.6m
LDFLAGS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -lmkl_rt -lmkl_def

CC = icc
COBJ = eigendecomposition.o decimation.o arrman.o arrchk.o arrcal.o
SCOBJ = eigendecomposition_seq.o decimation_seq.o arrman_seq.o arrchk_seq.o arrcal_seq.o
PYOBJ = selfe_cython.o

.PHONY: self_energy cobjects

parallel: pcobjects
	cython -2 selfe_cython.pyx
	$(CC) $(CFLAGS) -c selfe_cython.c -o $(PYOBJ) $(CPYFLAGS)
	$(CC) -g -O3 -shared $(CFLAGS) -o selfe_cython.so $(COBJ) $(PYOBJ) $(LDFLAGS) $(LDPYFLAGS)

sequential: scobjects
	cython -2 selfe_cython.pyx
	$(CC) $(CFLAGS) -c selfe_cython.c -o $(PYOBJ) $(CPYFLAGS)
	$(CC) -g -O3 -shared $(CFLAGS) -o selfe_cython.so $(SCOBJ) $(PYOBJ) $(LDFLAGS) $(LDPYFLAGS)

pcobjects:
	cd ccode && $(MAKE) parallel

scobjects:
	cd ccode && $(MAKE) sequential

clean:
	rm -f *.c *.o *.so
