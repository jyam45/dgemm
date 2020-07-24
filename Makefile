CC=gcc
FCC=gfortran

CFLAGS=-O3 --std=c11 -I./include/ -I./src/
LDFLAGS=-L./lib/
LIBS=-lcpuid -ltimer
CBLASLIBS= -lcblas -lblas

MYBLAS_OBJS= src/myblas_dgemm.o src/myblas_dgemm_main.o src/myblas_xerbla.o

all: libs test

libs: ./lib ./include lib/libblas.a lib/libcblas.a lib/libcpuid.a lib/libtimer.a

test: unit_test speed_test total_test

unit_test: test/unit_test.o test/check_error.o test/check_speed.o test/init_matrix.o $(MYBLAS_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) $(CBLASLIBS)

speed_test: test/speed_test.o test/check_error.o  test/check_speed.o test/init_matrix.o test/peak_flops.o $(MYBLAS_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) $(CBLASLIBS)

total_test: test/total_test.o test/check_error.o  test/check_speed.o test/init_matrix.o test/peak_flops.o $(MYBLAS_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) $(CBLASLIBS)

test/check_error.o : test/check_error.c test/dgemm_test.h
test/check_speed.o : test/check_speed.c test/dgemm_test.h include/Timer.h
test/init_matrix.o : test/init_matrix.c test/dgemm_test.h
test/peak_flops.o  : test/peak_flops.c test/dgemm_test.h include/cpuid.h
test/speed_test.o  : test/speed_test.c test/dgemm_test.h
test/unit_test.o   : test/unit_test.c test/dgemm_test.h
test/total_test.o  : test/total_test.c test/dgemm_test.h
test/dgemm_test.h  : include/cblas.h
src/myblas_dgemm.o      : src/myblas_dgemm.c src/myblas.h src/myblas_internal.h
src/myblas_dgemm_main.o : src/myblas_dgemm_main.c src/myblas_internal.h
src/myblas_xerbla.o     : src/myblas_xerbla.c src/myblas_internal.h

./lib:
	mkdir -p lib

./include:
	mkdir -p include

lib/libblas.a :
	cd blas; make

lib/libcblas.a : lib/libblas.a


lib/libcpuid.a:
	cd cpuid; make; make install;

lib/libtimer.a:
	cd timer; make;

.PHONY: clean distclean

clean:
	rm -f test/*.o src/*.o
	cd blas/ ; make clean
	cd cpuid/ ; make clean
	cd timer/ ; make clean

distclean: clean
	rm -f unit_test speed_test total_test
	cd blas/ ; make distclean 
	cd cpuid/ ; make distclean
	cd timer/ ; make distclean
	#rm -rf include/ lib/ 

.SUFFIXES: .o .c
.c.o:
	$(CC) $(CFLAGS) -c $< -o $@
