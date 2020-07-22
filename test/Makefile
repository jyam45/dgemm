CC=gcc
FCC=gfortran

CFLAGS=-O3 --std=c11 -I./include/
LDFLAGS=-L./lib/
LIBS=-lcpuid -ltimer
CBLASLIBS= -lcblas -lblas

all: libs test

libs: ./lib ./include lib/libblas.a lib/libcblas.a lib/libcpuid.a lib/libtimer.a

test: unit_test speed_test total_test

unit_test: src/unit_test.o src/check_error.o src/check_speed.o src/init_matrix.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) $(CBLASLIBS)

speed_test: src/speed_test.o src/check_error.o  src/check_speed.o src/init_matrix.o src/peak_flops.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) $(CBLASLIBS)

total_test: src/total_test.o src/check_error.o  src/check_speed.o src/init_matrix.o src/peak_flops.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) $(CBLASLIBS)

src/check_error.o : src/check_error.c src/dgemm_test.h
src/check_speed.o : src/check_speed.c src/dgemm_test.h include/Timer.h
src/init_matrix.o : src/init_matrix.c src/dgemm_test.h
src/peak_flops.o  : src/peak_flops.c src/dgemm_test.h include/cpuid.h
src/speed_test.o  : src/speed_test.c src/dgemm_test.h
src/unit_test.o   : src/unit_test.c src/dgemm_test.h
src/total_test.o  : src/total_test.c src/dgemm_test.h
src/dgemm_test.h  : include/cblas.h

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
	rm -f src/*.o
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
