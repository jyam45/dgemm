CC=gcc
FCC=gfortran

CFLAGS=-O3 --std=c11 -I./include/ -I./src/ -I./test/
LDFLAGS=-L./lib/
LIBS=-lcpuid -ltimer
CBLASLIBS= -lcblas -lblas

MYBLAS_OBJS= src/myblas_dgemm.o src/myblas_dgemm_main.o src/myblas_xerbla.o \
             src/myblas_dgemm_scale2d.o src/myblas_dgemm_copy_n.o src/myblas_dgemm_copy_t.o src/myblas_dgemm_kernel.o

MYBLAS_COPY_OBJS=src/myblas_dgemm_copy_n.o src/myblas_dgemm_copy_t.o
MYBLAS_SCAL_OBJS=src/myblas_dgemm_scale2d.o
MYBLAS_KRNL_OBJS=src/myblas_dgemm_kernel.o

all: libs test

libs: ./lib ./include lib/libblas.a lib/libcblas.a lib/libcpuid.a lib/libtimer.a

test: unit_test speed_test total_test copy2d_unit_test copy2d_speed_test scale2d_unit_test scale2d_speed_test kernel_unit_test kernel_speed_test

unit_test: test/unit_test.o test/check_error.o test/check_matrix.o test/check_speed.o test/init_matrix.o $(MYBLAS_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) $(CBLASLIBS)

speed_test: test/speed_test.o test/check_error.o  test/check_speed.o test/check_matrix.o test/init_matrix.o test/peak_flops.o $(MYBLAS_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) $(CBLASLIBS)

total_test: test/total_test.o test/check_error.o  test/check_speed.o test/check_matrix.o test/init_matrix.o test/peak_flops.o $(MYBLAS_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) $(CBLASLIBS)

test/check_error.o : test/check_error.c test/dgemm_test.h
test/check_speed.o : test/check_speed.c test/dgemm_test.h include/Timer.h
test/check_matrix.o: test/check_matrix.c test/dgemm_test.h
test/check_array.o : test/check_array.c test/dgemm_test.h
test/init_matrix.o : test/init_matrix.c test/dgemm_test.h
test/peak_flops.o  : test/peak_flops.c test/dgemm_test.h include/cpuid.h
test/speed_test.o  : test/speed_test.c test/dgemm_test.h
test/unit_test.o   : test/unit_test.c test/dgemm_test.h
test/total_test.o  : test/total_test.c test/dgemm_test.h
test/dgemm_test.h  : include/cblas.h

src/myblas_xerbla.o     : src/myblas_xerbla.c src/myblas_internal.h
src/myblas_dgemm.o      : src/myblas_dgemm.c src/myblas.h src/myblas_internal.h
src/myblas_dgemm_main.o : src/myblas_dgemm_main.c src/myblas_internal.h
src/myblas_dgemm_scale2d.o : src/myblas_dgemm_scale2d.c src/myblas_internal.h
src/myblas_dgemm_copy_n.o  : src/myblas_dgemm_copy_n.c src/myblas_internal.h
src/myblas_dgemm_copy_t.o  : src/myblas_dgemm_copy_t.c src/myblas_internal.h
src/myblas_dgemm_kernel.o  : src/myblas_dgemm_kernel.c src/myblas_internal.h

copy2d_unit_test: test/copy2d/copy2d_unit_test.o test/copy2d/myblas_basic_copy_n.o test/copy2d/myblas_basic_copy_t.o test/check_array.o test/init_matrix.o $(MYBLAS_COPY_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) $(CBLASLIBS)

copy2d_speed_test: test/copy2d/copy2d_speed_test.o test/copy2d/myblas_basic_copy_n.o test/copy2d/myblas_basic_copy_t.o test/init_matrix.o $(MYBLAS_COPY_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) $(CBLASLIBS)

test/copy2d/copy2d_speed_test.o   : test/copy2d/copy2d_speed_test.c test/copy2d/copy2d_test.h include/Timer.h
test/copy2d/copy2d_unit_test.o    : test/copy2d/copy2d_unit_test.c test/copy2d/copy2d_test.h
test/copy2d/myblas_basic_copy_n.o : test/copy2d/myblas_basic_copy_n.c test/copy2d/copy2d_test.h
test/copy2d/myblas_basic_copy_t.o : test/copy2d/myblas_basic_copy_t.c test/copy2d/copy2d_test.h

scale2d_unit_test: test/scale2d/scale2d_unit_test.o test/scale2d/myblas_basic_scale2d.o test/check_matrix.o test/init_matrix.o $(MYBLAS_SCAL_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) $(CBLASLIBS)

scale2d_speed_test: test/scale2d/scale2d_speed_test.o test/scale2d/myblas_basic_scale2d.o test/init_matrix.o $(MYBLAS_SCAL_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) $(CBLASLIBS)

test/scale2d/scale2d_speed_test.o   : test/scale2d/scale2d_speed_test.c test/scale2d/scale2d_test.h include/Timer.h
test/scale2d/scale2d_unit_test.o    : test/scale2d/scale2d_unit_test.c test/scale2d/scale2d_test.h
test/scale2d/myblas_basic_scale2d.o : test/scale2d/myblas_basic_scale2d.c test/scale2d/scale2d_test.h

kernel_unit_test: test/kernel/kernel_unit_test.o test/kernel/myblas_basic_kernel.o test/check_matrix.o test/init_matrix.o $(MYBLAS_KRNL_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) $(CBLASLIBS)

kernel_speed_test: test/kernel/kernel_speed_test.o test/kernel/myblas_basic_kernel.o test/init_matrix.o test/peak_flops.o $(MYBLAS_KRNL_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) $(CBLASLIBS)

test/kernel/kernel_speed_test.o   : test/kernel/kernel_speed_test.c test/kernel/kernel_test.h include/Timer.h 
test/kernel/kernel_unit_test.o    : test/kernel/kernel_unit_test.c test/kernel/kernel_test.h  
test/kernel/myblas_basic_kernel.o : test/kernel/myblas_basic_kernel.c test/kernel/kernel_test.h

test/copy2d/copy2d_test.h    : test/dgemm_test.h src/myblas_internal.h
test/scale2d/scale2d_test.h  : test/dgemm_test.h src/myblas_internal.h
test/kernel/kernel_test.h    : test/dgemm_test.h src/myblas_internal.h

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
	rm -f test/*.o src/*.o test/*/*.o
	cd blas/ ; make clean
	cd cpuid/ ; make clean
	cd timer/ ; make clean

distclean: clean
	rm -f unit_test speed_test total_test copy2d_unit_test copy2d_speed_test scale2d_unit_test scale2d_speed_test kernel_unit_test kernel_speed_test
	cd blas/ ; make distclean 
	cd cpuid/ ; make distclean
	cd timer/ ; make distclean
	#rm -rf include/ lib/ 

.SUFFIXES: .o .c
.c.o:
	$(CC) $(CFLAGS) -c $< -o $@
