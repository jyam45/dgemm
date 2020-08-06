#ifndef KERNEL_TEST_H
#define KERNEL_TEST_H

#include "myblas_internal.h"
#include "dgemm_test.h"

typedef void(*kernel_func_t)(double alpha, const double *A, const double *B, 
                            double* C, size_t ldc, const block3d_info_t* info );

typedef struct _kernel_test_t {
	kernel_func_t    func;
	double           alpha;
	const double    *A;
	const double    *B;
	double          *C;
	size_t           ldc;
	block3d_info_t*  info;
} kernel_test_t;


void myblas_basic_kernel(double alpha, const double *A2, const double *B2, 
                         double *C, size_t ldc, const block3d_info_t* info );

void myblas_basic_kernel_core(double alpha, const double *A2, const double *B2, 
                              double *C, size_t ldc, const block3d_info_t* info );

#endif//KERNEL_TEST_H
