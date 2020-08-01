#ifndef SCALE2D_TEST_H
#define SCALE2D_TEST_H

#include "myblas_internal.h"
#include "dgemm_test.h"

typedef void(scale2d_func_t)(double beta, double *C, size_t ldc, const block2d_info_t* info );

typedef struct _scale2d_test_t {
	scale2d_func_t * func;
	double           beta;
	double          *C;
	size_t           ldc;
	block2d_info_t*  info;
} scale2d_test_t;


void myblas_basic_scale2d(double beta, double *C, size_t ldc, const block2d_info_t* info );


#endif//SCALE2D_TEST_H
