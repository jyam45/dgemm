#ifndef COPY2D_TEST_H
#define COPY2D_TEST_H

#include "myblas_internal.h"
#include "dgemm_test.h"

typedef void(*copy2d_func_t)(const double *A, size_t ldc, double *A2, const block2d_info_t* info );

typedef struct _copy2d_test_t {
	copy2d_func_t    func;
	const double    *A;
	size_t           lda;
	double          *buf;
	block2d_info_t*  info;
} copy2d_test_t;


void myblas_basic_copy_n(const double* B, size_t ldb, double* B2, const block2d_info_t* info );
void myblas_basic_copy_t(const double* A, size_t lda, double* A2, const block2d_info_t* info );

void myblas_basic_copy_n_core(const double* B, size_t ldb, double* B2, const block2d_info_t* info );
void myblas_basic_copy_t_core(const double* A, size_t lda, double* A2, const block2d_info_t* info );

#endif//COPY2D_TEST_H
