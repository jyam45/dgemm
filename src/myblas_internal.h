#ifndef MYBLAS_INTERNAL_H
#define MYBLAS_INTERNAL_H

#include <stddef.h>

#define  MASK_TRANS  0x01
#define  MASK_CONJ   0x02

typedef struct _gemm_args_t {
	size_t       TransA;
	size_t       TransB;
	size_t       M;
	size_t       N;
	size_t       K;
	double       alpha;
	const double *A;
	size_t       lda;
	const double *B;
	size_t       ldb;
	double       beta;
	double       *C;
	size_t       ldc;
} gemm_args_t;

typedef struct _block3d_info_t {
	size_t i2;
	size_t j2;
	size_t k2;
	size_t M;
	size_t N;
	size_t K;
	size_t M2;
	size_t N2;
	size_t K2;
	size_t tile_M;
	size_t tile_N;
	size_t tile_K;
} block3d_info_t;

typedef struct _block2d_info_t {
	size_t i2;
	size_t j2;
	size_t M;
	size_t N;
	size_t M2;
	size_t N2;
	size_t tile_M;
	size_t tile_N;
} block2d_info_t;



void myblas_dgemm_main( gemm_args_t* args );

void myblas_dgemm_scale2d(double beta, double *C, size_t ldc, const block2d_info_t* info );
void myblas_dgemm_copy_t(const double* A, size_t lda, double* A2, const block2d_info_t* info );
void myblas_dgemm_copy_n(const double* B, size_t ldb, double* B2, const block2d_info_t* info );
void myblas_dgemm_kernel(double alpha, const double *A2, const double *B2, 
                         double *C, size_t ldc, const block3d_info_t* info );


#endif//MYBLAS_INTERNAL_H
