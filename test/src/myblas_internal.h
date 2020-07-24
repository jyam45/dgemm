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

void myblas_dgemm_main( gemm_args_t* args );


#endif//MYBLAS_INTERNAL_H
