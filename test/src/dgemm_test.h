#ifndef DGEMM_TEST_H
#define DGEMM_TEST_H

#include <stdlib.h>
#include "cblas.h"

typedef void (*dgemm_func_t) 
             (const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
              const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
              const int K, const double alpha, const double *A,
              const int lda, const double *B, const int ldb,
              const double beta, double *C, const int ldc);

typedef struct _dgemm_test_t {
	dgemm_func_t dgemm;
	enum CBLAS_ORDER Order;
	enum CBLAS_TRANSPOSE TransA;
	enum CBLAS_TRANSPOSE TransB;
	int M;
	int N;
	int K;
	double alpha;
	double* A;
	int lda;
	double* B;
	int ldb;
	double beta;
	double* C;
	int ldc;
} dgemm_test_t;

typedef struct _flops_info_t {
	size_t base_freq;
	size_t max_freq;
	size_t fp_operator;
	size_t fp_operation;
	size_t vlen_bits;
	double mflops_single_base;
	double mflops_single_max; 
	double mflops_double_base;
	double mflops_double_max; 
} flops_info_t;


void   peak_flops( flops_info_t* out );

void   do_dgemm( dgemm_test_t* test );

void   init_matrix( const int m, const int n, double* A, const int di, const int dj, double value );

void   rand_matrix( const int m, const int n, double* A, const int di, const int dj, uint64_t seed );

int    check_error( const dgemm_test_t *test1, const dgemm_test_t *test2 );

double check_speed( dgemm_test_t* test );

#endif//DTEMM_TEST_H
