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
	size_t num_cores;
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

int    check_matrix( const int m, const int n, const int k,
                     const double *C1, const int di1, const int dj1,
                     const double *C2, const int di2, const int dj2 );

void   print_matrix( size_t M, size_t N, const double *A, size_t lda );

int    check_array( const size_t n, const double *a, const double *b );

double flop_count_dgemm_simple(size_t M, size_t N, size_t K);
double flop_count_dgemm_minimum(size_t M, size_t N, size_t K);
double flop_count_dgemm_redundant(size_t M, size_t N, size_t K);
double flop_count_dgemm_implement(size_t M, size_t N, size_t K);
	
#endif//DTEMM_TEST_H
