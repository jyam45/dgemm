#include "dgemm_test.h"

double flop_count_dgemm_simple(size_t M, size_t N, size_t K){
	double m = (double)M;
	double n = (double)N;
	double k = (double)K;
	return 2*m*n*k;
}

double flop_count_dgemm_minimum(size_t M, size_t N, size_t K){
	double m = (double)M;
	double n = (double)N;
	double k = (double)K;
	return m*n*(2*k+3);
}

double flop_count_dgemm_redundant(size_t M, size_t N, size_t K){
	double m = (double)M;
	double n = (double)N;
	double k = (double)K;
	return m*n*(3*k+1);
}

double flop_count_dgemm_implement(size_t M, size_t N, size_t K){
	double m = (double)M;
	double n = (double)N;
	double k = (double)K;
	double k_tile = k/32;
	return m*n*(2*k+k_tile+1);
}
