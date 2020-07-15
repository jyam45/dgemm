#include "dgemm_test.h"
#include <stdio.h>

#define  MAX_SIZE 2048

int main(int argc, char** arv){

	int error = 0;

	flops_info_t cpu;

	double *A, *B, *C;
	dgemm_test_t myblas = { cblas_dgemm, CblasRowMajor, CblasNoTrans, CblasNoTrans, MAX_SIZE, MAX_SIZE, MAX_SIZE, 1e0, NULL, MAX_SIZE, NULL, MAX_SIZE, 1e0, NULL, MAX_SIZE };

	A = calloc( MAX_SIZE*MAX_SIZE, sizeof(double) );
	B = calloc( MAX_SIZE*MAX_SIZE, sizeof(double) );
	C = calloc( MAX_SIZE*MAX_SIZE, sizeof(double) );

	myblas.A = A;
	myblas.B = B;
	myblas.C = C;

	peak_flops( &cpu );
	double mpeak = cpu.mflops_double_max;
	double bpeak = cpu.mflops_double_base;

	printf("Max  Peak MFlops : %G MFlops \n",mpeak);
	printf("Base Peak MFlops : %G MFlops \n",bpeak);

	printf("size  , elapsed time[s],          MFlops,   base ratio[%%],    max ratio[%%] \n");
	for( size_t n=16; n<=MAX_SIZE; n=n*2 ){

		myblas.M   = n;
		myblas.N   = n;
		myblas.K   = n;
		myblas.lda = n;
		myblas.ldb = n;
		myblas.ldc = n;

		double nflop = ((double)myblas.M) * ((double)myblas.K) * ((double)myblas.N) * 2; // MUL + ADD
		double dt = check_speed( &myblas );
		double mflops = nflop / dt / 1000 / 1000;

		printf("%6u, %15G, %15G, %15G, %15G \n",n,dt,mflops,mflops/bpeak*100,mflops/mpeak*100);

	}

	free(A);
	free(B);
	free(C);

	return error;
}
