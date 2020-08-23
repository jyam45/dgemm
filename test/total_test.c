#include "dgemm_test.h"
#include "myblas.h"
#include <stdio.h>

#define  MAX_SIZE 2048
#define  ALIGNMENT_B  32 // AVX2

int main(int argc, char** arv){

	int error = 0;

	flops_info_t cpu;

	double *A, *B, *C;
	dgemm_test_t myblas = {myblas_dgemm, CblasRowMajor, CblasNoTrans, CblasNoTrans, MAX_SIZE, MAX_SIZE, MAX_SIZE, 1e0, NULL, MAX_SIZE, NULL, MAX_SIZE, 1e0, NULL, MAX_SIZE };

	//A = calloc( MAX_SIZE*MAX_SIZE, sizeof(double) );
	//B = calloc( MAX_SIZE*MAX_SIZE, sizeof(double) );
	//C = calloc( MAX_SIZE*MAX_SIZE, sizeof(double) );
	A = aligned_alloc( ALIGNMENT_B, MAX_SIZE*MAX_SIZE*sizeof(double) );
	B = aligned_alloc( ALIGNMENT_B, MAX_SIZE*MAX_SIZE*sizeof(double) );
	C = aligned_alloc( ALIGNMENT_B, MAX_SIZE*MAX_SIZE*sizeof(double) );

	myblas.A = A;
	myblas.B = B;
	myblas.C = C;

	peak_flops( &cpu );
	double mpeak = cpu.mflops_double_max  / cpu.num_cores;
	double bpeak = cpu.mflops_double_base / cpu.num_cores;

	printf("Max  Peak MFlops per Core: %G MFlops \n",mpeak);
	printf("Base Peak MFlops per Core: %G MFlops \n",bpeak);

	printf("size  , elapsed time[s],          MFlops,   base ratio[%%],    max ratio[%%] \n");
	//for( size_t n=32; n<=MAX_SIZE; n=n+32 ){
	for( size_t n=16; n<=MAX_SIZE; n=n+16 ){

		myblas.M   = n;
		myblas.N   = n;
		myblas.K   = n;
		myblas.lda = n;
		myblas.ldb = n;
		myblas.ldc = n;

		//double nflop = ((double)myblas.M) * ((double)myblas.K) * ((double)myblas.N) * 2 + ((double)myblas.M) * ((double)myblas.N) * 2 ; // A*B+C, alpha*AB, beta*C
		//  C(i,j) = sum_K{alpha*A(i,k)*B(k,j)} + beta*C(i,j) -> *=2*K*M*N +=(K-1)*M*N +=M*N *=M*N -> 2*K*M*N + K*M*N - M*N + 2*M*N -> 3*K*M*N + M*N -> (3*K+1)*M*N
		double nflop = (3*((double)myblas.K)+1)*((double)myblas.M) * ((double)myblas.N) ; // A*B+C, alpha*AB, beta*C
		//  C(i,j) = alpha*sum_K{A(i,k)*B(k,j)} + beta*C(i,j) -> *=M*N, *=K*M*N, +=(K-1)*M*N, +=M*N, *=M*N -> (2*K-1)*M*N + 3*M*N -> (2*K+2)*M*N -> 2*(K+1)*M*N
		//double nflop = 2*(((double)myblas.K)+1)*((double)myblas.M) * ((double)myblas.N) ; // A*B+C, alpha*AB, beta*C
		double dt = check_speed( &myblas );
		double mflops = nflop / dt / 1000 / 1000;

		printf("%6u, %15G, %15G, %15G, %15G \n",n,dt,mflops,mflops/bpeak*100,mflops/mpeak*100);

	}

	free(A);
	free(B);
	free(C);

	return error;
}
