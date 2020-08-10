#include "kernel_test.h"
#include "myblas_internal.h"
#include "Timer.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define MAX_SIZE 2048
#define TILE_M     32 
#define TILE_N     32
#define TILE_K     32
#define SEED 13892393

#define ALIGNMENT_B  32

#define F_CORE  0x0001

static kernel_func_t myblas_funcs[] = { myblas_dgemm_kernel, myblas_dgemm_kernel_core };

static void do_kernel( kernel_test_t *test ){

	test->func( test->alpha, test->A, test->B, test->C, test->ldc, test->info );

}

int main( int argc, char** argv ){

	block3d_info_t sizes={MAX_SIZE,MAX_SIZE,MAX_SIZE,TILE_M,TILE_N,TILE_K};
	kernel_test_t test  ={NULL,0e0,NULL,NULL,NULL,MAX_SIZE,&sizes};
	double alpha = 1.9192829e0;

	opterr = 0;
	int flags = 0;
	int c;
	while((c=getopt(argc,argv,"c")) != -1 ){
		switch(c){
		case 'c' : flags |= F_CORE; break;
		default  :
			printf("Unknown option : %c\n",c);
			return -1;
		}
	}


	test.func = myblas_funcs[flags];

	//double* A = calloc(MAX_SIZE*MAX_SIZE,sizeof(double));
	//double* B = calloc(MAX_SIZE*MAX_SIZE,sizeof(double));
	//double* C = calloc(MAX_SIZE*MAX_SIZE,sizeof(double));
	double* A  = aligned_alloc(ALIGNMENT_B,MAX_SIZE*MAX_SIZE*sizeof(double));
	double* B  = aligned_alloc(ALIGNMENT_B,MAX_SIZE*MAX_SIZE*sizeof(double));
	double* C  = aligned_alloc(ALIGNMENT_B,MAX_SIZE*MAX_SIZE*sizeof(double));
	double* buf= aligned_alloc(ALIGNMENT_B,MAX_SIZE*MAX_SIZE*2*sizeof(double));

	rand_matrix( MAX_SIZE, MAX_SIZE, A, 1, MAX_SIZE, SEED ); // ColMajor
	rand_matrix( MAX_SIZE, MAX_SIZE, B, 1, MAX_SIZE, SEED ); // ColMajor

	sizes.buf = buf;
	sizes.use_buffer = 1;

	test.A    = A;
	test.B    = B;
	test.C    = C;
	test.ldc  = MAX_SIZE;
	test.info = &sizes;
	test.alpha= alpha;

	flops_info_t cpu;
	peak_flops( &cpu );
	double mpeak = cpu.mflops_double_max  / cpu.num_cores;
	double bpeak = cpu.mflops_double_base / cpu.num_cores;

	int error = 0;

	printf("Max  Peak MFlops per Core: %G MFlops \n",mpeak);
	printf("Base Peak MFlops per Core: %G MFlops \n",bpeak);
	printf("size  , elapsed time[s],          MFlops,   base ratio[%%],    max ratio[%%] \n");
	for( size_t n=16; n <= MAX_SIZE; n*=2 ){
		test.info->M2 = n;
		test.info->N2 = n;

		init_matrix( n, n, C, 1, MAX_SIZE, 0e0  ); // ColMajor

		double t1 = get_realtime();
		do_kernel( &test );
		double t2 = get_realtime();
		double dt = t2 - t1;
		double nflop = ((double)test.info->M2) * ((double)test.info->K2) * ((double)test.info->N2) * 2 + ((double)test.info->M2) * ((double)test.info->N2) * 2 ; // A*B+C, alpha*AB, beta*C
		double mflops = nflop / dt / 1000 / 1000;
		printf("%6u, %15G, %15G, %15G, %15G \n",n,dt,mflops,mflops/bpeak*100,mflops/mpeak*100);
	}

	free(A);
	free(B);
	free(C);
	free(buf);

	return error;
}

