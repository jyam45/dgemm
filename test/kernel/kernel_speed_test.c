#include "kernel_test.h"
#include "myblas_internal.h"
#include "Timer.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

#define MAX_SIZE 2048
#define TILE_M     32 
#define TILE_N     32
#define TILE_K     32
#define SEED 13892393

#define ALIGNMENT_B  32

#define F_CORE  0x0001

#define  FLOP_SIMPLE     0
#define  FLOP_MINIMUM    1
#define  FLOP_REDUNDANT  2
#define  FLOP_IMPLEMENT  3

static kernel_func_t myblas_funcs[] = { myblas_dgemm_kernel, myblas_dgemm_kernel_core };

static void do_kernel( kernel_test_t *test ){

	test->func( test->alpha, test->A, test->B, test->C, test->ldc, test->info );

}

typedef double(*flop_count_func_t)(size_t M, size_t N, size_t K);

static flop_count_func_t flop_counters[]={flop_count_dgemm_simple,
                                          flop_count_dgemm_minimum,
                                          flop_count_dgemm_redundant,
                                          flop_count_dgemm_implement};

static void usage(){
	printf("usage : kernel_speed_test [-c] [-n <max_size>] [-i <begin_size>] [-d <incliment_size>] [-f {simple|minimum|redundant|implement}] [-h]\n\n");
	printf("  loop controll:\n");
	printf("     <max_size>       = max testing size of square matrix (default=2048):\n");
	printf("     <begin_size>     = initial testing size of square matrix (default=16):\n");
	printf("     <incliment_size> = inliment testing size of square matrix (default:twice):\n");
	printf("\n");
	printf("  flop count type:\n");
	printf("     simple    = 2*M*N*K\n");
	printf("     minimum   = M*N*(2*K+3)  (default)\n");
	printf("     redandant = M*N*(3*K+1)\n");
	printf("     implement = M*N*(2*K+K/32+1)\n");
	printf("\n");
}


int main( int argc, char** argv ){

	block3d_info_t sizes={MAX_SIZE,MAX_SIZE,MAX_SIZE,TILE_M,TILE_N,TILE_K};
	kernel_test_t test  ={NULL,0e0,NULL,NULL,NULL,MAX_SIZE,&sizes};
	double alpha = 1.9192829e0;

	// loop configuration
	size_t init = 16;
	size_t dist =  0;
	size_t nmax = MAX_SIZE;

	// flop count configuration
	size_t flop_type = FLOP_MINIMUM;

	// processing command option
	char*  argend;
	opterr = 0;
	int flags = 0;
	int c;
	while((c=getopt(argc,argv,"cf:n:i:d:h")) != -1 ){
		switch(c){
		case 'c' : flags |= F_CORE; break;
		case 'n' : nmax = strtol(optarg,&argend,10);if( *argend != '\0' ){ usage(); return -1; }; break;  
		case 'i' : init = strtol(optarg,&argend,10);if( *argend != '\0' ){ usage(); return -1; }; break;  
		case 'd' : dist = strtol(optarg,&argend,10);if( *argend != '\0' ){ usage(); return -1; }; break;  
		case 'f' :
			if     ( strncmp(optarg,"simple"   ,6)==0 ){ flop_type=FLOP_SIMPLE   ; }
			else if( strncmp(optarg,"minimum"  ,7)==0 ){ flop_type=FLOP_MINIMUM  ; }
			else if( strncmp(optarg,"redundant",9)==0 ){ flop_type=FLOP_REDUNDANT; }
			else if( strncmp(optarg,"implement",9)==0 ){ flop_type=FLOP_IMPLEMENT; }
			else { usage(); return -1; }
			break;
		case 'h' : usage(); return 0; 
		default  :
			printf("Option Error : %c\n",c);
			usage();
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

	rand_matrix( MAX_SIZE, MAX_SIZE, A, 1, MAX_SIZE, SEED ); // ColMajor
	rand_matrix( MAX_SIZE, MAX_SIZE, B, 1, MAX_SIZE, SEED ); // ColMajor

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
	flop_count_func_t flop_count = flop_counters[flop_type];

	int error = 0;

	printf("Max  Peak MFlops per Core: %G MFlops \n",mpeak);
	printf("Base Peak MFlops per Core: %G MFlops \n",bpeak);
	printf("size  , elapsed time[s],          MFlops,   base ratio[%%],    max ratio[%%] \n");
	size_t n = init;
	while ( n <= nmax ){
		test.info->M2 = n;
		test.info->N2 = n;

		init_matrix( n, n, C, 1, MAX_SIZE, 0e0  ); // ColMajor

		double t1 = get_realtime();
		do_kernel( &test );
		double t2 = get_realtime();
		double dt = t2 - t1;
		double nflop = flop_count(test.info->M2,test.info->N2,test.info->K2) ;
		double mflops = nflop / dt / 1000 / 1000;
		printf("%6u, %15G, %15G, %15G, %15G \n",n,dt,mflops,mflops/bpeak*100,mflops/mpeak*100);
		n = ( dist==0 ? n*2 : n+dist );
	}

	free(A);
	free(B);
	free(C);

	return error;
}

