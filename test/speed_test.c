#include "dgemm_test.h"
#include "myblas.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define  MAX_SIZE 2048
#define  ALIGNMENT_B  32 // AVX2

#define  FLOP_SIMPLE     0
#define  FLOP_MINIMUM    1
#define  FLOP_REDUNDANT  2
#define  FLOP_IMPLEMENT  3

typedef double(*flop_count_func_t)(size_t M, size_t N, size_t K);

static flop_count_func_t flop_counters[]={flop_count_dgemm_simple,
                                          flop_count_dgemm_minimum,
                                          flop_count_dgemm_redundant,
                                          flop_count_dgemm_implement};

static void usage(){
	printf("usage :\n");
	printf("  speed_test [-n <max_size>] [-i <begin_size>] [-d <incliment_size>] [-f {simple|minimum|redundant|implement}] [-h]\n\n");
	printf("  loop controll:\n");
	printf("     -n <max_size>       : max testing size of square matrix (default=2048)\n");
	printf("     -i <begin_size>     : initial testing size of square matrix (default=16)\n");
	printf("     -d <incliment_size> : inliment testing size of square matrix (default:x2)\n");
	printf("\n");
	printf("  flop count type:\n");
	printf("     simple    = 2*M*N*K\n");
	printf("     minimum   = M*N*(2*K+3)  (default)\n");
	printf("     redandant = M*N*(3*K+1)\n");
	printf("     implement = M*N*(2*K+K/32+1)\n");
	printf("\n");
}

int main(int argc, char** argv){

	int error = 0;

	flops_info_t cpu;

	double *A, *B, *C;
	dgemm_test_t myblas = {myblas_dgemm, CblasColMajor, CblasNoTrans, CblasNoTrans, MAX_SIZE, MAX_SIZE, MAX_SIZE, 1e0, NULL, MAX_SIZE, NULL, MAX_SIZE, 1e0, NULL, MAX_SIZE };

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
	while((c=getopt(argc,argv,"f:n:i:d:h")) != -1 ){
		switch(c){
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

	//A = calloc( nmax*nmax, sizeof(double) );
	//B = calloc( nmax*nmax, sizeof(double) );
	//C = calloc( nmax*nmax, sizeof(double) );
	A = aligned_alloc( ALIGNMENT_B, nmax*nmax*sizeof(double) );
	B = aligned_alloc( ALIGNMENT_B, nmax*nmax*sizeof(double) );
	C = aligned_alloc( ALIGNMENT_B, nmax*nmax*sizeof(double) );

	myblas.A = A;
	myblas.B = B;
	myblas.C = C;

	peak_flops( &cpu );
	double mpeak = cpu.mflops_double_max  / cpu.num_cores;
	double bpeak = cpu.mflops_double_base / cpu.num_cores;
	flop_count_func_t flop_count = flop_counters[flop_type];

	printf("Max  Peak MFlops per Core: %G MFlops \n",mpeak);
	printf("Base Peak MFlops per Core: %G MFlops \n",bpeak);

	printf("size  , elapsed time[s],          MFlops,   base ratio[%%],    max ratio[%%] \n");
	size_t n = init;
	while ( n <= nmax ){

		myblas.M   = n;
		myblas.N   = n;
		myblas.K   = n;
		myblas.lda = n;
		myblas.ldb = n;
		myblas.ldc = n;

		double nflop = flop_count( myblas.M, myblas.N, myblas.K );
		double dt = check_speed( &myblas );
		double mflops = nflop / dt / 1000 / 1000;

		printf("%6u, %15G, %15G, %15G, %15G \n",n,dt,mflops,mflops/bpeak*100,mflops/mpeak*100);

		n = ( dist==0 ? n*2 : n+dist );

	}

	free(A);
	free(B);
	free(C);

	return error;
}
