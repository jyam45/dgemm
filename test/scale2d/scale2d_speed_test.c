#include "scale2d_test.h"
#include "myblas_internal.h"
#include "Timer.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define MAX_SIZE 2048

#define ALIGNMENT_B 32 // AVX2

static void do_scale2d( scale2d_test_t *test ){

	test->func( test->beta, test->C, test->ldc, test->info );

}

static void usage(){
	printf("usage :\n");
	printf("  scale2d_speed_test [-n <max_size>] [-i <begin_size>] [-d <incliment_size>] [-h]\n\n");
	printf("  loop controll:\n");
	printf("     -n <max_size>       : max testing size of square matrix (default=2048)\n");
	printf("     -i <begin_size>     : initial testing size of square matrix (default=16)\n");
	printf("     -d <incliment_size> : inliment testing size of square matrix (default:x2)\n");
	printf("\n");
}


int main( int argc, char** argv ){

	block2d_info_t sizes={0,0,MAX_SIZE,MAX_SIZE,1,1};
	scale2d_test_t test ={myblas_dgemm_scale2d,0e0,NULL,MAX_SIZE,&sizes};

	// loop configuration
	size_t init = 16;
	size_t dist =  0;
	size_t nmax = MAX_SIZE;

	// processing command option
	char*  argend;
	opterr = 0;
	int flags = 0;
	int c;
	while((c=getopt(argc,argv,"n:i:d:h")) != -1 ){
		switch(c){
		case 'n' : nmax = strtol(optarg,&argend,10);if( *argend != '\0' ){ usage(); return -1; }; break;  
		case 'i' : init = strtol(optarg,&argend,10);if( *argend != '\0' ){ usage(); return -1; }; break;  
		case 'd' : dist = strtol(optarg,&argend,10);if( *argend != '\0' ){ usage(); return -1; }; break;  
		case 'h' : usage(); return 0; 
		default  :
			printf("Option Error : %c\n",c);
			usage();
			return -1;
		}
	}

	double  beta = 1.3923842e0;
	//double* C = calloc(nmax*nmax,sizeof(double));
	double* C = aligned_alloc(ALIGNMENT_B,nmax*nmax*sizeof(double));

	test.beta = beta;
	test.C    = C;
	test.ldc  = nmax;

	int error = 0;
	printf("size  , elapsed time[s],   copy size[KB],          MFLOPS,            MB/s,          B/FLOP \n");
	//for( size_t n=16; n <= MAX_SIZE; n*=2 ){
	size_t n = init;
	while ( n <= nmax ){
		test.info->M2 = n;
		test.info->N2 = n;

		init_matrix(test.info->M2, test.info->N2, test.C, 1, test.ldc, 1e0 );

		double t1 = get_realtime();
		do_scale2d( &test );
		double t2 = get_realtime();
		double dt = t2 - t1;
		double mflop = ((double)test.info->M2) *((double)test.info->N2) / (1024*1024);
		double bytes = test.info->M2 * test.info->N2 * sizeof(double);
		printf("%6u, %15G, %15G, %15G, %15G, %15G \n",n,dt,bytes/1024,mflop/dt,bytes/dt/(1024*1024),bytes/(mflop*1024*1024));
		n = ( dist==0 ? n*2 : n+dist );
	}

	free(C);

	return error;
}

