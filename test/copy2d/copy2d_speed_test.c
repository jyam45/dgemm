#include "copy2d_test.h"
#include "myblas_internal.h"
#include "Timer.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define MAX_SIZE 2048
#define TILE_M     32 
#define TILE_N     32

#define ALIGNMENT_B  32 // AVX2

#define F_TRANS 0x0001
#define F_CORE  0x0002
#define F_AXIS  0x0004

#define COPY_MK  0
#define COPY_NK  1

static copy2d_func_t myblas_funcs[] = { myblas_dgemm_copy_n, myblas_dgemm_copy_t, myblas_dgemm_copy_n_core, myblas_dgemm_copy_t_core };

static void do_copy2d( copy2d_test_t *test ){

	test->func( test->A, test->lda, test->buf, test->info );

}

static void usage(){
	printf("usage :\n");
	printf("  copy2d_speed_test [-n <max_size>] [-i <begin_size>] [-d <incliment_size>] [-t] [-c] [-h]\n\n");
	printf("  loop controll:\n");
	printf("     -n <max_size>       : max testing size of square matrix (default=2048)\n");
	printf("     -i <begin_size>     : initial testing size of square matrix (default=16)\n");
	printf("     -d <incliment_size> : inliment testing size of square matrix (default:x2)\n");
	printf("\n");
	printf("  algorithm selection:\n");
	printf("     -t                  : Transforming algorithm\n");
	printf("     -c                  : Core algorithm without L1-cache blocking\n");
	printf("     -u                  : turn on NK-axis switch (default=MK-axis)\n");
	printf("\n");
}

int main( int argc, char** argv ){

	block2d_info_t sizes={0,0,MAX_SIZE,MAX_SIZE,TILE_M,TILE_N,COPY_MK};
	copy2d_test_t test ={NULL,NULL,MAX_SIZE,NULL,&sizes};

	// loop configuration
	size_t init = 16;
	size_t dist =  0;
	size_t nmax = MAX_SIZE;

	// processing command option
	char*  argend;
	opterr = 0;
	int flags = 0;
	int c;
	while((c=getopt(argc,argv,"n:i:d:tcuh")) != -1 ){
		switch(c){
		case 'n' : nmax = strtol(optarg,&argend,10);if( *argend != '\0' ){ usage(); return -1; }; break;  
		case 'i' : init = strtol(optarg,&argend,10);if( *argend != '\0' ){ usage(); return -1; }; break;  
		case 'd' : dist = strtol(optarg,&argend,10);if( *argend != '\0' ){ usage(); return -1; }; break;  
		case 't' : flags |= F_TRANS; break; 
		case 'c' : flags |= F_CORE; break;
		case 'u' : flags |= F_AXIS ; break;
		case 'h' : usage(); return 0; 
		default  :
			printf("Option Error : %c\n",c);
			usage();
			return -1;
		}
	}

	test.func = myblas_funcs[flags&0x3];

	double  beta = 1.3923842e0;
	//double* A = calloc(nmax*nmax,sizeof(double));
	//double* A2= calloc(nmax*nmax,sizeof(double));
	double* A = aligned_alloc(ALIGNMENT_B,nmax*nmax*sizeof(double));
	double* A2= aligned_alloc(ALIGNMENT_B,nmax*nmax*sizeof(double));

	if( flags & F_AXIS ){
	  sizes.type = COPY_NK;
	}else{
	  sizes.type = COPY_MK;
	}
	test.A    = A;
	test.lda  = nmax;
	test.buf  = A2;

	int error = 0;

	printf("size  , elapsed time[s],   copy size[KB],            MB/s \n");
	//for( size_t n=16; n <= MAX_SIZE; n*=2 ){
	//for( size_t n=16; n <= MAX_SIZE; n+=16 ){
	size_t n = init;
	while ( n <= nmax ){

		test.info->M2 = n;
		test.info->N2 = n;

		init_matrix(test.info->M2, test.info->N2, A, 1, test.lda, 1e0 );

		double t1 = get_realtime();
		do_copy2d( &test );
		double t2 = get_realtime();
		double dt = t2 - t1;
		double bytes = test.info->M2 * test.info->N2 * sizeof(double);
		printf("%6u, %15G, %15G, %15G \n",n,dt,bytes/1024,bytes/dt/(1024*1024));

		n = ( dist==0 ? n*2 : n+dist );
	}

	free(A);
	free(A2);

	return error;
}

