#include "copy2d_test.h"
#include "myblas_internal.h"
#include "Timer.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define MAX_SIZE 2048
#define TILE_M     32 
#define TILE_N     32

static void do_copy2d( copy2d_test_t *test ){

	test->func( test->A, test->lda, test->buf, test->info );

}

int main( int argc, char** argv ){

	block2d_info_t sizes={0,0,MAX_SIZE,MAX_SIZE,TILE_M,TILE_N};
	copy2d_test_t test ={myblas_dgemm_copy_n,NULL,MAX_SIZE,NULL,&sizes};

	opterr = 0;
	int c;
	while((c=getopt(argc,argv,"t")) != -1 ){
		if( c == 't' ){
			test.func = myblas_dgemm_copy_t;
		}else{
			printf("Unknown option : %c\n",c);
			return -1;
		}
	}

	double  beta = 1.3923842e0;
	double* A = calloc(MAX_SIZE*MAX_SIZE,sizeof(double));
	double* A2= calloc(MAX_SIZE*MAX_SIZE,sizeof(double));

	test.A    = A;
	test.lda  = MAX_SIZE;
	test.buf  = A2;

	int error = 0;

	printf("size  , elapsed time[s],   copy size[KB],            MB/s \n");
	for( size_t n=16; n <= MAX_SIZE; n*=2 ){
		test.info->M2 = n;
		test.info->N2 = n;

		init_matrix(test.info->M2, test.info->N2, A, test.lda, 1, 1e0 );

		double t1 = get_realtime();
		do_copy2d( &test );
		double t2 = get_realtime();
		double dt = t2 - t1;
		double bytes = test.info->M2 * test.info->N2 * sizeof(double);
		printf("%6u, %15G, %15G, %15G \n",n,dt,bytes/2014,bytes/dt/(1024*1024));
	}

	free(A);
	free(A2);

	return error;
}

