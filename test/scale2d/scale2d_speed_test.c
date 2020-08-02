#include "scale2d_test.h"
#include "myblas_internal.h"
#include "Timer.h"
#include <stdlib.h>
#include <stdio.h>

#define MAX_SIZE 2048

static void do_scale2d( scale2d_test_t *test ){

	test->func( test->beta, test->C, test->ldc, test->info );

}

int main( int argc, char** argv ){

	block2d_info_t sizes={0,0,MAX_SIZE,MAX_SIZE,1,1};
	scale2d_test_t test ={myblas_dgemm_scale2d,0e0,NULL,MAX_SIZE,&sizes};

	double  beta = 1.3923842e0;
	double* C = calloc(MAX_SIZE*MAX_SIZE,sizeof(double));

	test.beta = beta;
	test.C    = C;
	test.ldc  = MAX_SIZE;

	int error = 0;
	printf("size  , elapsed time[s],   copy size[KB],          MFLOPS,            MB/s,          B/FLOP \n");
	for( size_t n=16; n <= MAX_SIZE; n*=2 ){
		test.info->M2 = n;
		test.info->N2 = n;

		init_matrix(test.info->M2, test.info->N2, test.C, test.ldc, 1, 1e0 );

		double t1 = get_realtime();
		do_scale2d( &test );
		double t2 = get_realtime();
		double dt = t2 - t1;
		double mflop = ((double)test.info->M2) *((double)test.info->N2) / (1024*1024);
		double bytes = test.info->M2 * test.info->N2 * sizeof(double);
		printf("%6u, %15G, %15G, %15G, %15G, %15G \n",n,dt,bytes/1024,mflop/dt,bytes/dt/(1024*1024),bytes/(mflop*1024*1024));
	}

	free(C);

	return error;
}

