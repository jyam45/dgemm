#include "scale2d_test.h"
#include "myblas_internal.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define MAX_SIZE 1023
#define SEED 13892393

static void do_scale2d( scale2d_test_t *test ){

	test->func( test->beta, test->C, test->ldc, test->info );

}

static int scale2d_check_error( const scale2d_test_t *test1, const scale2d_test_t *test2 ){

	return check_matrix( test1->info->M2, test1->info->N2, 1,
	                     test1->C, test1->ldc, 1, test2->C, test2->ldc, 1 );

}

int main( int argc, char** argv ){

	block2d_info_t sizes={0,0,MAX_SIZE,MAX_SIZE,1,1};
	scale2d_test_t test1={myblas_basic_scale2d,0e0,NULL,MAX_SIZE,&sizes};
	scale2d_test_t test2={myblas_dgemm_scale2d,0e0,NULL,MAX_SIZE,&sizes};

	double  beta = 1.3923842e0;
	double* C1 = calloc(MAX_SIZE*MAX_SIZE,sizeof(double));
	double* C2 = calloc(MAX_SIZE*MAX_SIZE,sizeof(double));

	rand_matrix(MAX_SIZE,MAX_SIZE,C1,MAX_SIZE,1,SEED);

	C2 = memcpy(C2,C1,MAX_SIZE*MAX_SIZE*sizeof(double));	

	test1.beta = beta;
	test1.C    = C1;
	test1.ldc  = MAX_SIZE;

	test2.beta = beta;
	test2.C    = C2;
	test2.ldc  = MAX_SIZE;

	do_scale2d( &test1 );
	do_scale2d( &test2 );

	int error = scale2d_check_error( &test1, &test2 );

	if( error ){ printf("NG\n"); }else{ printf("OK\n"); }

	free(C1);
	free(C2);

	return error;
}


