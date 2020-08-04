#include "copy2d_test.h"
#include "myblas_internal.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define MAX_SIZE 1023
#define TILE_M     32 
#define TILE_N     32
#define SEED 13892393

#define F_TRANS 0x0001
#define F_CORE  0x0002

static copy2d_func_t myblas_funcs[] = { myblas_dgemm_copy_n, myblas_dgemm_copy_t, myblas_dgemm_copy_n_core, myblas_dgemm_copy_t_core };
static copy2d_func_t basic_funcs[]  = { myblas_basic_copy_n, myblas_basic_copy_t, myblas_basic_copy_n_core, myblas_basic_copy_t_core };

static void do_copy2d( copy2d_test_t *test ){

	test->func( test->A, test->lda, test->buf, test->info );

}

static int copy2d_check_error( const copy2d_test_t *test1, const copy2d_test_t *test2 ){

	return check_array( test1->info->M2*test1->info->N2, test1->buf, test2->buf ); 

}

int main( int argc, char** argv ){

	block2d_info_t sizes={0,0,MAX_SIZE,MAX_SIZE,TILE_M,TILE_N};
	copy2d_test_t test1 ={NULL,NULL,MAX_SIZE,NULL,&sizes};
	copy2d_test_t test2 ={NULL,NULL,MAX_SIZE,NULL,&sizes};

	opterr = 0;
	int flags = 0;
	int c;
	while((c=getopt(argc,argv,"tk")) != -1 ){
		if( c == 't' ){
			flags |= F_TRANS;
		}else if( c == 'k' ){
			flags |= F_CORE;
		}else{
			printf("Unknown option : %c\n",c);
			return -1;
		}
	}
	test1.func = basic_funcs[flags];
	test2.func = myblas_funcs[flags];


	double* A    = calloc(MAX_SIZE*MAX_SIZE,sizeof(double));
	double* A1   = calloc(MAX_SIZE*MAX_SIZE,sizeof(double));
	double* A2   = calloc(MAX_SIZE*MAX_SIZE,sizeof(double));

	rand_matrix(MAX_SIZE,MAX_SIZE,A,MAX_SIZE,1,SEED);

	test1.A    = A;
	test1.lda  = MAX_SIZE;
	test1.buf  = A1;

	test2.A    = A;
	test2.lda  = MAX_SIZE;
	test2.buf  = A2;

	do_copy2d( &test1 );
	do_copy2d( &test2 );

	int error = copy2d_check_error( &test1, &test2 );

	if( error ){ printf("NG\n"); }else{ printf("OK\n"); }

	free(A);
	free(A1);
	free(A2);

	return error;
}

