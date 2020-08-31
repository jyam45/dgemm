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


static void usage(){
	printf("Usage :\n");
	printf("  copy2d_unit_test [-t] [-c] [-m <M-size>] [-n <N-size>] [-h]\n");
	printf("\n");
	printf("  -t          : test a transposing-copy function\n");
	printf("  -c          : test a function without L1 cache blocking\n");
	printf("  -m <M-size> : set a size of first dimenstion of M-by-N matrix (default:%u)\n",MAX_SIZE);
	printf("  -n <N-size> : set a size of second dimenstion of M-by-N matrix (default:%u)\n",MAX_SIZE);
	printf("  -h          : show this\n");
	printf("\n");
}


int main( int argc, char** argv ){

	block2d_info_t sizes={0,0,MAX_SIZE,MAX_SIZE,TILE_M,TILE_N};
	copy2d_test_t test1 ={NULL,NULL,MAX_SIZE,NULL,&sizes};
	copy2d_test_t test2 ={NULL,NULL,MAX_SIZE,NULL,&sizes};

	size_t lda=MAX_SIZE;
	size_t M=MAX_SIZE;
	size_t N=MAX_SIZE;

	// processing command option
	char*  argend;
	opterr = 0;
	int flags = 0;
	int c;
	while((c=getopt(argc,argv,"tcm:n:h")) != -1 ){
		switch(c){
		case 't' : flags |= F_TRANS; break;
		case 'c' : flags |= F_CORE ; break;
		case 'm' : M=strtol(optarg,&argend,10); if( *argend != '\0' ){ usage(); return -1; }; break;
		case 'n' : N=strtol(optarg,&argend,10); if( *argend != '\0' ){ usage(); return -1; }; break;
		case 'h' : usage(); return 0;
		default  :
			printf("Unknown option : %c\n",c);
			usage();
			return -1;
		}
	}


	double* A    = calloc(M*N,sizeof(double));
	double* A1   = calloc(M*N,sizeof(double));
	double* A2   = calloc(M*N,sizeof(double));

	if( flags & F_TRANS ){
	  lda = N;
	  rand_matrix(M,N,A,lda,1,SEED);
	}else{
	  lda = M;
	  rand_matrix(M,N,A,1,lda,SEED);
	}

	test1.func = basic_funcs[flags];
	test1.A    = A;
	test1.lda  = lda;
	test1.buf  = A1;

	test2.func = myblas_funcs[flags];
	test2.A    = A;
	test2.lda  = lda;
	test2.buf  = A2;

	sizes.M2   = M;
	sizes.N2   = N;


	do_copy2d( &test1 );
	do_copy2d( &test2 );

	int error = copy2d_check_error( &test1, &test2 );

	if( error ){ printf("NG\n"); }else{ printf("OK\n"); }

	free(A);
	free(A1);
	free(A2);

	return error;
}


