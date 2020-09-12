#include "copy2d_test.h"
#include "myblas_internal.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define MAX_SIZE 1023
#define TILE_M     16 
#define TILE_N    128
#define SEED 13892393

#define F_TRANS 0x0001
#define F_CORE  0x0002
#define F_AXIS  0x0004

#define COPY_MK  0
#define COPY_NK  1

static copy2d_func_t myblas_funcs[] = { myblas_dgemm_copy_n, myblas_dgemm_copy_t, myblas_dgemm_copy_n_core, myblas_dgemm_copy_t_core };
static copy2d_func_t basic_funcs[]  = { myblas_basic_copy_n, myblas_basic_copy_t, myblas_basic_copy_n_core, myblas_basic_copy_t_core };

static void do_copy2d( copy2d_test_t *test ){

	test->func( test->A, test->lda, test->buf, test->info );

}

static int copy2d_check_error( const copy2d_test_t *test1, const copy2d_test_t *test2 ){

	return check_array( test1->info->M2*test1->info->N2, test1->buf, test2->buf ); 

}

static void print_2buffers( size_t n, const double* A, const double* B ){

	for( size_t i=0; i<n; i++ ){
		printf("%15d %15G %15G\n",i,*(A+i),*(B+i));
	}

}

static void usage(){
	printf("Usage :\n");
	printf("  copy2d_unit_test [-t] [-c] [-m <M-size>] [-n <N-size>] [-h]\n");
	printf("\n");
	printf("  -t          : turn on transposition switch\n");
	printf("  -c          : turn on no L1-cache blocking\n");
	printf("  -u          : turn on NK-axis switch (default=MK-axis)\n");
	printf("  -m <M-size> : set a size of first dimenstion of M-by-N matrix (default:%u)\n",MAX_SIZE);
	printf("  -n <N-size> : set a size of second dimenstion of M-by-N matrix (default:%u)\n",MAX_SIZE);
	printf("  -h          : show this\n");
	printf("\n");
}


int main( int argc, char** argv ){

	block2d_info_t sizes={0,0,MAX_SIZE,MAX_SIZE,TILE_M,TILE_N,COPY_MK};
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
	while((c=getopt(argc,argv,"tcum:n:h")) != -1 ){
		switch(c){
		case 't' : flags |= F_TRANS; break;
		case 'c' : flags |= F_CORE ; break;
		case 'u' : flags |= F_AXIS ; break;
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

	if( flags & F_AXIS ){
	  sizes.type = COPY_NK;
	}else{
	  sizes.type = COPY_MK;
	}

	test1.func = basic_funcs[flags&0x3];
	test1.A    = A;
	test1.lda  = lda;
	test1.buf  = A1;

	test2.func = myblas_funcs[flags&0x3];
	test2.A    = A;
	test2.lda  = lda;
	test2.buf  = A2;

	sizes.M2   = M;
	sizes.N2   = N;


	do_copy2d( &test1 );
	do_copy2d( &test2 );

	int error = copy2d_check_error( &test1, &test2 );

	if( error ){ printf("NG\n"); print_2buffers(M*N,A1,A2);  }else{ printf("OK\n"); }

	free(A);
	free(A1);
	free(A2);

	return error;
}


