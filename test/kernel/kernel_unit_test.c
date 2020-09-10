#include "kernel_test.h"
#include "myblas_internal.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define MAX_SIZE 1023
#define TILE_M     32 
#define TILE_N     32
#define TILE_K     32
#define SEED 13892393

#define F_CORE  0x0001

static kernel_func_t myblas_funcs[] = { myblas_dgemm_kernel, myblas_dgemm_kernel_core };
static kernel_func_t basic_funcs[]  = { myblas_basic_kernel, myblas_basic_kernel_core };

static void do_kernel( kernel_test_t *test ){

	test->func( test->alpha, test->A, test->B, test->C, test->ldc, test->info );

}

static int kernel_check_error( const kernel_test_t *test1, const kernel_test_t *test2 ){

	return check_matrix( test1->info->M2, test1->info->N2, test1->info->K2, test1->C, 1, test1->ldc, test2->C, 1, test2->ldc ); 

}

int main( int argc, char** argv ){

	block3d_info_t sizes={MAX_SIZE,MAX_SIZE,MAX_SIZE,TILE_M,TILE_N,TILE_K};
	kernel_test_t test1 ={NULL,0e0,NULL,NULL,NULL,MAX_SIZE,&sizes};
	kernel_test_t test2 ={NULL,0e0,NULL,NULL,NULL,MAX_SIZE,&sizes};

	size_t M=MAX_SIZE;
	size_t N=MAX_SIZE;
	size_t K=MAX_SIZE;

	double alpha=1.382934e0;

	char*  argend;

	opterr = 0;
	int flags = 0;
	int c;
	while((c=getopt(argc,argv,"m:n:k:a:c")) != -1 ){
		switch(c){
		case 'm' : M=strtol(optarg,&argend,10); if( *argend != '\0' ){ return -1; }; break;
		case 'n' : N=strtol(optarg,&argend,10); if( *argend != '\0' ){ return -1; }; break;
		case 'k' : K=strtol(optarg,&argend,10); if( *argend != '\0' ){ return -1; }; break;
		case 'a' : alpha=strtod(optarg,&argend); if( *argend != '\0' ){ return -1; }; break;
		case 'c' : flags |= F_CORE; break;
		default  :
			printf("Unknown option : %c\n",c);
			return -1;
		}
	}


	test1.func = basic_funcs[flags];
	test2.func = myblas_funcs[flags];

	sizes.M2 = M;
	sizes.N2 = N;
	sizes.K2 = K;

	double* A = calloc( M*K, sizeof(double));
	double* B = calloc( K*N, sizeof(double));
	double* C = calloc( M*N, sizeof(double));
	double* D = calloc( M*N, sizeof(double));

	rand_matrix( M, K, A, 1, M, SEED ); // ColMajor
	rand_matrix( K, N, B, 1, K, SEED ); // ColMajor
	init_matrix( M, N, C, 1, M, 0e0  ); // ColMajor
	init_matrix( M, N, D, 1, M, 0e0  ); // ColMajor

	test1.A    = A;
	test1.B    = B;
	test1.C    = C;
	test1.ldc  = M;
	test1.info = &sizes;
	test1.alpha= alpha;

	test2.A    = A;
	test2.B    = B;
	test2.C    = D;
	test2.ldc  = M;
	test2.info = &sizes;
	test2.alpha= alpha;

	do_kernel( &test1 );
	do_kernel( &test2 );

	int error = kernel_check_error( &test1, &test2 );

	//if( error ){ printf("NG\n"); }else{ printf("OK\n"); }
	if( error ){
	    printf("NG\n");
	    //break ;
	    if( M < 16 && N < 16 ){
	      print_matrix( M, N, test1.C, test1.ldc );
	      print_matrix( M, N, test2.C, test2.ldc );
	    }
	}else{
	    printf("OK\n"); 
	}


	free(A);
	free(B);
	free(C);
	free(D);

	return error;
}


