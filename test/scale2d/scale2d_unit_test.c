#include "scale2d_test.h"
#include "myblas_internal.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define MAX_SIZE 1023
#define SEED 13892393
#define BETA 1.3923842e0

static void do_scale2d( scale2d_test_t *test ){

	test->func( test->beta, test->C, test->ldc, test->info );

}

static int scale2d_check_error( const scale2d_test_t *test1, const scale2d_test_t *test2 ){

	return check_matrix( test1->info->M2, test1->info->N2, 1,
	                     test1->C, 1, test1->ldc, test2->C, 1, test2->ldc );

}

static void usage(){
	printf("Usage :\n");
	printf("  scale2d_unit_test [-b <beta>] [-m <M-size>] [-n <N-size>] [-h]\n");
	printf("\n");
	printf("  -b <beta>   : set the beta coefficient (default:%G)\n",BETA);
	printf("  -m <M-size> : set a size of first dimenstion of M-by-N matrix (default:%u)\n",MAX_SIZE);
	printf("  -n <N-size> : set a size of second dimenstion of M-by-N matrix (default:%u)\n",MAX_SIZE);
	printf("  -h          : show this\n");
	printf("\n");
}

int main( int argc, char** argv ){

	block2d_info_t sizes={0,0,MAX_SIZE,MAX_SIZE,1,1};
	scale2d_test_t test1={myblas_basic_scale2d,0e0,NULL,MAX_SIZE,&sizes};
	scale2d_test_t test2={myblas_dgemm_scale2d,0e0,NULL,MAX_SIZE,&sizes};

	size_t M=MAX_SIZE;
	size_t N=MAX_SIZE;
	double beta = BETA;

	// processing command option
	char*  argend;
	opterr = 0;
	int flags = 0;
	int c;
	while((c=getopt(argc,argv,"b:m:n:h")) != -1 ){
		switch(c){
		case 'b' : beta=strtod(optarg,&argend); if( *argend != '\0' ){ printf("Conversion Error : a character '%c'\n",*argend); usage(); return -1; };
		case 'm' : M=strtol(optarg,&argend,10); if( *argend != '\0' ){ usage(); return -1; }; break;
		case 'n' : N=strtol(optarg,&argend,10); if( *argend != '\0' ){ usage(); return -1; }; break;
		case 'h' : usage(); return 0;
		default  :
			printf("Unknown option : %c\n",c);
			usage();
			return -1;
		}
	}

	double* C1 = calloc(M*N,sizeof(double));
	double* C2 = calloc(M*N,sizeof(double));

	rand_matrix(M,N,C1,1,M,SEED);

	C2 = memcpy(C2,C1,M*N*sizeof(double));	

	//print_matrix(M,N,C1,M);
	//print_matrix(M,N,C2,M);

	test1.beta = beta;
	test1.C    = C1;
	test1.ldc  = M;

	test2.beta = beta;
	test2.C    = C2;
	test2.ldc  = M;

	sizes.M2   = M;
	sizes.N2   = N;

	do_scale2d( &test1 );
	do_scale2d( &test2 );

	//print_matrix(M,N,C1,M);
	//print_matrix(M,N,C2,M);

	int error = scale2d_check_error( &test1, &test2 );

	if( error ){ printf("NG\n"); }else{ printf("OK\n"); }

	free(C1);
	free(C2);

	return error;
}


