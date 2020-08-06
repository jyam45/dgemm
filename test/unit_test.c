#include "dgemm_test.h"
#include "myblas.h"
#include <stdio.h>
#include <unistd.h>

#define  SIZE 1023
#define  SEED 23091

int main(int argc, char** argv){

	enum CBLAS_ORDER     majors[2]={CblasRowMajor,CblasColMajor};
	enum CBLAS_TRANSPOSE transposes[2]={CblasNoTrans,CblasTrans};

	const char* cmajor[2]={"Row-major","Col-major"};
	const char* ctrans[2]={"NoTrans","Trans"};

	double *A, *B, *C, *D;
	dgemm_test_t cblas = { cblas_dgemm, CblasRowMajor, CblasNoTrans, CblasNoTrans, SIZE, SIZE, SIZE, 1e0, NULL, SIZE, NULL, SIZE, 1e0, NULL, SIZE };
	dgemm_test_t myblas= {myblas_dgemm, CblasRowMajor, CblasNoTrans, CblasNoTrans, SIZE, SIZE, SIZE, 1e0, NULL, SIZE, NULL, SIZE, 1e0, NULL, SIZE };

	size_t M=SIZE;
	size_t N=SIZE;
	size_t K=SIZE;

	size_t lda[2]={0,0};
	size_t ldb[2]={0,0};
	size_t ldc[2]={0,0};

	char*  argend;

	opterr = 0;
	int flags = 0;
	int c;
	while((c=getopt(argc,argv,"m:n:k:")) != -1 ){
		switch(c){
		case 'm' : M=strtol(optarg,&argend,10); if( *argend != '\0' ){ return -1; }; break;
		case 'n' : N=strtol(optarg,&argend,10); if( *argend != '\0' ){ return -1; }; break;
		case 'k' : K=strtol(optarg,&argend,10); if( *argend != '\0' ){ return -1; }; break;
		default  :
			printf("Unknown option : %c\n",c);
			return -1;
		}
	}

	A = calloc( M*K, sizeof(double) );
	B = calloc( K*N, sizeof(double) );
	C = calloc( M*N, sizeof(double) );
	D = calloc( M*N, sizeof(double) );

	rand_matrix( M, K, A, 1, K, SEED ); // ColMajor
	rand_matrix( K, N, B, 1, N, SEED ); // ColMajor

	cblas.A = A;
	cblas.B = B;
	cblas.C = C;
	cblas.M = M;
	cblas.N = N;
	cblas.K = K;

	myblas.A = A;
	myblas.B = B;
	myblas.C = D;
	myblas.M = M;
	myblas.N = N;
	myblas.K = K;

	lda[0]=K; // Row-major
	lda[1]=M; // Col-major
	ldb[0]=N; // Row-major
	ldb[1]=K; // Col-major
	ldc[0]=N; // Row-major
	ldc[1]=M; // Col-major

	int error = 0;

	for( int iorder=0; iorder<2; iorder++ ){
	    for( int itransa=0; itransa<2; itransa++ ){
	        for( int itransb=0; itransb<2; itransb++ ){
 
	            cblas.Order   = majors[iorder];
	            cblas.TransA  = transposes[itransa];
	            cblas.TransB  = transposes[itransb];
	            cblas.lda     = lda[iorder];
	            cblas.ldb     = ldb[iorder];
	            cblas.ldc     = ldc[iorder];

	            myblas.Order  = majors[iorder];
	            myblas.TransA = transposes[itransa];
	            myblas.TransB = transposes[itransb];
	            myblas.lda    = lda[iorder];
	            myblas.ldb    = ldb[iorder];
	            myblas.ldc    = ldc[iorder];

	            printf("Case : Order=%s, TransA=%s, TransB=%s ... ",cmajor[iorder],ctrans[itransa],ctrans[itransb]);

	            init_matrix( M, N, cblas.C , 1, N, 0e0 );
	            init_matrix( M, N, myblas.C, 1, N, 0e0 );

	            do_dgemm( &cblas  );
	            do_dgemm( &myblas );

	            error = check_error( &cblas, &myblas );

	            if( error ){
	                printf("NG\n");
			//break ;
	            }else{
	                printf("OK\n"); 
	            }

	        }
	    }
	}

	free(A);
	free(B);
	free(C);
	free(D);

	return error;
}
