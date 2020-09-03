#include "dgemm_test.h"
#include "myblas.h"
#include <stdio.h>
#include <unistd.h>

#define  SIZE 1023
#define  SEED 23091

typedef void(*init_matrix_t)( size_t M, size_t N, double* A, size_t lda );

static void rand_matrix_n( size_t M, size_t N, double* A, size_t lda ){
	rand_matrix( M, N, A, 1, lda, SEED );
	//print_matrix( M, N, A, lda );
}
static void rand_matrix_t( size_t M, size_t N, double* A, size_t lda ){
	rand_matrix( M, N, A, lda, 1, SEED );
	//print_matrix( N, M, A, lda );
}

static void zero_matrix_n( size_t M, size_t N, double* A, size_t lda ){
	init_matrix( M, N, A, 1, lda, 0e0 );
	//print_matrix( M, N, A, lda );
}
static void zero_matrix_t( size_t M, size_t N, double* A, size_t lda ){
	init_matrix( M, N, A, lda, 1, 0e0 );
	//print_matrix( M, N, A, lda );
}

static void print_matrix_r( size_t M, size_t N, double* A, size_t lda ){
	print_matrix( N, M, A, lda );
}
static void print_matrix_c( size_t M, size_t N, double* A, size_t lda ){
	print_matrix( M, N, A, lda );
}

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

	size_t lda[2][2]={{0,0},{0,0}};
	size_t ldb[2][2]={{0,0},{0,0}};
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

	//rand_matrix( M, K, A, 1, K, SEED ); // ColMajor
	//rand_matrix( K, N, B, 1, N, SEED ); // ColMajor

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

	//lda[0]=K; // Row-major
	//lda[1]=M; // Col-major
	//ldb[0]=N; // Row-major
	//ldb[1]=K; // Col-major

	lda[0][0]=K; // Row-major NoTrans B*A
	lda[0][1]=M; // Row-major Trans   B*T(A)
	lda[1][0]=M; // Col-major NoTrans A*B
	lda[1][1]=K; // Col-major Trans   T(A)*B

	ldb[0][0]=N; // Row-major NoTrans  B*A
	ldb[0][1]=K; // Row-major Trans    T(B)*A
	ldb[1][0]=K; // Col-major NoTrans  A*B
	ldb[1][1]=N; // col-major Trans    A*T(B)

	ldc[0]=N; // Row-major
	ldc[1]=M; // Col-major

	init_matrix_t init_A[2][2];
	init_matrix_t init_B[2][2];
	init_matrix_t init_C[2];
	init_matrix_t print_C[2];

	init_A[0][0] = rand_matrix_t; //Row-major NoTrans  B*A
	init_A[0][1] = rand_matrix_n; //Row-major Trans    B*T(A)
	init_A[1][0] = rand_matrix_n; //Col-major NoTrans  A*B
	init_A[1][1] = rand_matrix_t; //col-major Trans    T(A)*B

	init_B[0][0] = rand_matrix_t; //Row-major NoTrans  B*A
	init_B[0][1] = rand_matrix_n; //Row-major Trans    T(B)*A
	init_B[1][0] = rand_matrix_n; //Col-major NoTrans  A*B
	init_B[1][1] = rand_matrix_t; //col-major Trans    A*T(B)

	init_C[0] = zero_matrix_t; // Row-major
	init_C[1] = zero_matrix_n; // Col-major

	print_C[0] = print_matrix_r; // Row-major
	print_C[1] = print_matrix_c; // Col-major
	
	int error = 0;

	for( int iorder=0; iorder<2; iorder++ ){
	    for( int itransa=0; itransa<2; itransa++ ){
	        for( int itransb=0; itransb<2; itransb++ ){
	            
	            cblas.Order   = majors[iorder];
	            cblas.TransA  = transposes[itransa];
	            cblas.TransB  = transposes[itransb];
	            cblas.lda     = lda[iorder][itransa];
	            cblas.ldb     = ldb[iorder][itransb];
	            cblas.ldc     = ldc[iorder];

	            myblas.Order  = majors[iorder];
	            myblas.TransA = transposes[itransa];
	            myblas.TransB = transposes[itransb];
	            myblas.lda    = lda[iorder][itransa];
	            myblas.ldb    = ldb[iorder][itransb];
	            myblas.ldc    = ldc[iorder];

	            printf("Case : Order=%s, TransA=%s, TransB=%s ... ",cmajor[iorder],ctrans[itransa],ctrans[itransb]);

	            init_matrix_t init_matrix_A = init_A[iorder][itransa];
	            init_matrix_t init_matrix_B = init_B[iorder][itransb];
	            init_matrix_t init_matrix_C = init_C[iorder];

	            init_matrix_A( M, K, A, lda[iorder][itransa] );
	            init_matrix_B( K, N, B, ldb[iorder][itransb] );

	            init_matrix_C( M, N, cblas.C , ldc[iorder] );
	            init_matrix_C( M, N, myblas.C, ldc[iorder] );

	            do_dgemm( &cblas  );
	            do_dgemm( &myblas );

	            error = check_error( &cblas, &myblas );

	            if( error ){
	                printf("NG\n");
			//break ;
	                if( M < 16 && N < 16 ){
	                  init_matrix_t print_matrix_C = print_C[iorder];
	                  print_matrix_C( M, N, cblas.C , ldc[iorder] );
	                  print_matrix_C( M, N, myblas.C, ldc[iorder] );
	                }
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
