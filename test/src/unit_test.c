#include "dgemm_test.h"
#include <stdio.h>

#define  SIZE 1023
#define  SEED 23091

int main(int argc, char** arv){

	enum CBLAS_ORDER     majors[2]={CblasRowMajor,CblasColMajor};
	enum CBLAS_TRANSPOSE transposes[2]={CblasNoTrans,CblasTrans};

	const char* cmajor[2]={"Row-major","Col-major"};
	const char* ctrans[2]={"NoTrans","Trans"};

	double *A, *B, *C, *D;
	dgemm_test_t cblas = { cblas_dgemm, CblasRowMajor, CblasNoTrans, CblasNoTrans, SIZE, SIZE, SIZE, 1e0, NULL, SIZE, NULL, SIZE, 1e0, NULL, SIZE };
	dgemm_test_t myblas= { cblas_dgemm, CblasRowMajor, CblasNoTrans, CblasNoTrans, SIZE, SIZE, SIZE, 1e0, NULL, SIZE, NULL, SIZE, 1e0, NULL, SIZE };

	A = calloc( SIZE*SIZE, sizeof(double) );
	B = calloc( SIZE*SIZE, sizeof(double) );
	C = calloc( SIZE*SIZE, sizeof(double) );
	D = calloc( SIZE*SIZE, sizeof(double) );

	rand_matrix( SIZE, SIZE, A, 1, SIZE, SEED ); // RowMajor
	rand_matrix( SIZE, SIZE, B, 1, SIZE, SEED ); // RowMajor

	cblas.A = A;
	cblas.B = B;
	cblas.C = C;

	myblas.A = A;
	myblas.B = B;
	myblas.C = D;

	int error = 0;

	for( int iorder=0; iorder<2; iorder++ ){
	    for( int itransa=0; itransa<2; itransa++ ){
	        for( int itransb=0; itransb<2; itransb++ ){
 
	            cblas.Order   = majors[iorder];
	            cblas.TransA  = transposes[itransa];
	            cblas.TransB  = transposes[itransb];

	            myblas.Order  = majors[iorder];
	            myblas.TransA = transposes[itransa];
	            myblas.TransB = transposes[itransb];

	            printf("Case : Order=%s, TransA=%s, TransB=%s ... ",cmajor[iorder],ctrans[itransa],ctrans[itransb]);

	            init_matrix( SIZE, SIZE, cblas.C , 1, SIZE, 0e0 );
	            init_matrix( SIZE, SIZE, myblas.C, 1, SIZE, 0e0 );

	            do_dgemm( &cblas  );
	            do_dgemm( &myblas );

	            error = check_error( &cblas, &myblas );

	            if( error ){
	                printf("NG\n");
			break ;
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
