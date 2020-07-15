#include "dgemm_test.h"
#include <stdio.h>

#define  SIZE 1024

int main(int argc, char** arv){

	double *A, *B, *C, *D;
	dgemm_test_t cblas = { cblas_dgemm, CblasRowMajor, CblasNoTrans, CblasNoTrans, SIZE, SIZE, SIZE, 1e0, NULL, SIZE, NULL, SIZE, 1e0, NULL, SIZE };
	dgemm_test_t myblas= { cblas_dgemm, CblasRowMajor, CblasNoTrans, CblasNoTrans, SIZE, SIZE, SIZE, 1e0, NULL, SIZE, NULL, SIZE, 1e0, NULL, SIZE };

	A = calloc( SIZE*SIZE, sizeof(double) );
	B = calloc( SIZE*SIZE, sizeof(double) );
	C = calloc( SIZE*SIZE, sizeof(double) );
	D = calloc( SIZE*SIZE, sizeof(double) );

	rand_matrix( SIZE, SIZE, A, 1, SIZE, 0 ); // RowMajor
	rand_matrix( SIZE, SIZE, B, 1, SIZE, 0 ); // RowMajor

	cblas.A = A;
	cblas.B = B;
	cblas.C = C;

	myblas.A = A;
	myblas.B = B;
	myblas.C = D;

	do_dgemm( &cblas  );
	do_dgemm( &myblas );

	int error = check_error( &cblas, &myblas );

	if( !error ) { printf("OK\n"); }

	free(A);
	free(B);
	free(C);
	free(D);

	return error;
}
