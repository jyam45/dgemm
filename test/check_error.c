#include "dgemm_test.h"
#include <stdio.h>

int check_error( const dgemm_test_t* t1, const dgemm_test_t* t2 ){

	double* C1 = t1->C;
	double* C2 = t2->C;
	int ldc1 = t1->ldc;
	int ldc2 = t2->ldc;

	if( t1->M != t2->M ){
		fprintf(stderr,"[ERROR] M is not equal : t1.M=%d, t2.M=%d \n",t1->M,t2->M);
	}

	if( t1->N != t2->N ){
		fprintf(stderr,"[ERROR] N is not equal : t1.N=%d, t2.N=%d \n",t1->N,t2->N);
	}	

	int m = t1->M;
	int n = t1->N;
	int k = t1->K;

	int error = 0;
	if( t1->Order == CblasRowMajor && t2->Order == CblasRowMajor ){

		error = check_matrix( m, n, k, C1, 1, ldc1, C2, 1, ldc2 );

	}else if( t1->Order == CblasColMajor && t2->Order == CblasRowMajor ){

		error = check_matrix( m, n, k, C1, ldc1, 1, C2, 1, ldc2 );

	}else if( t1->Order == CblasRowMajor && t2->Order == CblasColMajor ){

		error = check_matrix( m, n, k, C1, 1, ldc1, C2, ldc2, 1 );

	}else if( t1->Order == CblasColMajor && t2->Order == CblasColMajor ){

		error = check_matrix( m, n, k, C1, ldc1, 1, C2, ldc2, 1 );

	}

	return error;
}

