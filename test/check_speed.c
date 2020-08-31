#include "dgemm_test.h"
#include "Timer.h"
#include <stdio.h>
#include <float.h>
#include <math.h>

void do_dgemm( dgemm_test_t* test ){

	test->dgemm(test->Order, test->TransA, test->TransB, test->M, test->N, test->K,
	            test->alpha, test->A, test->lda, test->B, test->ldb,
	            test->beta , test->C, test->ldc);

}

double check_speed( dgemm_test_t* test ){

	if( test->Order == CblasColMajor ){
		init_matrix( test->M, test->K, test->A, 1, test->lda, 1e0 );
		init_matrix( test->K, test->N, test->B, 1, test->ldb, 1e0 );
		init_matrix( test->M, test->N, test->C, 1, test->ldc, 0e0 );
	}else{
		init_matrix( test->M, test->K, test->A, test->lda, 1, 1e0 );
		init_matrix( test->K, test->N, test->B, test->ldb, 1, 1e0 );
		init_matrix( test->M, test->N, test->C, test->ldc, 1, 0e0 );
	}

	double t1,t2;
	int error;

	t1 = get_realtime();
	do_dgemm( test );
	t2 = get_realtime();	

	return (t2-t1);
}

