#include "dgemm_test.h"
#include <stdio.h>
#include <float.h>
#include <math.h>

static int check_matrix( const int m, const int n, const int k,
                         const double *C1, const int di1, const int dj1,
                         const double *C2, const int di2, const int dj2 )
{
	for( int j=0; j<n; j++ ){
		for( int i = 0; i<m; i++ ){

			double c1 = fabs(*C1);
			double c2 = fabs(*C2);
			double cmax = ( c1>c2 ? c1 : c2 );
			double cdiff = fabs(c1-c2);
			double sqrtk = sqrt((double)k);
			double eps = DBL_EPSILON * sqrtk * cmax;

			if( cdiff > eps  ){
				fprintf(stderr,"[ERROR] An element C(%d,%d) is invalid : t1=%G, t2=%G, diff=%G, eps=%G\n",i,j,*C1,*C2,cdiff,eps);
				return 1;
			}
			C1 = C1 + di1;
			C2 = C2 + di2;
		}
		C1 = C1 - m*di1 + dj1;
		C2 = C2 - m*di2 + dj2;
	}

	return 0;
}

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

