#include "dgemm_test.h"
#include <stdio.h>
#include <float.h>
#include <math.h>

int check_matrix( const int m, const int n, const int k,
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


