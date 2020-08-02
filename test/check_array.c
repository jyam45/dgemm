#include "dgemm_test.h"
#include <stdio.h>
#include <float.h>
#include <math.h>

int check_array( const size_t n, const double *a, const double *b )
{
	for( size_t j=0; j<n; j++ ){

		double c1 = fabs(a[j]);
		double c2 = fabs(b[j]);
		double cmax = ( c1>c2 ? c1 : c2 );
		double cdiff = fabs(c1-c2);
		double eps = DBL_EPSILON * cmax;

		if( cdiff > eps  ){
			fprintf(stderr,"[ERROR] An element a(%d) is invalid : t1=%G, t2=%G, diff=%G, eps=%G\n",j,a[j],b[j],cdiff,eps);
			return 1;
		}
	}

	return 0;
}


