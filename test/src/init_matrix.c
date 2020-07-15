#include "dgemm_test.h"

void init_matrix( const int m, const int n,
                  double* A, const int di, const int dj, double value )
{
	double *a = A;
	for( int j=0; j<n; j++ ){
		for( int i=0; i<m; i++ ){
			*a = value;
			a = a + di;
		}
		a = a - m*di + dj;
	}
}

void rand_matrix( const int m, const int n,
                  double* A, const int di, const int dj, uint64_t seed )
{
	unsigned short xseed[3]={0};
	if( seed != 0 ){
		xseed[0] = (seed    )&0xffff;
		xseed[1] = (seed>>16)&0xffff;
		xseed[2] = (seed>>32)&0xffff;
		seed48(xseed);
	}

	double *a = A;
	for( int j=0; j<n; j++ ){
		for( int i=0; i<m; i++ ){
			*a = drand48();
			a = a + di;
		}
		a = a - m*di + dj;
	}
}


