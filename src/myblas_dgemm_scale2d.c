#include "myblas_internal.h"

void myblas_dgemm_scale2d(double beta, double *C, size_t ldc, const block2d_info_t* info ){

	size_t M      = info->M2    ;
	size_t N      = info->N2    ;
	size_t tile_M = info->tile_M;
	size_t tile_N = info->tile_N;

	 // scaling beta*C
	 size_t n = N;
	 while( n-- ){
	     size_t m = M;
	     while( m-- ){
	 	*C=beta*(*C);
	         C++;
	     }
	     C = C - M + ldc;
	 }
	 C = C - ldc*N; // retern to head of pointer.


}

