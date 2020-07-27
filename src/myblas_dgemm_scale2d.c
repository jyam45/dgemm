#include "myblas_internal.h"

void myblas_dgemm_scale2d(double beta, double *C, size_t ldc, const block2d_info_t* info ){

	size_t i2     = info->i2    ;
	size_t j2     = info->j2    ;
	size_t M      = info->M     ;
	size_t N      = info->N     ;
	size_t M2     = info->M2    ;
	size_t N2     = info->N2    ;
	size_t tile_M = info->tile_M;
	size_t tile_N = info->tile_N;

	 // scaling beta*C
	 for( size_t j=0; j<N; j++ ){
	     for( size_t i=0; i<M; i++ ){
	 	*C=beta*(*C);
	         C++;
	     }
	     C = C - M + ldc;
	 }
	 C = C - ldc*N; // retern to head of pointer.


}

