#include "myblas_internal.h"
#include <stdio.h>

void myblas_dgemm_kernel(double alpha, const double *A2, const double *B2, 
                         double *C, size_t ldc, const block3d_info_t* info ){

	size_t M2     = info->M2    ;
	size_t N2     = info->N2    ;
	size_t K2     = info->K2    ;
	size_t tile_M = info->tile_M;
	size_t tile_N = info->tile_N;
	size_t tile_K = info->tile_K;

	size_t MQ = M2/tile_M;
	size_t MR = M2%tile_M;
	size_t NQ = N2/tile_N;
	size_t NR = N2%tile_N;
	size_t KQ = K2/tile_K;
	size_t KR = K2%tile_K;

	if( MR >  0 ){ MQ++; }
	if( MR == 0 ){ MR = tile_M; }
	if( NR >  0 ){ NQ++; }
	if( NR == 0 ){ NR = tile_N; }
	if( KR >  0 ){ KQ++; }
	if( KR == 0 ){ KR = tile_K; }

	// L1-cache blocking
	size_t k1 = KQ;
	while( k1-- ){
	  size_t K1 = tile_K; if( k1==0 ){ K1=KR; }

	  size_t n1 = NQ;
	  while( n1-- ){
	    size_t N1 = tile_N; if( n1==0 ){ N1=NR; }

	    size_t m1 = MQ;
	    while( m1-- ){
	      size_t M1 = tile_M; if( m1==0 ){ M1=MR; }

	      myblas_dgemm_kernel_detail( M1, N1, K1, alpha, A2, B2, C, ldc );
	
              A2 = A2 + M1*K1;
	      C  = C  + M1;

	    } // end of m2-loop
	    A2 = A2 - M2*K1;
	    B2 = B2 + K1*N1;
	    C  = C - M2 + ldc*N1;

	  } // end of n2-loop
	  A2 = A2 + M2*K1;
	  C  = C - ldc*N2;

	} // end of k2-loop
	A2 = A2 - M2*K2;
	B2 = B2 - K2*N2;

}


void myblas_dgemm_kernel_core(double alpha, const double *A2, const double *B2, 
                              double *C, size_t ldc, const block3d_info_t* info ){

	myblas_dgemm_kernel_detail( info->M2, info->N2, info->K2, alpha, A2, B2, C, ldc );

}

