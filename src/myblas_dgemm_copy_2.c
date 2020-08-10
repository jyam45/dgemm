#include "myblas_internal.h"

void myblas_dgemm_copy_2(const double *A2, const block3d_info_t* info ){

	size_t M2     = info->M2    ;
	size_t K2     = info->K2    ;
	size_t tile_M = info->tile_M;
	size_t tile_K = info->tile_K;

	size_t MQ = M2/tile_M;
	size_t MR = M2%tile_M;
	size_t KQ = K2/tile_K;
	size_t KR = K2%tile_K;

	double* Abuf = info->buf;

	if( MR >  0 ){ MQ++; }
	if( MR == 0 ){ MR = tile_M; }
	if( KR >  0 ){ KQ++; }
	if( KR == 0 ){ KR = tile_K; }

	// L1-cache blocking
	size_t k1 = KQ;
	while( k1-- ){
	  size_t K1 = tile_K; if( k1==0 ){ K1=KR; }

	    size_t m1 = MQ;
	    while( m1-- ){
	      size_t M1 = tile_M; if( m1==0 ){ M1=MR; }

	      myblas_dgemm_copy_2_detail( M1, K1, A2, Abuf );
	
              A2 = A2 + M1*K1;
	      Abuf = Abuf + 2*M1*K1;

	    } // end of m2-loop

	} // end of k2-loop
	A2 = A2 - M2*K2;
	Abuf = Abuf - 2*M2*K2;


}

void myblas_dgemm_copy_2_core(const double *A2, const block3d_info_t* info ){

	myblas_dgemm_copy_2_detail( info->M2, info->K2, A2, info->buf );

}
