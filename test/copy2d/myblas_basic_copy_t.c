#include "copy2d_test.h"

// On L2-Cache Copy for A
void myblas_basic_copy_t(const double* A, size_t lda, double* A2, const block2d_info_t* info ){

	size_t k2     = info->i2    ;
	size_t i2     = info->j2    ;
	size_t K2     = info->M2    ;
	size_t M2     = info->N2    ;
	size_t tile_K = info->tile_M;
	size_t tile_M = info->tile_N;

	size_t MQ = M2/tile_M;
	size_t MR = M2%tile_M;
	size_t KQ = K2/tile_K;
	size_t KR = K2%tile_K;

	block2d_info_t tile = { 0, 0, 0, 0, 0, 0 };

	A = A + lda*k2 + i2; // start point

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

	    tile.M2 = K1; tile.N2 = M1;
	    myblas_basic_copy_t_core( A, lda, A2, &tile );

	    A  = A  + M1;
	    A2 = A2 + M1*K1;
	  }
	  A = A  - M2 + lda *K1;
	}

}


void myblas_basic_copy_t_core(const double* A, size_t lda, double* A2, const block2d_info_t* info ){

	size_t K1     = info->M2    ;
	size_t M1     = info->N2    ;

	size_t m = M1;
	while( m-- ){
	  size_t k = K1;
	  while( k-- ){
	    (*A2) = (*A);
	    A += lda ;
	    A2++;
	  }
	  A  = A  - lda *K1 + 1;
	}

}

