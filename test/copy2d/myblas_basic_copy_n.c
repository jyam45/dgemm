#include "copy2d_test.h"

// On L2-Cache Copy for B
void myblas_basic_copy_n(const double* B, size_t ldb, double* B2, const block2d_info_t* info ){

	size_t k2     = info->i2    ;
	size_t j2     = info->j2    ;
	size_t K2     = info->M2    ;
	size_t N2     = info->N2    ;
	size_t tile_K = info->tile_M;
	size_t tile_N = info->tile_N;

	size_t NQ = N2/tile_N;
	size_t NR = N2%tile_N;
	size_t KQ = K2/tile_K;
	size_t KR = K2%tile_K;

	block2d_info_t tile = { 0, 0, 0, 0, 0, 0 };

	B = B + k2 + ldb*j2; // start point

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

	    tile.M2 = K1; tile.N2 = N1;
	    myblas_basic_copy_n_core( B, ldb, B2, &tile );

	    B  = B  + N1*ldb;
	    B2 = B2 + N1*K1;

	  }
	  B  = B  - ldb *N2 + K1;
	}

}

void myblas_basic_copy_n_core(const double* B, size_t ldb, double* B2, const block2d_info_t* info ){

	size_t K1     = info->M2    ;
	size_t N1     = info->N2    ;

	size_t n = N1;
	while( n-- ){
	  size_t k = K1;
	  while( k-- ){
	    *B2=*B;
	    B++;
	    B2++;
	  }
	  B  = B  - K1 + ldb ;
	}

}