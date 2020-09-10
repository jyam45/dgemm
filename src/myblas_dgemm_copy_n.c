#include "myblas_internal.h"
//#include <stdio.h>

static copy_detail_func_t copy_n_detail[]={ myblas_dgemm_copy_n_4x8, myblas_dgemm_copy_n_2x8 };

// On L2-Cache Copy for B
void myblas_dgemm_copy_n(const double* B, size_t ldb, double* B2, const block2d_info_t* info ){

	size_t k2     = info->i2    ;
	size_t j2     = info->j2    ;
	size_t K2     = info->M2    ;
	size_t N2     = info->N2    ;
	size_t tile_K = info->tile_M;
	size_t tile_N = info->tile_N;

	size_t NB = N2/tile_N;
	size_t NR = N2%tile_N;
	size_t KB = K2/tile_K;
	size_t KR = K2%tile_K;

	copy_detail_func_t myblas_dgemm_copy_n_detail = copy_n_detail[info->type];

	B = B + k2 + ldb*j2; // start point

	if( NR >  0 ){ NB++; }
	if( NR == 0 ){ NR = tile_N; }
	if( KR >  0 ){ KB++; }
	if( KR == 0 ){ KR = tile_K; }

	// L1-cache blocking
	size_t k1 = KB;
	while( k1-- ){
	  size_t K1 = tile_K; if( k1==0 ){ K1=KR; }
	  size_t n1 = NB;
	  while( n1-- ){
	    size_t N1 = tile_N; if( n1==0 ){ N1=NR; }

	    myblas_dgemm_copy_n_detail( K1, N1, B, 0, 0, ldb, B2 );

	    B = B + N1*ldb;
	    B2= B2+ N1*K1;
	  }
	  B = B - N2*ldb + K1;
	}

}


void myblas_dgemm_copy_n_core(const double* B, size_t ldb, double* B2, const block2d_info_t* info ){

	copy_detail_func_t myblas_dgemm_copy_n_detail = copy_n_detail[info->type];

	myblas_dgemm_copy_n_detail( info->M2, info->N2, B, info->i2, info->j2, ldb, B2 );

}


