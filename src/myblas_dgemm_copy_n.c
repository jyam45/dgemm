#include "myblas_internal.h"

void myblas_dgemm_copy_n(const double* B, size_t ldb, double* B2, const block2d_info_t* info ){

	size_t k2     = info->i2    ;
	size_t j2     = info->j2    ;
	size_t K      = info->M     ;
	size_t N      = info->N     ;
	size_t K2     = info->M2    ;
	size_t N2     = info->N2    ;
	size_t tile_K = info->tile_M;
	size_t tile_N = info->tile_N;

	// On L2-Cache Copy for B
	for( size_t k1=k2; k1<k2+K2; k1+=MIN(K-k1,MYBLAS_TILE_K ) ){
	  size_t K1 = MIN(MYBLAS_TILE_K ,K-k1);
	  for( size_t j1=j2; j1<j2+N2; j1+=MIN(N-j1,MYBLAS_TILE_N ) ){
	    size_t N1 = MIN(MYBLAS_TILE_N ,N-j1);
	    for( size_t j =j1; j < j1+N1; j++ ){
	      for( size_t k =k1; k < k1+K1; k++ ){
	        *B2=*B;
	        B++;
	        B2++;
	      }
	      B  = B  - K1 + ldb ;
	    }
	  }
	  B  = B  - ldb *N2 + K1;
	}
	B2 = B2 - K2*N2;
	B  = B  - K2;

}
