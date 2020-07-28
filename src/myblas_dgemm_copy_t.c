#include "myblas_internal.h"

#define MIN(x,y)  (((x)<(y))?(x):(y))
void myblas_dgemm_copy_t(const double* A, size_t lda, double* A2, const block2d_info_t* info ){

	size_t i2     = info->i2    ;
	size_t k2     = info->j2    ;
	size_t M      = info->M     ;
	size_t K      = info->N     ;
	size_t M2     = info->M2    ;
	size_t K2     = info->N2    ;
	size_t tile_M = info->tile_M;
	size_t tile_K = info->tile_N;

	// On L2-Cache Copy for A
	for( size_t k1=k2; k1<k2+K2; k1+=MIN(K-k1,tile_K ) ){
	  size_t K1 = MIN(tile_K ,K-k1);
	  for( size_t i1=i2; i1<i2+M2; i1+=MIN(M-i1,tile_M )  ){
	    size_t M1 = MIN(tile_M ,M-i1);
	    for( size_t i =i1; i < i1+M1; i++ ){
	      for( size_t k =k1; k < k1+K1; k++ ){
	        (*A2) = (*A);
	        A += lda ;
	        A2++;
	      }
	      A  = A  - lda *K1 + 1;
	    }
	  }
	  A = A  - M2 + lda *K1;
	}
	A2 = A2  - M2*K2;
	A = A  - lda*K2;

}
