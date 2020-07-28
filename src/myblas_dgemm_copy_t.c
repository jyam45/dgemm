#include "myblas_internal.h"

void myblas_dgemm_copy_t(const double* A, size_t lda, double* A2, const block2d_info_t* info ){

	size_t M2     = info->M2    ;
	size_t K2     = info->N2    ;
	size_t tile_M = info->tile_M;
	size_t tile_K = info->tile_N;

	size_t MQ = M2/tile_M;
	size_t MR = M2%tile_M;
	size_t KQ = K2/tile_K;
	size_t KR = K2%tile_K;

	// On L2-Cache Copy for A
	size_t k2 = KQ;
	while( k2-- ){
	  size_t K1 = tile_K;
	  size_t m2 = MQ;
	  while( m2-- ){
	    size_t M1 = tile_M;
	    size_t m1 = M1;
	    while( m1-- ){
	      size_t k1 = K1;
	      while( k1-- ){
	        (*A2) = (*A);
	        A += lda ;
	        A2++;
	      }
	      A  = A  - lda *K1 + 1;
	    }
	  }{
	    size_t M1 = MR;
	    size_t m1 = M1;
	    while( m1-- ){
	      size_t k1 = K1;
	      while( k1-- ){
	        (*A2) = (*A);
	        A += lda ;
	        A2++;
	      }
	      A  = A  - lda *K1 + 1;
	    }
	  }
	  A = A  - M2 + lda *K1;
	}{
	  size_t K1 = KR;
	  size_t m2 = MQ;
	  while( m2-- ){
	    size_t M1 = tile_M;
	    size_t m1 = M1;
	    while( m1-- ){
	      size_t k1 = K1;
	      while( k1-- ){
	        (*A2) = (*A);
	        A += lda ;
	        A2++;
	      }
	      A  = A  - lda *K1 + 1;
	    }
	  }{
	    size_t M1 = MR;
	    size_t m1 = M1;
	    while( m1-- ){
	      size_t k1 = K1;
	      while( k1-- ){
	        (*A2) = (*A);
	        A += lda ;
	        A2++;
	      }
	      A  = A  - lda *K1 + 1;
	    }
	  }
	  A = A  - M2 + lda *K1;
	}

}
