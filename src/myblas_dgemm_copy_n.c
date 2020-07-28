#include "myblas_internal.h"

void myblas_dgemm_copy_n(const double* B, size_t ldb, double* B2, const block2d_info_t* info ){

	size_t K2     = info->M2    ;
	size_t N2     = info->N2    ;
	size_t tile_K = info->tile_M;
	size_t tile_N = info->tile_N;

	size_t NQ = N2/tile_N;
	size_t NR = N2%tile_N;
	size_t KQ = K2/tile_K;
	size_t KR = K2%tile_K;

	// On L2-Cache Copy for B
	size_t k2 = KQ;
	while( k2-- ){
	  size_t K1 = tile_K;
	  size_t n2 = NQ;
	  while( n2-- ){
	    size_t N1 = tile_N;
	    size_t n1 = N1;
	    while( n1-- ){
	      size_t k1 = K1;
	      while( k1-- ){
	        *B2=*B;
	        B++;
	        B2++;
	      }
	      B  = B  - K1 + ldb ;
	    }
	  }{
	    size_t N1 = NR;
	    size_t n1 = N1;
	    while( n1-- ){
	      size_t k1 = K1;
	      while( k1-- ){
	        *B2=*B;
	        B++;
	        B2++;
	      }
	      B  = B  - K1 + ldb ;
	    }

	  }
	  B  = B  - ldb *N2 + K1;
	}{
	  size_t K1 = KR;
	  size_t n2 = NQ;
	  while( n2-- ){
	    size_t N1 = tile_N;
	    size_t n1 = N1;
	    while( n1-- ){
	      size_t k1 = K1;
	      while( k1-- ){
	        *B2=*B;
	        B++;
	        B2++;
	      }
	      B  = B  - K1 + ldb ;
	    }
	  }{
	    size_t N1 = NR;
	    size_t n1 = N1;
	    while( n1-- ){
	      size_t k1 = K1;
	      while( k1-- ){
	        *B2=*B;
	        B++;
	        B2++;
	      }
	      B  = B  - K1 + ldb ;
	    }

	  }
	  B  = B  - ldb *N2 + K1;
	}

}
