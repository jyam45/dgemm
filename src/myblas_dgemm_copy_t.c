#include "myblas_internal.h"

// On L2-Cache Copy for A
void myblas_dgemm_copy_t(const double* A, size_t lda, double* A2, const block2d_info_t* info ){

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

	double ymm0 ,ymm1 ,ymm2 ,ymm3 ;
	double ymm4 ,ymm5 ,ymm6 ,ymm7 ;
	double ymm8 ,ymm9 ,ymm10,ymm11;
	double ymm12,ymm13,ymm14,ymm15;

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
	    size_t m = M1;

	    if( m >> 3 ){//1
	      size_t m8 = ( m >> 3 ); // unrolling
	      while( m8-- ){//2
	        size_t k = K1;

	        if( k >> 2 ){//3
	          size_t k4 = ( k >> 2 ); // unrolling
	          while( k4-- ){//4
	            ymm0  = *(A + 0*lda + 0);
	            ymm1  = *(A + 0*lda + 1);
	            ymm2  = *(A + 0*lda + 2);
	            ymm3  = *(A + 0*lda + 3);
	            *(A2+0*K1 + 0) = ymm0 ;
	            *(A2+1*K1 + 0) = ymm1 ;
	            *(A2+2*K1 + 0) = ymm2 ;
	            *(A2+3*K1 + 0) = ymm3 ;
	            ymm4  = *(A + 1*lda + 0);
	            ymm5  = *(A + 1*lda + 1);
	            ymm6  = *(A + 1*lda + 2);
	            ymm7  = *(A + 1*lda + 3);
	            *(A2+0*K1 + 1) = ymm4 ;
	            *(A2+1*K1 + 1) = ymm5 ;
	            *(A2+2*K1 + 1) = ymm6 ;
	            *(A2+3*K1 + 1) = ymm7 ;
	            ymm8  = *(A + 2*lda + 0);
	            ymm9  = *(A + 2*lda + 1);
	            ymm10 = *(A + 2*lda + 2);
	            ymm11 = *(A + 2*lda + 3);
	            *(A2+0*K1 + 2) = ymm8 ;
	            *(A2+1*K1 + 2) = ymm9 ;
	            *(A2+2*K1 + 2) = ymm10;
	            *(A2+3*K1 + 2) = ymm11;
	            ymm12 = *(A + 3*lda + 0);
	            ymm13 = *(A + 3*lda + 1);
	            ymm14 = *(A + 3*lda + 2);
	            ymm15 = *(A + 3*lda + 3);
	            *(A2+0*K1 + 3) = ymm12;
	            *(A2+1*K1 + 3) = ymm13;
	            *(A2+2*K1 + 3) = ymm14;
	            *(A2+3*K1 + 3) = ymm15;

	            ymm0  = *(A + 0*lda + 4);
	            ymm1  = *(A + 0*lda + 5);
	            ymm2  = *(A + 0*lda + 6);
	            ymm3  = *(A + 0*lda + 7);
	            *(A2+4*K1 + 0) = ymm0 ;
	            *(A2+5*K1 + 0) = ymm1 ;
	            *(A2+6*K1 + 0) = ymm2 ;
	            *(A2+7*K1 + 0) = ymm3 ;
	            ymm4  = *(A + 1*lda + 4);
	            ymm5  = *(A + 1*lda + 5);
	            ymm6  = *(A + 1*lda + 6);
	            ymm7  = *(A + 1*lda + 7);
	            *(A2+4*K1 + 1) = ymm4 ;
	            *(A2+5*K1 + 1) = ymm5 ;
	            *(A2+6*K1 + 1) = ymm6 ;
	            *(A2+7*K1 + 1) = ymm7 ;
	            ymm8  = *(A + 2*lda + 4);
	            ymm9  = *(A + 2*lda + 5);
	            ymm10 = *(A + 2*lda + 6);
	            ymm11 = *(A + 2*lda + 7);
	            *(A2+4*K1 + 2) = ymm8 ;
	            *(A2+5*K1 + 2) = ymm9 ;
	            *(A2+6*K1 + 2) = ymm10;
	            *(A2+7*K1 + 2) = ymm11;
	            ymm12 = *(A + 3*lda + 4);
	            ymm13 = *(A + 3*lda + 5);
	            ymm14 = *(A + 3*lda + 6);
	            ymm15 = *(A + 3*lda + 7);
	            *(A2+4*K1 + 3) = ymm12;
	            *(A2+5*K1 + 3) = ymm13;
	            *(A2+6*K1 + 3) = ymm14;
	            *(A2+7*K1 + 3) = ymm15;

	            A += 4*lda ;
	            A2+= 4;;
	          }//4
	        }//3
	        if( k & 3 ){//5
	          size_t kr = ( k & 3 ); 
	          while( kr-- ){//6
	            ymm0  = *(A + 0);
	            ymm1  = *(A + 1);
	            ymm2  = *(A + 2);
	            ymm3  = *(A + 3);
	            *(A2+0*K1) = ymm0 ;
	            *(A2+1*K1) = ymm1 ;
	            *(A2+2*K1) = ymm2 ;
	            *(A2+3*K1) = ymm3 ;
	            ymm0  = *(A + 4);
	            ymm1  = *(A + 5);
	            ymm2  = *(A + 6);
	            ymm3  = *(A + 7);
	            *(A2+4*K1) = ymm0 ;
	            *(A2+5*K1) = ymm1 ;
	            *(A2+6*K1) = ymm2 ;
	            *(A2+7*K1) = ymm3 ;
	            //(*A2) = (*A);
	            A += lda ;
	            A2++;
	          }//6
	        }//5
	        A  = A  - lda *K1 + 8;
	        A2 = A2 - K1 + 8*K1;

	      }//2
	    }//1

	    //if( m >> 2 ){
	    //  size_t m4 = ( m >> 2 ); // unrolling
	    //  while( m4-- ){
	    if( m & 4 ){//7
	      {
	        size_t k = K1;

	        if( k >> 2 ){
	          size_t k4 = ( k >> 2 ); // unrolling
	          while( k4-- ){
	            ymm0  = *(A + 0*lda + 0);
	            ymm1  = *(A + 0*lda + 1);
	            ymm2  = *(A + 0*lda + 2);
	            ymm3  = *(A + 0*lda + 3);
	            *(A2+0*K1 + 0) = ymm0 ;
	            *(A2+1*K1 + 0) = ymm1 ;
	            *(A2+2*K1 + 0) = ymm2 ;
	            *(A2+3*K1 + 0) = ymm3 ;
	            ymm4  = *(A + 1*lda + 0);
	            ymm5  = *(A + 1*lda + 1);
	            ymm6  = *(A + 1*lda + 2);
	            ymm7  = *(A + 1*lda + 3);
	            *(A2+0*K1 + 1) = ymm4 ;
	            *(A2+1*K1 + 1) = ymm5 ;
	            *(A2+2*K1 + 1) = ymm6 ;
	            *(A2+3*K1 + 1) = ymm7 ;
	            ymm8  = *(A + 2*lda + 0);
	            ymm9  = *(A + 2*lda + 1);
	            ymm10 = *(A + 2*lda + 2);
	            ymm11 = *(A + 2*lda + 3);
	            *(A2+0*K1 + 2) = ymm8 ;
	            *(A2+1*K1 + 2) = ymm9 ;
	            *(A2+2*K1 + 2) = ymm10;
	            *(A2+3*K1 + 2) = ymm11;
	            ymm12 = *(A + 3*lda + 0);
	            ymm13 = *(A + 3*lda + 1);
	            ymm14 = *(A + 3*lda + 2);
	            ymm15 = *(A + 3*lda + 3);
	            *(A2+0*K1 + 3) = ymm12;
	            *(A2+1*K1 + 3) = ymm13;
	            *(A2+2*K1 + 3) = ymm14;
	            *(A2+3*K1 + 3) = ymm15;
	            A += 4*lda ;
	            A2+= 4;;
	          }
	        }
	        if( k & 3 ){
	          size_t kr = ( k & 3 ); 
	          while( kr-- ){
	            ymm0  = *(A + 0);
	            ymm1  = *(A + 1);
	            ymm2  = *(A + 2);
	            ymm3  = *(A + 3);
	            *(A2+0*K1) = ymm0 ;
	            *(A2+1*K1) = ymm1 ;
	            *(A2+2*K1) = ymm2 ;
	            *(A2+3*K1) = ymm3 ;
	            //(*A2) = (*A);
	            A += lda ;
	            A2++;
	          }
	        }
	        A  = A  - lda *K1 + 4;
	        A2 = A2 - K1 + 4*K1;

	      }
	    }
	    if( m & 3 ){
	      size_t mr = ( m & 3 );
	      while( mr-- ){
	        size_t k = K1;
	        while( k-- ){
	          (*A2) = (*A);
	          A += lda ;
	          A2++;
	        }
	        A  = A  - lda *K1 + 1;
	      }
	    }

	  }
	  A = A  - M2 + lda *K1;
	}

}
