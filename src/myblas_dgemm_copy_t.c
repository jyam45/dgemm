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

	    block2d_info_t tile = { 0, 0, K1, M1 };
	    myblas_dgemm_copy_t_core( A, lda, A2, &tile );

	    A  = A  + M1;
	    A2 = A2 + M1*K1;

	  }
	  A = A  - M2 + lda *K1;
	}

}


void myblas_dgemm_copy_t_core(const double* A, size_t lda, double* A2, const block2d_info_t* info ){

	size_t k1     = info->i2    ;
	size_t i1     = info->j2    ;
	size_t K1     = info->M2    ;
	size_t M1     = info->N2    ;

	double ymm0 ,ymm1 ,ymm2 ,ymm3 ;
	double ymm4 ,ymm5 ,ymm6 ,ymm7 ;
	double ymm8 ,ymm9 ,ymm10,ymm11;
	double ymm12,ymm13,ymm14,ymm15;

	A = A + lda*k1 + i1; // start point

	size_t m = M1;
	if( m >> 3 ){//1
	  size_t m8 = ( m >> 3 ); // unrolling
	  while( m8-- ){//2
	    size_t k = K1;

	    if( k >> 2 ){//3
	      size_t k4 = ( k >> 2 ); // unrolling
	      while( k4-- ){//4
	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda); ymm8  = *(A + 0 + 2*lda); ymm12 = *(A + 0 + 3*lda);
	        ymm1  = *(A + 1 + 0*lda); ymm5  = *(A + 1 + 1*lda); ymm9  = *(A + 1 + 2*lda); ymm13 = *(A + 1 + 3*lda);
	        ymm2  = *(A + 2 + 0*lda); ymm6  = *(A + 2 + 1*lda); ymm10 = *(A + 2 + 2*lda); ymm14 = *(A + 2 + 3*lda);
	        ymm3  = *(A + 3 + 0*lda); ymm7  = *(A + 3 + 1*lda); ymm11 = *(A + 3 + 2*lda); ymm15 = *(A + 3 + 3*lda);
	        *(A2 + 0 + 0*K1) = ymm0 ; *(A2 + 1 + 0*K1) = ymm4 ; *(A2 + 2 + 0*K1) = ymm8 ; *(A2 + 3 + 0*K1) = ymm12;
	        *(A2 + 0 + 1*K1) = ymm1 ; *(A2 + 1 + 1*K1) = ymm5 ; *(A2 + 2 + 1*K1) = ymm9 ; *(A2 + 3 + 1*K1) = ymm13;
	        *(A2 + 0 + 2*K1) = ymm2 ; *(A2 + 1 + 2*K1) = ymm6 ; *(A2 + 2 + 2*K1) = ymm10; *(A2 + 3 + 2*K1) = ymm14;
	        *(A2 + 0 + 3*K1) = ymm3 ; *(A2 + 1 + 3*K1) = ymm7 ; *(A2 + 2 + 3*K1) = ymm11; *(A2 + 3 + 3*K1) = ymm15;
	        ymm0  = *(A + 4 + 0*lda); ymm4  = *(A + 4 + 1*lda); ymm8  = *(A + 4 + 2*lda); ymm12 = *(A + 4 + 3*lda);
	        ymm1  = *(A + 5 + 0*lda); ymm5  = *(A + 5 + 1*lda); ymm9  = *(A + 5 + 2*lda); ymm13 = *(A + 5 + 3*lda);
	        ymm2  = *(A + 6 + 0*lda); ymm6  = *(A + 6 + 1*lda); ymm10 = *(A + 6 + 2*lda); ymm14 = *(A + 6 + 3*lda);
	        ymm3  = *(A + 7 + 0*lda); ymm7  = *(A + 7 + 1*lda); ymm11 = *(A + 7 + 2*lda); ymm15 = *(A + 7 + 3*lda);
	        *(A2 + 0 + 4*K1) = ymm0 ; *(A2 + 1 + 4*K1) = ymm4 ; *(A2 + 2 + 4*K1) = ymm8 ; *(A2 + 3 + 4*K1) = ymm12;
	        *(A2 + 0 + 5*K1) = ymm1 ; *(A2 + 1 + 5*K1) = ymm5 ; *(A2 + 2 + 5*K1) = ymm9 ; *(A2 + 3 + 5*K1) = ymm13;
	        *(A2 + 0 + 6*K1) = ymm2 ; *(A2 + 1 + 6*K1) = ymm6 ; *(A2 + 2 + 6*K1) = ymm10; *(A2 + 3 + 6*K1) = ymm14;
	        *(A2 + 0 + 7*K1) = ymm3 ; *(A2 + 1 + 7*K1) = ymm7 ; *(A2 + 2 + 7*K1) = ymm11; *(A2 + 3 + 7*K1) = ymm15;
	        A += 4*lda ;
	        A2+= 4;;
	      }//4
	    }//3
	    if( k & 3 ){//5
	      size_t kr = ( k & 3 ); 
	      while( kr-- ){//6
	        ymm0  = *(A + 0 + 0*lda);
	        ymm1  = *(A + 1 + 0*lda);
	        ymm2  = *(A + 2 + 0*lda);
	        ymm3  = *(A + 3 + 0*lda);
	        *(A2 + 0 + 0*K1) = ymm0 ;
	        *(A2 + 0 + 1*K1) = ymm1 ;
	        *(A2 + 0 + 2*K1) = ymm2 ;
	        *(A2 + 0 + 3*K1) = ymm3 ;
	        ymm0  = *(A + 4 + 0*lda);
	        ymm1  = *(A + 5 + 0*lda);
	        ymm2  = *(A + 6 + 0*lda);
	        ymm3  = *(A + 7 + 0*lda);
	        *(A2 + 0 + 4*K1) = ymm0 ;
	        *(A2 + 0 + 5*K1) = ymm1 ;
	        *(A2 + 0 + 6*K1) = ymm2 ;
	        *(A2 + 0 + 7*K1) = ymm3 ;
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
	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda); ymm8  = *(A + 0 + 2*lda); ymm12 = *(A + 0 + 3*lda);
	        ymm1  = *(A + 1 + 0*lda); ymm5  = *(A + 1 + 1*lda); ymm9  = *(A + 1 + 2*lda); ymm13 = *(A + 1 + 3*lda);
	        ymm2  = *(A + 2 + 0*lda); ymm6  = *(A + 2 + 1*lda); ymm10 = *(A + 2 + 2*lda); ymm14 = *(A + 2 + 3*lda);
	        ymm3  = *(A + 3 + 0*lda); ymm7  = *(A + 3 + 1*lda); ymm11 = *(A + 3 + 2*lda); ymm15 = *(A + 3 + 3*lda);
	        *(A2 + 0 + 0*K1) = ymm0 ; *(A2 + 1 + 0*K1) = ymm4 ; *(A2 + 2 + 0*K1) = ymm8 ; *(A2 + 3 + 0*K1) = ymm12;
	        *(A2 + 0 + 1*K1) = ymm1 ; *(A2 + 1 + 1*K1) = ymm5 ; *(A2 + 2 + 1*K1) = ymm9 ; *(A2 + 3 + 1*K1) = ymm13;
	        *(A2 + 0 + 2*K1) = ymm2 ; *(A2 + 1 + 2*K1) = ymm6 ; *(A2 + 2 + 2*K1) = ymm10; *(A2 + 3 + 2*K1) = ymm14;
	        *(A2 + 0 + 3*K1) = ymm3 ; *(A2 + 1 + 3*K1) = ymm7 ; *(A2 + 2 + 3*K1) = ymm11; *(A2 + 3 + 3*K1) = ymm15;
	        A += 4*lda ;
	        A2+= 4;;
	      }
	    }
	    if( k & 3 ){
	      size_t kr = ( k & 3 ); 
	      while( kr-- ){
	        ymm0  = *(A + 0 + 0*lda);
	        ymm1  = *(A + 1 + 0*lda);
	        ymm2  = *(A + 2 + 0*lda);
	        ymm3  = *(A + 3 + 0*lda);
	        *(A2 + 0 + 0*K1) = ymm0 ;
	        *(A2 + 0 + 1*K1) = ymm1 ;
	        *(A2 + 0 + 2*K1) = ymm2 ;
	        *(A2 + 0 + 3*K1) = ymm3 ;
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
