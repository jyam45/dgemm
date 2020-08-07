#include "myblas_internal.h"


void myblas_dgemm_copy_t_detail(size_t K1, size_t M1, const double* A, size_t k1, size_t i1, size_t lda, double* A2 ){

	double ymm0 ,ymm1 ,ymm2 ,ymm3 ;
	double ymm4 ,ymm5 ,ymm6 ,ymm7 ;
	double ymm8 ,ymm9 ,ymm10,ymm11;
	double ymm12,ymm13,ymm14,ymm15;

	A = A + lda*k1 + i1; // start point

	size_t m = M1;
	if( m >> 2 ){
	  size_t m4 = ( m >> 2 ); // unrolling
	  while( m4-- ){
	    size_t k = K1;
	    if( k >> 3 ){
	      size_t k8 = ( k >> 3 ); // unrolling
	      while( k8-- ){
	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda); ymm8  = *(A + 0 + 2*lda); ymm12 = *(A + 0 + 3*lda);
	        ymm1  = *(A + 1 + 0*lda); ymm5  = *(A + 1 + 1*lda); ymm9  = *(A + 1 + 2*lda); ymm13 = *(A + 1 + 3*lda);
	        ymm2  = *(A + 2 + 0*lda); ymm6  = *(A + 2 + 1*lda); ymm10 = *(A + 2 + 2*lda); ymm14 = *(A + 2 + 3*lda);
	        ymm3  = *(A + 3 + 0*lda); ymm7  = *(A + 3 + 1*lda); ymm11 = *(A + 3 + 2*lda); ymm15 = *(A + 3 + 3*lda);
	        *(A2 + 0 + 0*8) = ymm0 ; *(A2 + 1 + 0*8) = ymm4 ; *(A2 + 2 + 0*8) = ymm8 ; *(A2 + 3 + 0*8) = ymm12;
	        *(A2 + 0 + 1*8) = ymm1 ; *(A2 + 1 + 1*8) = ymm5 ; *(A2 + 2 + 1*8) = ymm9 ; *(A2 + 3 + 1*8) = ymm13;
	        *(A2 + 0 + 2*8) = ymm2 ; *(A2 + 1 + 2*8) = ymm6 ; *(A2 + 2 + 2*8) = ymm10; *(A2 + 3 + 2*8) = ymm14;
	        *(A2 + 0 + 3*8) = ymm3 ; *(A2 + 1 + 3*8) = ymm7 ; *(A2 + 2 + 3*8) = ymm11; *(A2 + 3 + 3*8) = ymm15;
	        ymm0  = *(A + 0 + 4*lda); ymm4  = *(A + 0 + 5*lda); ymm8  = *(A + 0 + 6*lda); ymm12 = *(A + 0 + 7*lda);
	        ymm1  = *(A + 1 + 4*lda); ymm5  = *(A + 1 + 5*lda); ymm9  = *(A + 1 + 6*lda); ymm13 = *(A + 1 + 7*lda);
	        ymm2  = *(A + 2 + 4*lda); ymm6  = *(A + 2 + 5*lda); ymm10 = *(A + 2 + 6*lda); ymm14 = *(A + 2 + 7*lda);
	        ymm3  = *(A + 3 + 4*lda); ymm7  = *(A + 3 + 5*lda); ymm11 = *(A + 3 + 6*lda); ymm15 = *(A + 3 + 7*lda);
	        *(A2 + 4 + 0*8) = ymm0 ; *(A2 + 5 + 0*8) = ymm4 ; *(A2 + 6 + 0*8) = ymm8 ; *(A2 + 7 + 0*8) = ymm12;
	        *(A2 + 4 + 1*8) = ymm1 ; *(A2 + 5 + 1*8) = ymm5 ; *(A2 + 6 + 1*8) = ymm9 ; *(A2 + 7 + 1*8) = ymm13;
	        *(A2 + 4 + 2*8) = ymm2 ; *(A2 + 5 + 2*8) = ymm6 ; *(A2 + 6 + 2*8) = ymm10; *(A2 + 7 + 2*8) = ymm14;
	        *(A2 + 4 + 3*8) = ymm3 ; *(A2 + 5 + 3*8) = ymm7 ; *(A2 + 6 + 3*8) = ymm11; *(A2 + 7 + 3*8) = ymm15;
	        A += 8*lda ;
	        A2+= 32;;
	      }
	    }
	    if( k & 4 ){
	    //if( k >> 2 ){
	    //  size_t k4 = ( k >> 2 ); // unrolling
	    //  while( k4-- ){
	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda); ymm8  = *(A + 0 + 2*lda); ymm12 = *(A + 0 + 3*lda);
	        ymm1  = *(A + 1 + 0*lda); ymm5  = *(A + 1 + 1*lda); ymm9  = *(A + 1 + 2*lda); ymm13 = *(A + 1 + 3*lda);
	        ymm2  = *(A + 2 + 0*lda); ymm6  = *(A + 2 + 1*lda); ymm10 = *(A + 2 + 2*lda); ymm14 = *(A + 2 + 3*lda);
	        ymm3  = *(A + 3 + 0*lda); ymm7  = *(A + 3 + 1*lda); ymm11 = *(A + 3 + 2*lda); ymm15 = *(A + 3 + 3*lda);
	        *(A2 + 0 + 0*4) = ymm0 ; *(A2 + 1 + 0*4) = ymm4 ; *(A2 + 2 + 0*4) = ymm8 ; *(A2 + 3 + 0*4) = ymm12;
	        *(A2 + 0 + 1*4) = ymm1 ; *(A2 + 1 + 1*4) = ymm5 ; *(A2 + 2 + 1*4) = ymm9 ; *(A2 + 3 + 1*4) = ymm13;
	        *(A2 + 0 + 2*4) = ymm2 ; *(A2 + 1 + 2*4) = ymm6 ; *(A2 + 2 + 2*4) = ymm10; *(A2 + 3 + 2*4) = ymm14;
	        *(A2 + 0 + 3*4) = ymm3 ; *(A2 + 1 + 3*4) = ymm7 ; *(A2 + 2 + 3*4) = ymm11; *(A2 + 3 + 3*4) = ymm15;
	        A += 4*lda ;
	        A2+= 16;;
	    //  }
	    }
	    if( k & 2 ){
	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda);
	        ymm1  = *(A + 1 + 0*lda); ymm5  = *(A + 1 + 1*lda);
	        ymm2  = *(A + 2 + 0*lda); ymm6  = *(A + 2 + 1*lda);
	        ymm3  = *(A + 3 + 0*lda); ymm7  = *(A + 3 + 1*lda);
	        *(A2 + 0 + 0*2) = ymm0 ; *(A2 + 1 + 0*2) = ymm4 ;
	        *(A2 + 0 + 1*2) = ymm1 ; *(A2 + 1 + 1*2) = ymm5 ;
	        *(A2 + 0 + 2*2) = ymm2 ; *(A2 + 1 + 2*2) = ymm6 ;
	        *(A2 + 0 + 3*2) = ymm3 ; *(A2 + 1 + 3*2) = ymm7 ;
	        A += 2*lda ;
	        A2+= 8;
	    }
	    if( k & 1 ){
	        ymm0  = *(A + 0 + 0*lda);
	        ymm1  = *(A + 1 + 0*lda);
	        ymm2  = *(A + 2 + 0*lda);
	        ymm3  = *(A + 3 + 0*lda);
	        *(A2 + 0 + 0*1) = ymm0 ;
	        *(A2 + 0 + 1*1) = ymm1 ;
	        *(A2 + 0 + 2*1) = ymm2 ;
	        *(A2 + 0 + 3*1) = ymm3 ;
	        A += lda ;
	        A2+= 4;
	    }
	    //if( k & 3 ){
	    //  size_t kr = ( k & 3 ); 
	    //  while( kr-- ){
	    //    ymm0  = *(A + 0 + 0*lda);
	    //    ymm1  = *(A + 1 + 0*lda);
	    //    ymm2  = *(A + 2 + 0*lda);
	    //    ymm3  = *(A + 3 + 0*lda);
	    //    *(A2 + 0 + 0*4) = ymm0 ;
	    //    *(A2 + 0 + 1*4) = ymm1 ;
	    //    *(A2 + 0 + 2*4) = ymm2 ;
	    //    *(A2 + 0 + 3*4) = ymm3 ;
	    //    //(*A2) = (*A);
	    //    A += lda ;
	    //    A2+=4;
	    //  }
	    //}
	    A  = A  - lda *K1 + 4;
	  }
	}
	if( m & 2 ){

	    size_t k = K1;
	    if( k >> 3 ){
	      size_t k8 = ( k >> 3 ); // unrolling
	      while( k8-- ){
	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda); ymm8  = *(A + 0 + 2*lda); ymm12 = *(A + 0 + 3*lda);
	        ymm1  = *(A + 1 + 0*lda); ymm5  = *(A + 1 + 1*lda); ymm9  = *(A + 1 + 2*lda); ymm13 = *(A + 1 + 3*lda);
	        *(A2 + 0 + 0*8) = ymm0 ; *(A2 + 1 + 0*8) = ymm4 ; *(A2 + 2 + 0*8) = ymm8 ; *(A2 + 3 + 0*8) = ymm12;
	        *(A2 + 0 + 1*8) = ymm1 ; *(A2 + 1 + 1*8) = ymm5 ; *(A2 + 2 + 1*8) = ymm9 ; *(A2 + 3 + 1*8) = ymm13;
	        ymm0  = *(A + 0 + 4*lda); ymm4  = *(A + 0 + 5*lda); ymm8  = *(A + 0 + 6*lda); ymm12 = *(A + 0 + 7*lda);
	        ymm1  = *(A + 1 + 4*lda); ymm5  = *(A + 1 + 5*lda); ymm9  = *(A + 1 + 6*lda); ymm13 = *(A + 1 + 7*lda);
	        *(A2 + 4 + 0*8) = ymm0 ; *(A2 + 5 + 0*8) = ymm4 ; *(A2 + 6 + 0*8) = ymm8 ; *(A2 + 7 + 0*8) = ymm12;
	        *(A2 + 4 + 1*8) = ymm1 ; *(A2 + 5 + 1*8) = ymm5 ; *(A2 + 6 + 1*8) = ymm9 ; *(A2 + 7 + 1*8) = ymm13;
	        A += 8*lda ;
	        A2+= 16;;
	      }
	    }
	    if( k & 4 ){
	    //if( k >> 2 ){
	    //  size_t k4 = ( k >> 2 ); // unrolling
	    //  while( k4-- ){
	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda); ymm8  = *(A + 0 + 2*lda); ymm12 = *(A + 0 + 3*lda);
	        ymm1  = *(A + 1 + 0*lda); ymm5  = *(A + 1 + 1*lda); ymm9  = *(A + 1 + 2*lda); ymm13 = *(A + 1 + 3*lda);
	        *(A2 + 0 + 0*4) = ymm0 ; *(A2 + 1 + 0*4) = ymm4 ; *(A2 + 2 + 0*4) = ymm8 ; *(A2 + 3 + 0*4) = ymm12;
	        *(A2 + 0 + 1*4) = ymm1 ; *(A2 + 1 + 1*4) = ymm5 ; *(A2 + 2 + 1*4) = ymm9 ; *(A2 + 3 + 1*4) = ymm13;
	        A += 4*lda ;
	        A2+= 8;;
	    //  }
	    }
	    if( k & 2 ){
	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda);
	        ymm1  = *(A + 1 + 0*lda); ymm5  = *(A + 1 + 1*lda);
	        *(A2 + 0 + 0*2) = ymm0 ; *(A2 + 1 + 0*2) = ymm4 ;
	        *(A2 + 0 + 1*2) = ymm1 ; *(A2 + 1 + 1*2) = ymm5 ;
	        A += 2*lda ;
	        A2+= 4;
	    }
	    if( k & 1 ){
	        ymm0  = *(A + 0 + 0*lda);
	        ymm1  = *(A + 1 + 0*lda);
	        *(A2 + 0 + 0*1) = ymm0 ;
	        *(A2 + 0 + 1*1) = ymm1 ;
	        A += lda ;
	        A2+= 2;
	    }

	    A  = A  - lda *K1 + 2;

	}
	if( m & 1 ){

	    size_t k = K1;
	    if( k >> 3 ){
	      size_t k8 = ( k >> 3 ); // unrolling
	      while( k8-- ){
	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda); ymm8  = *(A + 0 + 2*lda); ymm12 = *(A + 0 + 3*lda);
	        *(A2 + 0 + 0*8) = ymm0 ; *(A2 + 1 + 0*8) = ymm4 ; *(A2 + 2 + 0*8) = ymm8 ; *(A2 + 3 + 0*8) = ymm12;
	        ymm0  = *(A + 0 + 4*lda); ymm4  = *(A + 0 + 5*lda); ymm8  = *(A + 0 + 6*lda); ymm12 = *(A + 0 + 7*lda);
	        *(A2 + 4 + 0*8) = ymm0 ; *(A2 + 5 + 0*8) = ymm4 ; *(A2 + 6 + 0*8) = ymm8 ; *(A2 + 7 + 0*8) = ymm12;
	        A += 8*lda ;
	        A2+= 8;;
	      }
	    }
	    if( k & 4 ){
	    //if( k >> 2 ){
	    //  size_t k4 = ( k >> 2 ); // unrolling
	    //  while( k4-- ){
	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda); ymm8  = *(A + 0 + 2*lda); ymm12 = *(A + 0 + 3*lda);
	        *(A2 + 0 + 0*4) = ymm0 ; *(A2 + 1 + 0*4) = ymm4 ; *(A2 + 2 + 0*4) = ymm8 ; *(A2 + 3 + 0*4) = ymm12;
	        A += 4*lda ;
	        A2+= 4;;
	    //  }
	    }
	    if( k & 2 ){
	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda);
	        *(A2 + 0 + 0*2) = ymm0 ; *(A2 + 1 + 0*2) = ymm4 ;
	        A += 2*lda ;
	        A2+= 2;
	    }
	    if( k & 1 ){
	        ymm0  = *(A + 0 + 0*lda);
	        *(A2 + 0 + 0*1) = ymm0 ;
	        A += lda ;
	        A2+= 1;
	    }

	    A  = A  - lda *K1 + 1;

	}

	//if( m & 3 ){
	//  size_t mr = ( m & 3 );
	//  while( mr-- ){
	//    size_t k = K1;
	//    while( k-- ){
	//      (*A2) = (*A);
	//      A += lda ;
	//      A2++;
	//    }
	//    A  = A  - lda *K1 + 1;
	//  }
	//}

}

