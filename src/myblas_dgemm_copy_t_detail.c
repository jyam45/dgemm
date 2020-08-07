#include "myblas_internal.h"


void myblas_dgemm_copy_t_detail(size_t K1, size_t M1, const double* A, size_t k1, size_t i1, size_t lda, double* A2 ){

	double ymm0 ,ymm1 ,ymm2 ,ymm3 ;
	double ymm4 ,ymm5 ,ymm6 ,ymm7 ;
	double ymm8 ,ymm9 ,ymm10,ymm11;
	double ymm12,ymm13,ymm14,ymm15;

	A = A + i1 + k1*lda;

	size_t m = M1;
	if( m >> 2 ){
	  size_t m4 = ( m >> 2 );
	  while( m4-- ){

	    size_t k = K1;
	    if( k >> 3 ){
	      size_t k8 = ( k >> 3 );
	      while( k8-- ){

	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda); ymm8  = *(A + 0 + 2*lda); ymm12 = *(A + 0 + 3*lda);
	        ymm1  = *(A + 1 + 0*lda); ymm5  = *(A + 1 + 1*lda); ymm9  = *(A + 1 + 2*lda); ymm13 = *(A + 1 + 3*lda);
	        ymm2  = *(A + 2 + 0*lda); ymm6  = *(A + 2 + 1*lda); ymm10 = *(A + 2 + 2*lda); ymm14 = *(A + 2 + 3*lda);
	        ymm3  = *(A + 3 + 0*lda); ymm7  = *(A + 3 + 1*lda); ymm11 = *(A + 3 + 2*lda); ymm15 = *(A + 3 + 3*lda);
	        *(A2 + 0*4 + 0) = ymm0 ; *(A2 + 1*4 + 0) = ymm4 ; *(A2 + 2*4 + 0) = ymm8 ; *(A2 + 3*4 + 0) = ymm12;
	        *(A2 + 0*4 + 1) = ymm1 ; *(A2 + 1*4 + 1) = ymm5 ; *(A2 + 2*4 + 1) = ymm9 ; *(A2 + 3*4 + 1) = ymm13;
	        *(A2 + 0*4 + 2) = ymm2 ; *(A2 + 1*4 + 2) = ymm6 ; *(A2 + 2*4 + 2) = ymm10; *(A2 + 3*4 + 2) = ymm14;
	        *(A2 + 0*4 + 3) = ymm3 ; *(A2 + 1*4 + 3) = ymm7 ; *(A2 + 2*4 + 3) = ymm11; *(A2 + 3*4 + 3) = ymm15;
	        ymm0  = *(A + 0 + 4*lda); ymm4  = *(A + 0 + 5*lda); ymm8  = *(A + 0 + 6*lda); ymm12 = *(A + 0 + 7*lda);
	        ymm1  = *(A + 1 + 4*lda); ymm5  = *(A + 1 + 5*lda); ymm9  = *(A + 1 + 6*lda); ymm13 = *(A + 1 + 7*lda);
	        ymm2  = *(A + 2 + 4*lda); ymm6  = *(A + 2 + 5*lda); ymm10 = *(A + 2 + 6*lda); ymm14 = *(A + 2 + 7*lda);
	        ymm3  = *(A + 3 + 4*lda); ymm7  = *(A + 3 + 5*lda); ymm11 = *(A + 3 + 6*lda); ymm15 = *(A + 3 + 7*lda);
	        *(A2 + 4*4 + 0) = ymm0 ; *(A2 + 5*4 + 0) = ymm4 ; *(A2 + 6*4 + 0) = ymm8 ; *(A2 + 7*4 + 0) = ymm12;
	        *(A2 + 4*4 + 1) = ymm1 ; *(A2 + 5*4 + 1) = ymm5 ; *(A2 + 6*4 + 1) = ymm9 ; *(A2 + 7*4 + 1) = ymm13;
	        *(A2 + 4*4 + 2) = ymm2 ; *(A2 + 5*4 + 2) = ymm6 ; *(A2 + 6*4 + 2) = ymm10; *(A2 + 7*4 + 2) = ymm14;
	        *(A2 + 4*4 + 3) = ymm3 ; *(A2 + 5*4 + 3) = ymm7 ; *(A2 + 6*4 + 3) = ymm11; *(A2 + 7*4 + 3) = ymm15;
	        A += 8*lda ;
	        A2+= 32;;

	      }
	    }
	    if( k & 4 ){
	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda); ymm8  = *(A + 0 + 2*lda); ymm12 = *(A + 0 + 3*lda);
	        ymm1  = *(A + 1 + 0*lda); ymm5  = *(A + 1 + 1*lda); ymm9  = *(A + 1 + 2*lda); ymm13 = *(A + 1 + 3*lda);
	        ymm2  = *(A + 2 + 0*lda); ymm6  = *(A + 2 + 1*lda); ymm10 = *(A + 2 + 2*lda); ymm14 = *(A + 2 + 3*lda);
	        ymm3  = *(A + 3 + 0*lda); ymm7  = *(A + 3 + 1*lda); ymm11 = *(A + 3 + 2*lda); ymm15 = *(A + 3 + 3*lda);
	        *(A2 + 0*4 + 0) = ymm0 ; *(A2 + 1*4 + 0) = ymm4 ; *(A2 + 2*4 + 0) = ymm8 ; *(A2 + 3*4 + 0) = ymm12;
	        *(A2 + 0*4 + 1) = ymm1 ; *(A2 + 1*4 + 1) = ymm5 ; *(A2 + 2*4 + 1) = ymm9 ; *(A2 + 3*4 + 1) = ymm13;
	        *(A2 + 0*4 + 2) = ymm2 ; *(A2 + 1*4 + 2) = ymm6 ; *(A2 + 2*4 + 2) = ymm10; *(A2 + 3*4 + 2) = ymm14;
	        *(A2 + 0*4 + 3) = ymm3 ; *(A2 + 1*4 + 3) = ymm7 ; *(A2 + 2*4 + 3) = ymm11; *(A2 + 3*4 + 3) = ymm15;
	        A += 4*lda ;
	        A2+= 16;;
	    }
	    if( k & 2 ){
	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda);
	        ymm1  = *(A + 1 + 0*lda); ymm5  = *(A + 1 + 1*lda);
	        ymm2  = *(A + 2 + 0*lda); ymm6  = *(A + 2 + 1*lda);
	        ymm3  = *(A + 3 + 0*lda); ymm7  = *(A + 3 + 1*lda);
	        *(A2 + 0*4 + 0) = ymm0 ; *(A2 + 1*4 + 0) = ymm4 ;
	        *(A2 + 0*4 + 1) = ymm1 ; *(A2 + 1*4 + 1) = ymm5 ;
	        *(A2 + 0*4 + 2) = ymm2 ; *(A2 + 1*4 + 2) = ymm6 ;
	        *(A2 + 0*4 + 3) = ymm3 ; *(A2 + 1*4 + 3) = ymm7 ;
	        A += 2*lda ;
	        A2+= 8;
	    }
	    if( k & 1 ){
	        ymm0  = *(A + 0 + 0*lda);
	        ymm1  = *(A + 1 + 0*lda);
	        ymm2  = *(A + 2 + 0*lda);
	        ymm3  = *(A + 3 + 0*lda);
	        *(A2 + 0*4 + 0) = ymm0 ;
	        *(A2 + 0*4 + 1) = ymm1 ;
	        *(A2 + 0*4 + 2) = ymm2 ;
	        *(A2 + 0*4 + 3) = ymm3 ;
	        A += 1*lda ;
	        A2+= 4;
	    }
	    A  = A  - lda *K1 + 4;

	  }
	}
	if( m & 2 ){

	    size_t k = K1;
	    if( k >> 3 ){
	      size_t k8 = ( k >> 3 );
	      while( k8-- ){

	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda); ymm8  = *(A + 0 + 2*lda); ymm12 = *(A + 0 + 3*lda);
	        ymm1  = *(A + 1 + 0*lda); ymm5  = *(A + 1 + 1*lda); ymm9  = *(A + 1 + 2*lda); ymm13 = *(A + 1 + 3*lda);
	        *(A2 + 0*2 + 0) = ymm0 ; *(A2 + 1*2 + 0) = ymm4 ; *(A2 + 2*2 + 0) = ymm8 ; *(A2 + 3*2 + 0) = ymm12;
	        *(A2 + 0*2 + 1) = ymm1 ; *(A2 + 1*2 + 1) = ymm5 ; *(A2 + 2*2 + 1) = ymm9 ; *(A2 + 3*2 + 1) = ymm13;
	        ymm0  = *(A + 0 + 4*lda); ymm4  = *(A + 0 + 5*lda); ymm8  = *(A + 0 + 6*lda); ymm12 = *(A + 0 + 7*lda);
	        ymm1  = *(A + 1 + 4*lda); ymm5  = *(A + 1 + 5*lda); ymm9  = *(A + 1 + 6*lda); ymm13 = *(A + 1 + 7*lda);
	        *(A2 + 4*2 + 0) = ymm0 ; *(A2 + 5*2 + 0) = ymm4 ; *(A2 + 6*2 + 0) = ymm8 ; *(A2 + 7*2 + 0) = ymm12;
	        *(A2 + 4*2 + 1) = ymm1 ; *(A2 + 5*2 + 1) = ymm5 ; *(A2 + 6*2 + 1) = ymm9 ; *(A2 + 7*2 + 1) = ymm13;
	        A += 8*lda ;
	        A2+= 16;;

	      }
	    }
	    if( k & 4 ){
	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda); ymm8  = *(A + 0 + 2*lda); ymm12 = *(A + 0 + 3*lda);
	        ymm1  = *(A + 1 + 0*lda); ymm5  = *(A + 1 + 1*lda); ymm9  = *(A + 1 + 2*lda); ymm13 = *(A + 1 + 3*lda);
	        *(A2 + 0*2 + 0) = ymm0 ; *(A2 + 1*2 + 0) = ymm4 ; *(A2 + 2*2 + 0) = ymm8 ; *(A2 + 3*2 + 0) = ymm12;
	        *(A2 + 0*2 + 1) = ymm1 ; *(A2 + 1*2 + 1) = ymm5 ; *(A2 + 2*2 + 1) = ymm9 ; *(A2 + 3*2 + 1) = ymm13;
	        A += 4*lda ;
	        A2+= 8;;
	    }
	    if( k & 2 ){
	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda);
	        ymm1  = *(A + 1 + 0*lda); ymm5  = *(A + 1 + 1*lda);
	        *(A2 + 0*2 + 0) = ymm0 ; *(A2 + 1*2 + 0) = ymm4 ;
	        *(A2 + 0*2 + 1) = ymm1 ; *(A2 + 1*2 + 1) = ymm5 ;
	        A += 2*lda ;
	        A2+= 4;
	    }
	    if( k & 1 ){
	        ymm0  = *(A + 0 + 0*lda);
	        ymm1  = *(A + 1 + 0*lda);
	        *(A2 + 0*2 + 0) = ymm0 ;
	        *(A2 + 0*2 + 1) = ymm1 ;
	        A += 1*lda ;
	        A2+= 2;
	    }
	    A  = A  - lda *K1 + 2;

	}
	if( m & 1 ){

	    size_t k = K1;
	    if( k >> 3 ){
	      size_t k8 = ( k >> 3 );
	      while( k8-- ){

	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda); ymm8  = *(A + 0 + 2*lda); ymm12 = *(A + 0 + 3*lda);
	        *(A2 + 0*1 + 0) = ymm0 ; *(A2 + 1*1 + 0) = ymm4 ; *(A2 + 2*1 + 0) = ymm8 ; *(A2 + 3*1 + 0) = ymm12;
	        ymm0  = *(A + 0 + 4*lda); ymm4  = *(A + 0 + 5*lda); ymm8  = *(A + 0 + 6*lda); ymm12 = *(A + 0 + 7*lda);
	        *(A2 + 4*1 + 0) = ymm0 ; *(A2 + 5*1 + 0) = ymm4 ; *(A2 + 6*1 + 0) = ymm8 ; *(A2 + 7*1 + 0) = ymm12;
	        A += 8*lda ;
	        A2+= 8;;

	      }
	    }
	    if( k & 4 ){
	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda); ymm8  = *(A + 0 + 2*lda); ymm12 = *(A + 0 + 3*lda);
	        *(A2 + 0*1 + 0) = ymm0 ; *(A2 + 1*1 + 0) = ymm4 ; *(A2 + 2*1 + 0) = ymm8 ; *(A2 + 3*1 + 0) = ymm12;
	        A += 4*lda ;
	        A2+= 4;;
	    }
	    if( k & 2 ){
	        ymm0  = *(A + 0 + 0*lda); ymm4  = *(A + 0 + 1*lda);
	        *(A2 + 0*1 + 0) = ymm0 ; *(A2 + 1*1 + 0) = ymm4 ;
	        A += 2*lda ;
	        A2+= 2;
	    }
	    if( k & 1 ){
	        ymm0  = *(A + 0 + 0*lda);
	        *(A2 + 0*1 + 0) = ymm0 ;
	        A += 1*lda ;
	        A2+= 1;
	    }
	    A  = A  - lda *K1 + 1;

	}


}

