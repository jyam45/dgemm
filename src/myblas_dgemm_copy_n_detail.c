#include "myblas_internal.h"

void myblas_dgemm_copy_n_detail(size_t K1, size_t N1, const double* B, size_t k, size_t j,  size_t ldb, double* B2 ){

	double ymm0 ,ymm1 ,ymm2 ,ymm3 ;
	double ymm4 ,ymm5 ,ymm6 ,ymm7 ;
	double ymm8 ,ymm9 ,ymm10,ymm11;
	double ymm12,ymm13,ymm14,ymm15;

	B = B + k + ldb*j; // start point

	size_t n = N1;
	if( n >> 2 ){
	  size_t n4 = ( n >> 2 );
	  while( n4-- ){

	    size_t k = K1;
	    if( k >> 3 ){
	      size_t k8 = ( k >> 3 );
	      while( k8-- ){
	          ymm0  = *(B+0+0*ldb); ymm4  = *(B+1+0*ldb); ymm8  = *(B+2+0*ldb); ymm12 = *(B+3+0*ldb);
	          ymm1  = *(B+0+1*ldb); ymm5  = *(B+1+1*ldb); ymm9  = *(B+2+1*ldb); ymm13 = *(B+3+1*ldb);
	          ymm2  = *(B+0+2*ldb); ymm6  = *(B+1+2*ldb); ymm10 = *(B+2+2*ldb); ymm14 = *(B+3+2*ldb);
	          ymm3  = *(B+0+3*ldb); ymm7  = *(B+1+3*ldb); ymm11 = *(B+2+3*ldb); ymm15 = *(B+3+3*ldb);
	          *(B2+0*4+0) = ymm0 ; *(B2+1*4+0) = ymm4 ; *(B2+2*4+0) = ymm8 ; *(B2+3*4+0) = ymm12;
	          *(B2+0*4+1) = ymm1 ; *(B2+1*4+1) = ymm5 ; *(B2+2*4+1) = ymm9 ; *(B2+3*4+1) = ymm13;
	          *(B2+0*4+2) = ymm2 ; *(B2+1*4+2) = ymm6 ; *(B2+2*4+2) = ymm10; *(B2+3*4+2) = ymm14;
	          *(B2+0*4+3) = ymm3 ; *(B2+1*4+3) = ymm7 ; *(B2+2*4+3) = ymm11; *(B2+3*4+3) = ymm15;
	          ymm0  = *(B+4+0*ldb); ymm4  = *(B+5+0*ldb); ymm8  = *(B+6+0*ldb); ymm12 = *(B+7+0*ldb);
	          ymm1  = *(B+4+1*ldb); ymm5  = *(B+5+1*ldb); ymm9  = *(B+6+1*ldb); ymm13 = *(B+7+1*ldb);
	          ymm2  = *(B+4+2*ldb); ymm6  = *(B+5+2*ldb); ymm10 = *(B+6+2*ldb); ymm14 = *(B+7+2*ldb);
	          ymm3  = *(B+4+3*ldb); ymm7  = *(B+5+3*ldb); ymm11 = *(B+6+3*ldb); ymm15 = *(B+7+3*ldb);
	          *(B2+4*4+0) = ymm0 ; *(B2+5*4+0) = ymm4 ; *(B2+6*4+0) = ymm8 ; *(B2+7*4+0) = ymm12;
	          *(B2+4*4+1) = ymm1 ; *(B2+5*4+1) = ymm5 ; *(B2+6*4+1) = ymm9 ; *(B2+7*4+1) = ymm13;
	          *(B2+4*4+2) = ymm2 ; *(B2+5*4+2) = ymm6 ; *(B2+6*4+2) = ymm10; *(B2+7*4+2) = ymm14;
	          *(B2+4*4+3) = ymm3 ; *(B2+5*4+3) = ymm7 ; *(B2+6*4+3) = ymm11; *(B2+7*4+3) = ymm15;
	          B+=8;
	          B2+=32;
	      }
	    }
	    if( k & 4  ){
	    //if( k >> 2 ){
	    //  size_t k4 = ( k >> 2 ); // unrolling with 8 elements
	    //  while( k4-- ){
	          ymm0  = *(B+0+0*ldb); ymm4  = *(B+1+0*ldb); ymm8  = *(B+2+0*ldb); ymm12 = *(B+3+0*ldb);
	          ymm1  = *(B+0+1*ldb); ymm5  = *(B+1+1*ldb); ymm9  = *(B+2+1*ldb); ymm13 = *(B+3+1*ldb);
	          ymm2  = *(B+0+2*ldb); ymm6  = *(B+1+2*ldb); ymm10 = *(B+2+2*ldb); ymm14 = *(B+3+2*ldb);
	          ymm3  = *(B+0+3*ldb); ymm7  = *(B+1+3*ldb); ymm11 = *(B+2+3*ldb); ymm15 = *(B+3+3*ldb);
	          *(B2+0*4+0) = ymm0 ; *(B2+1*4+0) = ymm4 ; *(B2+2*4+0) = ymm8 ; *(B2+3*4+0) = ymm12;
	          *(B2+0*4+1) = ymm1 ; *(B2+1*4+1) = ymm5 ; *(B2+2*4+1) = ymm9 ; *(B2+3*4+1) = ymm13;
	          *(B2+0*4+2) = ymm2 ; *(B2+1*4+2) = ymm6 ; *(B2+2*4+2) = ymm10; *(B2+3*4+2) = ymm14;
	          *(B2+0*4+3) = ymm3 ; *(B2+1*4+3) = ymm7 ; *(B2+2*4+3) = ymm11; *(B2+3*4+3) = ymm15;
	          B+=4;
	          B2+=16;
	    //  }
	    }
	    if( k & 2 ){
	          ymm0  = *(B+0+0*ldb); ymm4  = *(B+1+0*ldb);
	          ymm1  = *(B+0+1*ldb); ymm5  = *(B+1+1*ldb);
	          ymm2  = *(B+0+2*ldb); ymm6  = *(B+1+2*ldb);
	          ymm3  = *(B+0+3*ldb); ymm7  = *(B+1+3*ldb);
	          *(B2+0*4+0) = ymm0 ; *(B2+1*4+0) = ymm4 ;
	          *(B2+0*4+1) = ymm1 ; *(B2+1*4+1) = ymm5 ;
	          *(B2+0*4+2) = ymm2 ; *(B2+1*4+2) = ymm6 ;
	          *(B2+0*4+3) = ymm3 ; *(B2+1*4+3) = ymm7 ;
	          B+=2;
	          B2+=8;
	    }
	    if( k & 1 ){
	          ymm0  = *(B+0+0*ldb);
	          ymm1  = *(B+0+1*ldb);
	          ymm2  = *(B+0+2*ldb);
	          ymm3  = *(B+0+3*ldb);
	          *(B2+0*4+0) = ymm0 ;
	          *(B2+0*4+1) = ymm1 ;
	          *(B2+0*4+2) = ymm2 ;
	          *(B2+0*4+3) = ymm3 ;
	          B+=1;
	          B2+=4;
	    }
	    B  = B  - K1 + 4*ldb ;

	  }
	}
	if( n & 2 ){

	    size_t k = K1;
	    if( k >> 3 ){
	      size_t k8 = ( k >> 3 );
	      while( k8-- ){
	          ymm0  = *(B+0+0*ldb); ymm4  = *(B+1+0*ldb); ymm8  = *(B+2+0*ldb); ymm12 = *(B+3+0*ldb);
	          ymm1  = *(B+0+1*ldb); ymm5  = *(B+1+1*ldb); ymm9  = *(B+2+1*ldb); ymm13 = *(B+3+1*ldb);
	          *(B2+0*2+0) = ymm0 ; *(B2+1*2+0) = ymm4 ; *(B2+2*2+0) = ymm8 ; *(B2+3*2+0) = ymm12;
	          *(B2+0*2+1) = ymm1 ; *(B2+1*2+1) = ymm5 ; *(B2+2*2+1) = ymm9 ; *(B2+3*2+1) = ymm13;
	          ymm0  = *(B+4+0*ldb); ymm4  = *(B+5+0*ldb); ymm8  = *(B+6+0*ldb); ymm12 = *(B+7+0*ldb);
	          ymm1  = *(B+4+1*ldb); ymm5  = *(B+5+1*ldb); ymm9  = *(B+6+1*ldb); ymm13 = *(B+7+1*ldb);
	          *(B2+4*2+0) = ymm0 ; *(B2+5*2+0) = ymm4 ; *(B2+6*2+0) = ymm8 ; *(B2+7*2+0) = ymm12;
	          *(B2+4*2+1) = ymm1 ; *(B2+5*2+1) = ymm5 ; *(B2+6*2+1) = ymm9 ; *(B2+7*2+1) = ymm13;
	          B+=8;
	          B2+=16;
	      }
	    }
	    if( k & 4  ){
	    //if( k >> 2 ){
	    //  size_t k4 = ( k >> 2 ); // unrolling with 8 elements
	    //  while( k4-- ){
	          ymm0  = *(B+0+0*ldb); ymm4  = *(B+1+0*ldb); ymm8  = *(B+2+0*ldb); ymm12 = *(B+3+0*ldb);
	          ymm1  = *(B+0+1*ldb); ymm5  = *(B+1+1*ldb); ymm9  = *(B+2+1*ldb); ymm13 = *(B+3+1*ldb);
	          *(B2+0*2+0) = ymm0 ; *(B2+1*2+0) = ymm4 ; *(B2+2*2+0) = ymm8 ; *(B2+3*2+0) = ymm12;
	          *(B2+0*2+1) = ymm1 ; *(B2+1*2+1) = ymm5 ; *(B2+2*2+1) = ymm9 ; *(B2+3*2+1) = ymm13;
	          B+=4;
	          B2+=8;
	    //  }
	    }
	    if( k & 2 ){
	          ymm0  = *(B+0+0*ldb); ymm4  = *(B+1+0*ldb);
	          ymm1  = *(B+0+1*ldb); ymm5  = *(B+1+1*ldb);
	          *(B2+0*2+0) = ymm0 ; *(B2+1*2+0) = ymm4 ;
	          *(B2+0*2+1) = ymm1 ; *(B2+1*2+1) = ymm5 ;
	          B+=2;
	          B2+=4;
	    }
	    if( k & 1 ){
	          ymm0  = *(B+0+0*ldb);
	          ymm1  = *(B+0+1*ldb);
	          *(B2+0*2+0) = ymm0 ;
	          *(B2+0*2+1) = ymm1 ;
	          B+=1;
	          B2+=2;
	    }
	    B  = B  - K1 + 2*ldb ;

	}
	if( n & 1 ){

	    size_t k = K1;
	    if( k >> 3 ){
	      size_t k8 = ( k >> 3 );
	      while( k8-- ){
	          ymm0  = *(B+0+0*ldb); ymm4  = *(B+1+0*ldb); ymm8  = *(B+2+0*ldb); ymm12 = *(B+3+0*ldb);
	          *(B2+0*1+0) = ymm0 ; *(B2+1*1+0) = ymm4 ; *(B2+2*1+0) = ymm8 ; *(B2+3*1+0) = ymm12;
	          ymm0  = *(B+4+0*ldb); ymm4  = *(B+5+0*ldb); ymm8  = *(B+6+0*ldb); ymm12 = *(B+7+0*ldb);
	          *(B2+4*1+0) = ymm0 ; *(B2+5*1+0) = ymm4 ; *(B2+6*1+0) = ymm8 ; *(B2+7*1+0) = ymm12;
	          B+=8;
	          B2+=8;
	      }
	    }
	    if( k & 4  ){
	    //if( k >> 2 ){
	    //  size_t k4 = ( k >> 2 ); // unrolling with 8 elements
	    //  while( k4-- ){
	          ymm0  = *(B+0+0*ldb); ymm4  = *(B+1+0*ldb); ymm8  = *(B+2+0*ldb); ymm12 = *(B+3+0*ldb);
	          *(B2+0*1+0) = ymm0 ; *(B2+1*1+0) = ymm4 ; *(B2+2*1+0) = ymm8 ; *(B2+3*1+0) = ymm12;
	          B+=4;
	          B2+=4;
	    //  }
	    }
	    if( k & 2 ){
	          ymm0  = *(B+0+0*ldb); ymm4  = *(B+1+0*ldb);
	          *(B2+0*1+0) = ymm0 ; *(B2+1*1+0) = ymm4 ;
	          B+=2;
	          B2+=2;
	    }
	    if( k & 1 ){
	          ymm0  = *(B+0+0*ldb);
	          *(B2+0*1+0) = ymm0 ;
	          B+=1;
	          B2+=1;
	    }
	    B  = B  - K1 + ldb ;

	}


}


