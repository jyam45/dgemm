#include "myblas_internal.h"
#include <stdlib.h>
//#include <stdio.h>

void myblas_dgemm_copy_t_detail(size_t K, size_t M, const double* A, size_t k, size_t i, size_t lda, double* A2 ){

	//size_t k = K;
	//while( k-- ){
	//  size_t m = M;
	//  while( m-- ){
	//    (*A2) = (*A);
	//    A++;
	//    A2+=K;
	//  }
	//  A2 = A2 - K*M + 1;
	//  A  = A  - M + lda;
	//}
	double* A0 = A2;
	double* A2_2 = A2   + K*( M & ~3 );
	double* A2_1 = A2_2 + K*( M &  2 );;

	double x00,x01,x02,x03;
	double x10,x11,x12,x13;
	double x20,x21,x22,x23;
	double x30,x31,x32,x33;

	if( K >> 3 ){
	  size_t k8 = ( K >> 3 );
	  while( k8-- ){
	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        //for( size_t l=0; l<8; l++ ){
	        //  for( size_t i=0; i<4; i++ ){
	        //    (*A2) = *(A+i+l*lda);
	        //    A2++;
	        //  }
	        //}
	        x00 = *(A+0+0*lda); x01 = *(A+0+1*lda); x02 = *(A+0+2*lda); x03 = *(A+0+3*lda);
	        x10 = *(A+1+0*lda); x11 = *(A+1+1*lda); x12 = *(A+1+2*lda); x13 = *(A+1+3*lda);
	        x20 = *(A+2+0*lda); x21 = *(A+2+1*lda); x22 = *(A+2+2*lda); x23 = *(A+2+3*lda);
	        x30 = *(A+3+0*lda); x31 = *(A+3+1*lda); x32 = *(A+3+2*lda); x33 = *(A+3+3*lda);
	        *(A2+0+0*4) = x00; *(A2+0+1*4) = x01; *(A2+0+2*4) = x02; *(A2+0+3*4) = x03;
	        *(A2+1+0*4) = x10; *(A2+1+1*4) = x11; *(A2+1+2*4) = x12; *(A2+1+3*4) = x13;
	        *(A2+2+0*4) = x20; *(A2+2+1*4) = x21; *(A2+2+2*4) = x22; *(A2+2+3*4) = x23;
	        *(A2+3+0*4) = x30; *(A2+3+1*4) = x31; *(A2+3+2*4) = x32; *(A2+3+3*4) = x33;
	        x00 = *(A+0+4*lda); x01 = *(A+0+5*lda); x02 = *(A+0+6*lda); x03 = *(A+0+7*lda);
	        x10 = *(A+1+4*lda); x11 = *(A+1+5*lda); x12 = *(A+1+6*lda); x13 = *(A+1+7*lda);
	        x20 = *(A+2+4*lda); x21 = *(A+2+5*lda); x22 = *(A+2+6*lda); x23 = *(A+2+7*lda);
	        x30 = *(A+3+4*lda); x31 = *(A+3+5*lda); x32 = *(A+3+6*lda); x33 = *(A+3+7*lda);
	        *(A2+0+4*4) = x00; *(A2+0+5*4) = x01; *(A2+0+6*4) = x02; *(A2+0+7*4) = x03;
	        *(A2+1+4*4) = x10; *(A2+1+5*4) = x11; *(A2+1+6*4) = x12; *(A2+1+7*4) = x13;
	        *(A2+2+4*4) = x20; *(A2+2+5*4) = x21; *(A2+2+6*4) = x22; *(A2+2+7*4) = x23;
	        *(A2+3+4*4) = x30; *(A2+3+5*4) = x31; *(A2+3+6*4) = x32; *(A2+3+7*4) = x33;

	        A  = A  + 4;
	        //A2 = A2 - 4*8 + 4*K;
	        A2 = A2 + 4*K;

	      }
	      A2 = A2 - (M&~3)*K + 4*8; // move to next row block
	    }
	    if( M & 2 ){

	        //for( size_t l=0; l<8; l++ ){
	        //  for( size_t i=0; i<2; i++ ){
	        //    (*A2_2) = *(A+i+l*lda);
	        //    A2_2++;
	        //  }
	        //}
	        x00 = *(A+0+0*lda); x01 = *(A+0+1*lda); x02 = *(A+0+2*lda); x03 = *(A+0+3*lda);
	        x10 = *(A+1+0*lda); x11 = *(A+1+1*lda); x12 = *(A+1+2*lda); x13 = *(A+1+3*lda);
	        *(A2_2+0+0*2) = x00; *(A2_2+0+1*2) = x01; *(A2_2+0+2*2) = x02; *(A2_2+0+3*2) = x03;
	        *(A2_2+1+0*2) = x10; *(A2_2+1+1*2) = x11; *(A2_2+1+2*2) = x12; *(A2_2+1+3*2) = x13;
	        x00 = *(A+0+4*lda); x01 = *(A+0+5*lda); x02 = *(A+0+6*lda); x03 = *(A+0+7*lda);
	        x10 = *(A+1+4*lda); x11 = *(A+1+5*lda); x12 = *(A+1+6*lda); x13 = *(A+1+7*lda);
	        *(A2_2+0+4*2) = x00; *(A2_2+0+5*2) = x01; *(A2_2+0+6*2) = x02; *(A2_2+0+7*2) = x03;
	        *(A2_2+1+4*2) = x10; *(A2_2+1+5*2) = x11; *(A2_2+1+6*2) = x12; *(A2_2+1+7*2) = x13;
	        A  = A  + 2;
	        A2_2 += 2*8;

	    }
	    if( M & 1 ){

	        //for( size_t l=0; l<8; l++ ){
	        //  for( size_t i=0; i<1; i++ ){
	        //    (*A2_1) = *(A+i+l*lda);
	        //    A2_1++;
	        //  }
	        //}
	        x00 = *(A+0+0*lda); x01 = *(A+0+1*lda); x02 = *(A+0+2*lda); x03 = *(A+0+3*lda);
	        *(A2_1+0+0*1) = x00; *(A2_1+0+1*1) = x01; *(A2_1+0+2*1) = x02; *(A2_1+0+3*1) = x03;
	        x00 = *(A+0+4*lda); x01 = *(A+0+5*lda); x02 = *(A+0+6*lda); x03 = *(A+0+7*lda);
	        *(A2_1+0+4*1) = x00; *(A2_1+0+5*1) = x01; *(A2_1+0+6*1) = x02; *(A2_1+0+7*1) = x03;
	        A  = A  + 1;
	        A2_1 += 1*8;

	    }
	    A  = A - M + 8*lda;
	  }
	}
	if( K & 4  ){
	    if( M >> 2 ){
	      //printf("K4M4:A2-A0=%d\n",A2-A0);
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        //for( size_t l=0; l<4; l++ ){
	        //  for( size_t i=0; i<4; i++ ){
	        //    (*A2) = *(A+i+l*lda);
	        //    A2++;
	        //  }
	        //}
	        x00 = *(A+0+0*lda); x01 = *(A+0+1*lda); x02 = *(A+0+2*lda); x03 = *(A+0+3*lda);
	        x10 = *(A+1+0*lda); x11 = *(A+1+1*lda); x12 = *(A+1+2*lda); x13 = *(A+1+3*lda);
	        x20 = *(A+2+0*lda); x21 = *(A+2+1*lda); x22 = *(A+2+2*lda); x23 = *(A+2+3*lda);
	        x30 = *(A+3+0*lda); x31 = *(A+3+1*lda); x32 = *(A+3+2*lda); x33 = *(A+3+3*lda);
	        *(A2+0+0*4) = x00; *(A2+0+1*4) = x01; *(A2+0+2*4) = x02; *(A2+0+3*4) = x03;
	        *(A2+1+0*4) = x10; *(A2+1+1*4) = x11; *(A2+1+2*4) = x12; *(A2+1+3*4) = x13;
	        *(A2+2+0*4) = x20; *(A2+2+1*4) = x21; *(A2+2+2*4) = x22; *(A2+2+3*4) = x23;
	        *(A2+3+0*4) = x30; *(A2+3+1*4) = x31; *(A2+3+2*4) = x32; *(A2+3+3*4) = x33;
	        A  = A  + 4;
	        //A2 = A2 - 4*4 + 4*K;
	        A2 = A2 + 4*K;

	      }
	      A2 = A2 - (M&~3)*K + 4*4;
	    }
	    if( M & 2 ){

	        //printf("K4M2:A2_2-A0=%d\n",A2_2-A0);
	        //for( size_t l=0; l<4; l++ ){
	        //  for( size_t i=0; i<2; i++ ){
	        //    (*A2_2) = *(A+i+l*lda);
	        //    A2_2++;
	        //  }
	        //}
	        x00 = *(A+0+0*lda); x01 = *(A+0+1*lda); x02 = *(A+0+2*lda); x03 = *(A+0+3*lda);
	        x10 = *(A+1+0*lda); x11 = *(A+1+1*lda); x12 = *(A+1+2*lda); x13 = *(A+1+3*lda);
	        *(A2_2+0+0*2) = x00; *(A2_2+0+1*2) = x01; *(A2_2+0+2*2) = x02; *(A2_2+0+3*2) = x03;
	        *(A2_2+1+0*2) = x10; *(A2_2+1+1*2) = x11; *(A2_2+1+2*2) = x12; *(A2_2+1+3*2) = x13;
	        A  = A  + 2;
	        A2_2 += 2*4;

	    }
	    if( M & 1 ){

	        //printf("K4M1:A2_1-A0=%d\n",A2_1-A0);
	        //for( size_t l=0; l<4; l++ ){
	        //  for( size_t i=0; i<1; i++ ){
	        //    (*A2_1) = *(A+i+l*lda);
	        //    A2_1++;
	        //  }
	        //}
	        x00 = *(A+0+0*lda); x01 = *(A+0+1*lda); x02 = *(A+0+2*lda); x03 = *(A+0+3*lda);
	        *(A2_1+0+0*1) = x00; *(A2_1+0+1*1) = x01; *(A2_1+0+2*1) = x02; *(A2_1+0+3*1) = x03;
	        A  = A  + 1;
	        A2_1 += 1*4;

	    }
	    A  = A - M + 4*lda;

	}
	if( K & 2  ){
	    if( M >> 2 ){
	      //printf("K2M4:A2-A0=%d\n",A2-A0);
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        //for( size_t l=0; l<2; l++ ){
	        //  for( size_t i=0; i<4; i++ ){
	        //    (*A2) = *(A+i+l*lda);
	        //    A2++;
	        //  }
	        //}
	        x00 = *(A+0+0*lda); x01 = *(A+0+1*lda);
	        x10 = *(A+1+0*lda); x11 = *(A+1+1*lda);
	        x20 = *(A+2+0*lda); x21 = *(A+2+1*lda);
	        x30 = *(A+3+0*lda); x31 = *(A+3+1*lda);
	        *(A2+0+0*4) = x00; *(A2+0+1*4) = x01;
	        *(A2+1+0*4) = x10; *(A2+1+1*4) = x11;
	        *(A2+2+0*4) = x20; *(A2+2+1*4) = x21;
	        *(A2+3+0*4) = x30; *(A2+3+1*4) = x31;
	        A  = A  + 4;
	        //A2 = A2 - 4*2 + 4*K;
	        A2 = A2 + 4*K;

	      }
	      A2 = A2 - (M&~3)*K + 4*2;

	    }
	    if( M & 2 ){

	        //printf("K2M2:A2_2-A0=%d\n",A2_2-A0);
	        //for( size_t l=0; l<2; l++ ){
	        //  for( size_t i=0; i<2; i++ ){
	        //    (*A2_2) = *(A+i+l*lda);
	        //    A2_2++;
	        //  }
	        //}
	        x00 = *(A+0+0*lda); x01 = *(A+0+1*lda);
	        x10 = *(A+1+0*lda); x11 = *(A+1+1*lda);
	        *(A2_2+0+0*2) = x00; *(A2_2+0+1*2) = x01;
	        *(A2_2+1+0*2) = x10; *(A2_2+1+1*2) = x11;
	        A  = A  + 2;
	        A2_2 += 2*2;

	    }
	    if( M & 1 ){

	        //printf("K2M1:A2_1-A0=%d\n",A2_1-A0);
	        //for( size_t l=0; l<2; l++ ){
	        //  for( size_t i=0; i<1; i++ ){
	        //    (*A2_1) = *(A+i+l*lda);
	        //    A2_1++;
	        //  }
	        //}
	        x00 = *(A+0+0*lda); x01 = *(A+0+1*lda);
	        *(A2_1+0+0*1) = x00; *(A2_1+0+1*1) = x01;
	        A  = A  + 1;
	        A2_1 += 1*2;

	    }
	    A  = A - M + 2*lda;

	}
	if( K & 1  ){
	    if( M >> 2 ){
	      //printf("K1M4:A2-A0=%d\n",A2-A0);
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        //for( size_t l=0; l<1; l++ ){
	        //  for( size_t i=0; i<4; i++ ){
	        //    (*A2) = *(A+i+l*lda);
	        //    A2++;
	        //  }
	        //}
	        x00 = *(A+0+0*lda);
	        x10 = *(A+1+0*lda);
	        x20 = *(A+2+0*lda);
	        x30 = *(A+3+0*lda);
	        *(A2+0+0*4) = x00;
	        *(A2+1+0*4) = x10;
	        *(A2+2+0*4) = x20;
	        *(A2+3+0*4) = x30;
	        A  = A  + 4;
	        //A2 = A2 - 4*1 + 4*K;
	        A2 = A2 + 4*K;

	      }
	      A2 = A2 - (M&~3)*K + 4*1;

	    }
	    if( M & 2 ){

	        //printf("K1M2:A2_2-A0=%d\n",A2_2-A0);
	        //for( size_t l=0; l<1; l++ ){
	        //  for( size_t i=0; i<2; i++ ){
	        //    (*A2_2) = *(A+i+l*lda);
	        //    A2_2++;
	        //  }
	        //}
	        x00 = *(A+0+0*lda);
	        x10 = *(A+1+0*lda);
	        *(A2_2+0+0*2) = x00;
	        *(A2_2+1+0*2) = x10;
	        A  = A  + 2;
	        A2_2 += 2*1;

	    }
	    if( M & 1 ){

	        //printf("K1M1:A2_1-A0=%d\n",A2_1-A0);
	        //for( size_t l=0; l<1; l++ ){
	        //  for( size_t i=0; i<1; i++ ){
	        //    (*A2_1) = *(A+i+l*lda);
	        //    A2_1++;
	        //  }
	        //}
	        x00 = *(A+0+0*lda);
	        *(A2_1+0+0*1) = x00;
	        A  = A  + 1;
	        A2_1 += 1*1;

	    }
	    A  = A - M + 1*lda;

	}

	
}

