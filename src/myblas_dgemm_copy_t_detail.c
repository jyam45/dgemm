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


	if( K >> 3 ){
	  size_t k8 = ( K >> 3 );
	  while( k8-- ){
	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        for( size_t l=0; l<8; l++ ){
	          for( size_t i=0; i<4; i++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A  = A  + 4;
	        A2 = A2 - 4*8 + 4*K;

	      }
	      A2 = A2 - (M&~3)*K + 4*8; // move to next row block
	    }
	    if( M & 2 ){

	        for( size_t l=0; l<8; l++ ){
	          for( size_t i=0; i<2; i++ ){
	            (*A2_2) = *(A+i+l*lda);
	            A2_2++;
	          }
	        }
	        A  = A  + 2;

	    }
	    if( M & 1 ){

	        for( size_t l=0; l<8; l++ ){
	          for( size_t i=0; i<1; i++ ){
	            (*A2_1) = *(A+i+l*lda);
	            A2_1++;
	          }
	        }
	        A  = A  + 1;

	    }
	    A  = A - M + 8*lda;
	  }
	}
	if( K & 4  ){
	    if( M >> 2 ){
	      //printf("K4M4:A2-A0=%d\n",A2-A0);
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        for( size_t l=0; l<4; l++ ){
	          for( size_t i=0; i<4; i++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A  = A  + 4;
	        A2 = A2 - 4*4 + 4*K;

	      }
	      A2 = A2 - (M&~3)*K + 4*4;
	    }
	    if( M & 2 ){

	        //printf("K4M2:A2_2-A0=%d\n",A2_2-A0);
	        for( size_t l=0; l<4; l++ ){
	          for( size_t i=0; i<2; i++ ){
	            (*A2_2) = *(A+i+l*lda);
	            A2_2++;
	          }
	        }
	        A  = A  + 2;

	    }
	    if( M & 1 ){

	        //printf("K4M1:A2_1-A0=%d\n",A2_1-A0);
	        for( size_t l=0; l<4; l++ ){
	          for( size_t i=0; i<1; i++ ){
	            (*A2_1) = *(A+i+l*lda);
	            A2_1++;
	          }
	        }
	        A  = A  + 1;

	    }
	    A  = A - M + 4*lda;

	}
	if( K & 2  ){
	    if( M >> 2 ){
	      //printf("K2M4:A2-A0=%d\n",A2-A0);
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        for( size_t l=0; l<2; l++ ){
	          for( size_t i=0; i<4; i++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A  = A  + 4;
	        A2 = A2 - 4*2 + 4*K;

	      }
	      A2 = A2 - (M&~3)*K + 4*2;

	    }
	    if( M & 2 ){

	        //printf("K2M2:A2_2-A0=%d\n",A2_2-A0);
	        for( size_t l=0; l<2; l++ ){
	          for( size_t i=0; i<2; i++ ){
	            (*A2_2) = *(A+i+l*lda);
	            A2_2++;
	          }
	        }
	        A  = A  + 2;

	    }
	    if( M & 1 ){

	        //printf("K2M1:A2_1-A0=%d\n",A2_1-A0);
	        for( size_t l=0; l<2; l++ ){
	          for( size_t i=0; i<1; i++ ){
	            (*A2_1) = *(A+i+l*lda);
	            A2_1++;
	          }
	        }
	        A  = A  + 1;

	    }
	    A  = A - M + 2*lda;

	}
	if( K & 1  ){
	    if( M >> 2 ){
	      //printf("K1M4:A2-A0=%d\n",A2-A0);
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        for( size_t l=0; l<1; l++ ){
	          for( size_t i=0; i<4; i++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A  = A  + 4;
	        A2 = A2 - 4*1 + 4*K;

	      }
	      A2 = A2 - (M&~3)*K + 4*1;

	    }
	    if( M & 2 ){

	        //printf("K1M2:A2_2-A0=%d\n",A2_2-A0);
	        for( size_t l=0; l<1; l++ ){
	          for( size_t i=0; i<2; i++ ){
	            (*A2_2) = *(A+i+l*lda);
	            A2_2++;
	          }
	        }
	        A  = A  + 2;

	    }
	    if( M & 1 ){

	        //printf("K1M1:A2_1-A0=%d\n",A2_1-A0);
	        for( size_t l=0; l<1; l++ ){
	          for( size_t i=0; i<1; i++ ){
	            (*A2_1) = *(A+i+l*lda);
	            A2_1++;
	          }
	        }
	        A  = A  + 1;

	    }
	    A  = A - M + 1*lda;

	}

	
}

