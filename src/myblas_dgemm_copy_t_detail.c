#include "myblas_internal.h"


void myblas_dgemm_copy_t_detail(size_t K1, size_t M1, const double* A, size_t k1, size_t i1, size_t lda, double* A2 ){

	A = A + i1 + k1*lda;

	size_t m = M1;
	if( m >> 2 ){
	  size_t m4 = ( m >> 2 );
	  while( m4-- ){

	    size_t k = K1;
	    if( k >> 3 ){
	      size_t k8 = ( k >> 3 );
	      while( k8-- ){
	        for( size_t l=0; l<8; l++ ){
	          for( size_t i=0; i<4; i++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 8*lda ;
	      }
	    }
	    if( k & 4 ){
	        for( size_t l=0; l<4; l++ ){
	          for( size_t i=0; i<4; i++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 4*lda ;
	    }
	    if( k & 2 ){
	        for( size_t l=0; l<2; l++ ){
	          for( size_t i=0; i<4; i++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 2*lda ;
	    }
	    if( k & 1 ){
	        for( size_t l=0; l<1; l++ ){
	          for( size_t i=0; i<4; i++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 1*lda ;
	    }
	    A  = A  - lda *K1 + 4;

	  }
	}
	if( m & 2 ){

	    size_t k = K1;
	    if( k >> 3 ){
	      size_t k8 = ( k >> 3 );
	      while( k8-- ){
	        for( size_t l=0; l<8; l++ ){
	          for( size_t i=0; i<2; i++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 8*lda ;
	      }
	    }
	    if( k & 4 ){
	        for( size_t l=0; l<4; l++ ){
	          for( size_t i=0; i<2; i++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 4*lda ;
	    }
	    if( k & 2 ){
	        for( size_t l=0; l<2; l++ ){
	          for( size_t i=0; i<2; i++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 2*lda ;
	    }
	    if( k & 1 ){
	        for( size_t l=0; l<1; l++ ){
	          for( size_t i=0; i<2; i++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 1*lda ;
	    }
	    A  = A  - lda *K1 + 2;

	}
	if( m & 1 ){

	    size_t k = K1;
	    if( k >> 3 ){
	      size_t k8 = ( k >> 3 );
	      while( k8-- ){
	        for( size_t l=0; l<8; l++ ){
	          for( size_t i=0; i<1; i++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 8*lda ;
	      }
	    }
	    if( k & 4 ){
	        for( size_t l=0; l<4; l++ ){
	          for( size_t i=0; i<1; i++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 4*lda ;
	    }
	    if( k & 2 ){
	        for( size_t l=0; l<2; l++ ){
	          for( size_t i=0; i<1; i++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 2*lda ;
	    }
	    if( k & 1 ){
	        for( size_t l=0; l<1; l++ ){
	          for( size_t i=0; i<1; i++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 1*lda ;
	    }
	    A  = A  - lda *K1 + 1;

	}


}

