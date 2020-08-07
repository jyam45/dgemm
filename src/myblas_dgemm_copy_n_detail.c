#include "myblas_internal.h"

void myblas_dgemm_copy_n_detail(size_t K1, size_t N1, const double* B, size_t k, size_t j,  size_t ldb, double* B2 ){

	B = B + k + ldb*j; // start point

	size_t n = N1;
	if( n >> 2 ){
	  size_t n4 = ( n >> 2 );
	  while( n4-- ){

	    size_t k = K1;
	    if( k >> 3 ){
	      size_t k8 = ( k >> 3 );
	      while( k8-- ){
	        for( size_t l=0; l<8; l++ ){
	          for( size_t j=0; j<4; j++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=8;
	      }
	    }
	    if( k & 4  ){
	        for( size_t l=0; l<4; l++ ){
	          for( size_t j=0; j<4; j++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	    }
	    if( k & 2 ){
	        for( size_t l=0; l<2; l++ ){
	          for( size_t j=0; j<4; j++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( k & 1 ){
	        for( size_t l=0; l<1; l++ ){
	          for( size_t j=0; j<4; j++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K1 + 4*ldb ;

	  }
	}
	if( n & 2 ){

	    size_t k = K1;
	    if( k >> 3 ){
	      size_t k8 = ( k >> 3 );
	      while( k8-- ){
	        for( size_t l=0; l<8; l++ ){
	          for( size_t j=0; j<2; j++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=8;
	      }
	    }
	    if( k & 4  ){
	        for( size_t l=0; l<4; l++ ){
	          for( size_t j=0; j<2; j++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	    }
	    if( k & 2 ){
	        for( size_t l=0; l<2; l++ ){
	          for( size_t j=0; j<2; j++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( k & 1 ){
	        for( size_t l=0; l<1; l++ ){
	          for( size_t j=0; j<2; j++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K1 + 2*ldb ;

	}
	if( n & 1 ){

	    size_t k = K1;
	    if( k >> 3 ){
	      size_t k8 = ( k >> 3 );
	      while( k8-- ){
	        for( size_t l=0; l<8; l++ ){
	          for( size_t j=0; j<1; j++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=8;
	      }
	    }
	    if( k & 4 ){
	        for( size_t l=0; l<4; l++ ){
	          for( size_t j=0; j<1; j++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	    }
	    if( k & 2 ){
	        for( size_t l=0; l<2; l++ ){
	          for( size_t j=0; j<1; j++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( k & 1 ){
	        for( size_t l=0; l<1; l++ ){
	          for( size_t j=0; j<1; j++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K1 + ldb ;

	}


}


