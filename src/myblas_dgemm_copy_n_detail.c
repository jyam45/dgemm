#include "myblas_internal.h"

void myblas_dgemm_copy_n_4x8(size_t K, size_t N, const double* B, size_t k, size_t j,  size_t ldb, double* B2 ){

	B = B + k + ldb*j; // start point

	if( N >> 2 ){
	  size_t n4 = ( N >> 2 );
	  while( n4-- ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){
	          for( size_t j=0; j<4; j++ ){
	        for( size_t l=0; l<8; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=8;
	      }
	    }
	    if( K & 4 ){
	          for( size_t j=0; j<4; j++ ){
	        for( size_t l=0; l<4; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	    }
	    if( K & 2 ){
	          for( size_t j=0; j<4; j++ ){
	        for( size_t l=0; l<2; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( K & 1 ){
	          for( size_t j=0; j<4; j++ ){
	        for( size_t l=0; l<1; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K + 4*ldb ;

	  }
	}
	if( N & 2 ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<8; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=8;
	      }
	    }
	    if( K & 4 ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<4; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	    }
	    if( K & 2 ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<2; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( K & 1 ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<1; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K + 2*ldb ;

	}
	if( N & 1 ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<8; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=8;
	      }
	    }
	    if( K & 4 ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<4; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	    }
	    if( K & 2 ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<2; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( K & 1 ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<1; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K + ldb ;

	}

}

void myblas_dgemm_copy_n_2x8(size_t K, size_t N, const double* B, size_t k, size_t j,  size_t ldb, double* B2 ){

	B = B + k + ldb*j; // start point

	if( N >> 1 ){
	  size_t n2 = ( N >> 1 );
	  while( n2-- ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<8; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=8;
	      }
	    }
	    if( K & 4 ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<4; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	    }
	    if( K & 2 ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<2; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( K & 1 ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<1; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K + 2*ldb ;
	  }
	}
	if( N & 1 ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<8; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=8;
	      }
	    }
	    if( K & 4 ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<4; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	    }
	    if( K & 2 ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<2; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( K & 1 ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<1; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K + ldb ;

	}


}


void myblas_dgemm_copy_n_4x4(size_t K, size_t N, const double* B, size_t k, size_t j,  size_t ldb, double* B2 ){

	B = B + k + ldb*j; // start point

	if( N >> 2 ){
	  size_t n4 = ( N >> 2 );
	  while( n4-- ){

	    if( K >> 2 ){
	      size_t k4 = ( K >> 2 );
	      while( k4-- ){
	          for( size_t j=0; j<4; j++ ){
	        for( size_t l=0; l<4; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	      }
	    }
	    if( K & 2 ){
	          for( size_t j=0; j<4; j++ ){
	        for( size_t l=0; l<2; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( K & 1 ){
	          for( size_t j=0; j<4; j++ ){
	        for( size_t l=0; l<1; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K + 4*ldb ;

	  }
	}
	if( N & 2 ){

	    if( K >> 2 ){
	      size_t k4 = ( K >> 2 );
	      while( k4-- ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<4; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	      }
	    }
	    if( K & 2 ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<2; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( K & 1 ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<1; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K + 2*ldb ;

	}
	if( N & 1 ){

	    if( K >> 2 ){
	      size_t k4 = ( K >> 2 );
	      while( k4-- ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<4; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	      }
	    }
	    if( K & 2 ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<2; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( K & 1 ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<1; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K + ldb ;

	}


}


