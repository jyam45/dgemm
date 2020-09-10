#include "myblas_internal.h"


void myblas_dgemm_copy_t_4x8(size_t K, size_t M, const double* A, size_t k1, size_t i1, size_t lda, double* A2 ){

	A = A + i1 + k1*lda;

	if( M >> 2 ){
	  size_t m4 = ( M >> 2 );
	  while( m4-- ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){
	          for( size_t i=0; i<4; i++ ){
	        for( size_t l=0; l<8; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 8*lda ;
	      }
	    }
	    if( K & 4 ){
	          for( size_t i=0; i<4; i++ ){
	        for( size_t l=0; l<4; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 4*lda ;
	    }
	    if( K & 2 ){
	          for( size_t i=0; i<4; i++ ){
	        for( size_t l=0; l<2; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 2*lda ;
	    }
	    if( K & 1 ){
	          for( size_t i=0; i<4; i++ ){
	        for( size_t l=0; l<1; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 1*lda ;
	    }
	    A  = A  - lda *K + 4;

	  }
	}
	if( M & 2 ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){
	          for( size_t i=0; i<2; i++ ){
	        for( size_t l=0; l<8; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 8*lda ;
	      }
	    }
	    if( K & 4 ){
	          for( size_t i=0; i<2; i++ ){
	        for( size_t l=0; l<4; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 4*lda ;
	    }
	    if( K & 2 ){
	          for( size_t i=0; i<2; i++ ){
	        for( size_t l=0; l<2; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 2*lda ;
	    }
	    if( K & 1 ){
	          for( size_t i=0; i<2; i++ ){
	        for( size_t l=0; l<1; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 1*lda ;
	    }
	    A  = A  - lda *K + 2;

	}
	if( M & 1 ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){
	          for( size_t i=0; i<1; i++ ){
	        for( size_t l=0; l<8; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 8*lda ;
	      }
	    }
	    if( K & 4 ){
	          for( size_t i=0; i<1; i++ ){
	        for( size_t l=0; l<4; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 4*lda ;
	    }
	    if( K & 2 ){
	          for( size_t i=0; i<1; i++ ){
	        for( size_t l=0; l<2; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 2*lda ;
	    }
	    if( K & 1 ){
	          for( size_t i=0; i<1; i++ ){
	        for( size_t l=0; l<1; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 1*lda ;
	    }
	    A  = A  - lda *K + 1;

	}

}

void myblas_dgemm_copy_t_2x8(size_t K, size_t M, const double* A, size_t k1, size_t i1, size_t lda, double* A2 ){

	A = A + i1 + k1*lda;

	if( M >> 1 ){
	  size_t m2 = ( M >> 1 );
	  while( m2-- ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){
	          for( size_t i=0; i<2; i++ ){
	        for( size_t l=0; l<8; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 8*lda ;
	      }
	    }
	    if( K & 4 ){
	          for( size_t i=0; i<2; i++ ){
	        for( size_t l=0; l<4; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 4*lda ;
	    }
	    if( K & 2 ){
	          for( size_t i=0; i<2; i++ ){
	        for( size_t l=0; l<2; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 2*lda ;
	    }
	    if( K & 1 ){
	          for( size_t i=0; i<2; i++ ){
	        for( size_t l=0; l<1; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 1*lda ;
	    }
	    A  = A  - lda *K + 2;
	  }
	}
	if( M & 1 ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){
	          for( size_t i=0; i<1; i++ ){
	        for( size_t l=0; l<8; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 8*lda ;
	      }
	    }
	    if( K & 4 ){
	          for( size_t i=0; i<1; i++ ){
	        for( size_t l=0; l<4; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 4*lda ;
	    }
	    if( K & 2 ){
	          for( size_t i=0; i<1; i++ ){
	        for( size_t l=0; l<2; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 2*lda ;
	    }
	    if( K & 1 ){
	          for( size_t i=0; i<1; i++ ){
	        for( size_t l=0; l<1; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 1*lda ;
	    }
	    A  = A  - lda *K + 1;

	}

}



void myblas_dgemm_copy_t_4x4(size_t K, size_t M, const double* A, size_t k1, size_t i1, size_t lda, double* A2 ){

	A = A + i1 + k1*lda;

	if( M >> 2 ){
	  size_t m4 = ( M >> 2 );
	  while( m4-- ){

	    if( K >> 2 ){
	      size_t k4 = ( K >> 2 );
	      while( k4-- ){
	          for( size_t i=0; i<4; i++ ){
	        for( size_t l=0; l<4; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 4*lda ;
	      }
	    }
	    if( K & 2 ){
	          for( size_t i=0; i<4; i++ ){
	        for( size_t l=0; l<2; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 2*lda ;
	    }
	    if( K & 1 ){
	          for( size_t i=0; i<4; i++ ){
	        for( size_t l=0; l<1; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 1*lda ;
	    }
	    A  = A  - lda *K + 4;

	  }
	}
	if( M & 2 ){

	    if( K >> 2 ){
	      size_t k4 = ( K >> 2 );
	      while( k4-- ){
	          for( size_t i=0; i<2; i++ ){
	        for( size_t l=0; l<4; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 4*lda ;
	      }
	    }
	    if( K & 2 ){
	          for( size_t i=0; i<2; i++ ){
	        for( size_t l=0; l<2; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 2*lda ;
	    }
	    if( K & 1 ){
	          for( size_t i=0; i<2; i++ ){
	        for( size_t l=0; l<1; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 1*lda ;
	    }
	    A  = A  - lda *K + 2;

	}
	if( M & 1 ){

	    if( K >> 2 ){
	      size_t k4 = ( K >> 2 );
	      while( k4-- ){
	          for( size_t i=0; i<1; i++ ){
	        for( size_t l=0; l<4; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 4*lda ;
	      }
	    }
	    if( K & 2 ){
	          for( size_t i=0; i<1; i++ ){
	        for( size_t l=0; l<2; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 2*lda ;
	    }
	    if( K & 1 ){
	          for( size_t i=0; i<1; i++ ){
	        for( size_t l=0; l<1; l++ ){
	            (*A2) = *(A+i+l*lda);
	            A2++;
	          }
	        }
	        A += 1*lda ;
	    }
	    A  = A  - lda *K + 1;

	}


}

