#include "myblas_internal.h"
#include <stdio.h>

void myblas_dgemm_kernel_detail(
         size_t M, size_t N, size_t K,
         double alpha, const double *A2, const double *B2, 
         double *C, size_t ldc )
{
	double c[16];

	// ---- Kernel
	
	if( N >> 1 ){
	  size_t n2 = ( N >> 1 );
	  while( n2-- ){

	    
	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

		for( size_t ii=0; ii<16; ii++ ){ c[ii]=0e0; }
	        
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<4; i++ ){
	            for( size_t l=0; l<8; l++ ){
	                  c[i+4*j] += (*(A2+l+i*8))*(*(B2+l+j*8));
	                }
	              }
	            }
	            A2+=32;
	            B2+=16;
	          }
	        }
	        if( K & 4 ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<4; i++ ){
	            for( size_t l=0; l<4; l++ ){
	                  c[i+4*j] += (*(A2+l+i*4))*(*(B2+l+j*4));
	                }
	              }
	            }
	            A2+=16;
	            B2+=8;
	        }
	        if( K & 2 ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<4; i++ ){
	            for( size_t l=0; l<2; l++ ){
	                  c[i+4*j] += (*(A2+l+i*2))*(*(B2+l+j*2));
	                }
	              }
	            }
	            A2+=8;
	            B2+=4;
	        }
	        if( K & 1 ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<4; i++ ){
	            for( size_t l=0; l<1; l++ ){
	                  c[i+4*j] += (*(A2+l+i*1))*(*(B2+l+j*1));
	                }
	              }
	            }
	            A2+=4;
	            B2+=2;
	        }
	        for( size_t j=0; j<2; j++ ){
	          for( size_t i=0; i<4; i++ ){
	            *(C+i+j*ldc) += alpha*c[i+4*j];
	          }
	        }
	        //*C = (*C) + alpha*c00;
	        B2 = B2 - 2*K;//N*K
	        //C++;
	        C+=4;
	      }

	    }
	    if( M & 2 ){

		for( size_t ii=0; ii<16; ii++ ){ c[ii]=0e0; }
	        
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<2; i++ ){
	            for( size_t l=0; l<8; l++ ){
	                  c[i+2*j] += (*(A2+l+i*8))*(*(B2+l+j*8));
	                }
	              }
	            }
	            A2+=16;
	            B2+=16;
	          }
	        }
	        if( K & 4 ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<2; i++ ){
	            for( size_t l=0; l<4; l++ ){
	                  c[i+2*j] += (*(A2+l+i*4))*(*(B2+l+j*4));
	                }
	              }
	            }
	            A2+=8;
	            B2+=8;
	        }
	        if( K & 2 ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<2; i++ ){
	            for( size_t l=0; l<2; l++ ){
	                  c[i+2*j] += (*(A2+l+i*2))*(*(B2+l+j*2));
	                }
	              }
	            }
	            A2+=4;
	            B2+=4;
	        }
	        if( K & 1 ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<2; i++ ){
	            for( size_t l=0; l<1; l++ ){
	                  c[i+2*j] += (*(A2+l+i*1))*(*(B2+l+j*1));
	                }
	              }
	            }
	            A2+=2;
	            B2+=2;
	        }
	        for( size_t j=0; j<2; j++ ){
	          for( size_t i=0; i<2; i++ ){
	            *(C+i+j*ldc) += alpha*c[i+2*j];
	          }
	        }
	        //*C = (*C) + alpha*c00;
	        B2 = B2 - 2*K; // N*K
	        //C++;
	        C+=2;
	      //}

	    }
	    if( M & 1 ){

		for( size_t ii=0; ii<16; ii++ ){ c[ii]=0e0; }
	        
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<1; i++ ){
	            for( size_t l=0; l<8; l++ ){
	                  c[i+1*j] += (*(A2+l+i*8))*(*(B2+l+j*8));
	                }
	              }
	            }
	            A2+=8;
	            B2+=16;
	          }
	        }
	        if( K & 4 ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<1; i++ ){
	            for( size_t l=0; l<4; l++ ){
	                  c[i+1*j] += (*(A2+l+i*4))*(*(B2+l+j*4));
	                }
	              }
	            }
	            A2+=4;
	            B2+=8;
	        }
	        if( K & 2 ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<1; i++ ){
	            for( size_t l=0; l<2; l++ ){
	                  c[i+1*j] += (*(A2+l+i*2))*(*(B2+l+j*2));
	                }
	              }
	            }
	            A2+=2;
	            B2+=4;
	        }
	        if( K & 1 ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<1; i++ ){
	            for( size_t l=0; l<1; l++ ){
	                  c[i+1*j] += (*(A2+l+i*1))*(*(B2+l+j*1));
	                }
	              }
	            }
	            A2+=1;
	            B2+=2;
	        }
	        for( size_t j=0; j<2; j++ ){
	          for( size_t i=0; i<1; i++ ){
	            *(C+i+j*ldc) += alpha*c[i+1*j];
	          }
	        }
	        //*C = (*C) + alpha*c00;
	        B2 = B2 - 2*K; // N*K
	        //C++;
	        C+=1;
	      //}

	    }
	    A2 = A2 - M*K;
	    B2 = B2 + 2*K;
	    C  = C - M + 2*ldc;
	  }

	}
	if( N & 1 ){

	    
	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

		for( size_t ii=0; ii<16; ii++ ){ c[ii]=0e0; }
	        
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<4; i++ ){
	            for( size_t l=0; l<8; l++ ){
	                  c[i+4*j] += (*(A2+l+i*8))*(*(B2+l+j*8));
	                }
	              }
	            }
	            A2+=32;
	            B2+=8;
	          }
	        }
	        if( K & 4 ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<4; i++ ){
	            for( size_t l=0; l<4; l++ ){
	                  c[i+4*j] += (*(A2+l+i*4))*(*(B2+l+j*4));
	                }
	              }
	            }
	            A2+=16;
	            B2+=4;
	        }
	        if( K & 2 ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<4; i++ ){
	            for( size_t l=0; l<2; l++ ){
	                  c[i+4*j] += (*(A2+l+i*2))*(*(B2+l+j*2));
	                }
	              }
	            }
	            A2+=8;
	            B2+=2;
	        }
	        if( K & 1 ){
	            for( size_t l=0; l<1; l++ ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<4; i++ ){
	                  c[i+4*j] += (*(A2+l+i*1))*(*(B2+l+j*1));
	                }
	              }
	            }
	            A2+=4;
	            B2+=1;
	        }
	        for( size_t j=0; j<1; j++ ){
	          for( size_t i=0; i<4; i++ ){
	            *(C+i+j*ldc) += alpha*c[i+4*j];
	          }
	        }
	        //*C = (*C) + alpha*c00;
	        B2 = B2 - 1*K;//N*K
	        //C++;
	        C+=4;
	      }

	    }
	    if( M & 2 ){

		for( size_t ii=0; ii<16; ii++ ){ c[ii]=0e0; }
	        
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<2; i++ ){
	            for( size_t l=0; l<8; l++ ){
	                  c[i+2*j] += (*(A2+l+i*8))*(*(B2+l+j*8));
	                }
	              }
	            }
	            A2+=16;
	            B2+=8;
	          }
	        }
	        if( K & 4 ){
	            for( size_t l=0; l<4; l++ ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<2; i++ ){
	                  c[i+2*j] += (*(A2+l+i*4))*(*(B2+l+j*4));
	                }
	              }
	            }
	            A2+=8;
	            B2+=4;
	        }
	        if( K & 2 ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<2; i++ ){
	            for( size_t l=0; l<2; l++ ){
	                  c[i+2*j] += (*(A2+l+i*2))*(*(B2+l+j*2));
	                }
	              }
	            }
	            A2+=4;
	            B2+=2;
	        }
	        if( K & 1 ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<2; i++ ){
	            for( size_t l=0; l<1; l++ ){
	                  c[i+2*j] += (*(A2+l+i*1))*(*(B2+l+j*1));
	                }
	              }
	            }
	            A2+=2;
	            B2+=1;
	        }
	        for( size_t j=0; j<1; j++ ){
	          for( size_t i=0; i<2; i++ ){
	            *(C+i+j*ldc) += alpha*c[i+2*j];
	          }
	        }
	        //*C = (*C) + alpha*c00;
	        B2 = B2 - 1*K; // N*K
	        //C++;
	        C+=2;
	      //}

	    }
	    if( M & 1 ){

		for( size_t ii=0; ii<16; ii++ ){ c[ii]=0e0; }
	        
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<1; i++ ){
	            for( size_t l=0; l<8; l++ ){
	                  c[i+1*j] += (*(A2+l+i*8))*(*(B2+l+j*8));
	                }
	              }
	            }
	            A2+=8;
	            B2+=8;
	          }
	        }
	        if( K & 4 ){
	            for( size_t l=0; l<4; l++ ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<1; i++ ){
	                  c[i+1*j] += (*(A2+l+i*4))*(*(B2+l+j*4));
	                }
	              }
	            }
	            A2+=4;
	            B2+=4;
	        }
	        if( K & 2 ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<1; i++ ){
	            for( size_t l=0; l<2; l++ ){
	                  c[i+1*j] += (*(A2+l+i*2))*(*(B2+l+j*2));
	                }
	              }
	            }
	            A2+=2;
	            B2+=2;
	        }
	        if( K & 1 ){
	            for( size_t l=0; l<1; l++ ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<1; i++ ){
	                  c[i+1*j] += (*(A2+l+i*1))*(*(B2+l+j*1));
	                }
	              }
	            }
	            A2+=1;
	            B2+=1;
	        }
	        for( size_t j=0; j<1; j++ ){
	          for( size_t i=0; i<1; i++ ){
	            *(C+i+j*ldc) += alpha*c[i+1*j];
	          }
	        }
	        //*C = (*C) + alpha*c00;
	        B2 = B2 - 1*K; // N*K
	        //C++;
	        C+=1;
	      //}

	    }
	    A2 = A2 - M*K;
	    B2 = B2 + 1*K;
	    C  = C - M + 1*ldc;
	  //}


	}

	A2 = A2 + M*K;
	B2 = B2 - K*N;
	C  = C - ldc*N + M;
	// ---- Kernel


}

