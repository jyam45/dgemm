#include "myblas_internal.h"
#include <stdio.h>

void myblas_dgemm_kernel_detail(
         size_t M1, size_t N1, size_t K1,
         double alpha, const double *A2, const double *B2, 
         double *C, size_t ldc )
{
	double c[16];

	// ---- Kernel
	size_t n = N1;
	if( n >> 2 ){
	  size_t n4 = ( n >> 2 );
	  while( n4-- ){
	    size_t m = M1;
	    if( m >> 2 ){
	      size_t m4 = ( m >> 2 );
	      while( m4-- ){

		for( size_t ii=0; ii<16; ii++ ){ c[ii]=0e0; }
	        size_t k = K1;
	        if( k >> 3 ){
	          size_t k8 = ( k >> 3 );
	          while( k8-- ){
	            for( size_t l=0; l<8; l++ ){
	              for( size_t j=0; j<4; j++ ){
	                for( size_t i=0; i<4; i++ ){
	                  c[i+4*j] += (*(A2+l*4+i))*(*(B2+l*4+j));
	                }
	              }
	            }
	            A2+=32;
	            B2+=32;
	          }
	        }
	        if( k & 4 ){
	            for( size_t l=0; l<4; l++ ){
	              for( size_t j=0; j<4; j++ ){
	                for( size_t i=0; i<4; i++ ){
	                  c[i+4*j] += (*(A2+l*4+i))*(*(B2+l*4+j));
	                }
	              }
	            }
	            A2+=16;
	            B2+=16;
	        }
	        if( k & 2 ){
	            for( size_t l=0; l<2; l++ ){
	              for( size_t j=0; j<4; j++ ){
	                for( size_t i=0; i<4; i++ ){
	                  c[i+4*j] += (*(A2+l*4+i))*(*(B2+l*4+j));
	                }
	              }
	            }
	            A2+=8;
	            B2+=8;
	        }
	        if( k & 1 ){
	            for( size_t l=0; l<1; l++ ){
	              for( size_t j=0; j<4; j++ ){
	                for( size_t i=0; i<4; i++ ){
	                  c[i+4*j] += (*(A2+l*1+i))*(*(B2+l*1+j));
	                }
	              }
	            }
	            A2+=4;
	            B2+=4;
	        }
	        for( size_t j=0; j<4; j++ ){
	          for( size_t i=0; i<4; i++ ){
	            *(C+i+j*ldc) += alpha*c[i+4*j];
	          }
	        }
	        //*C = (*C) + alpha*c00;
	        B2 = B2 - 4*K1;//N*K1
	        //C++;
	        C+=4;
	      }

	    }
	    if( m & 2 ){

		for( size_t ii=0; ii<16; ii++ ){ c[ii]=0e0; }
	        size_t k = K1;
	        if( k >> 3 ){
	          size_t k8 = ( k >> 3 );
	          while( k8-- ){
	            for( size_t l=0; l<8; l++ ){
	              for( size_t j=0; j<4; j++ ){
	                for( size_t i=0; i<2; i++ ){
	                  c[i+2*j] += (*(A2+l*2+i))*(*(B2+l*4+j));
	                }
	              }
	            }
	            A2+=16;
	            B2+=32;
	          }
	        }
	        if( k & 4 ){
	            for( size_t l=0; l<4; l++ ){
	              for( size_t j=0; j<4; j++ ){
	                for( size_t i=0; i<2; i++ ){
	                  c[i+2*j] += (*(A2+l*2+i))*(*(B2+l*4+j));
	                }
	              }
	            }
	            A2+=8;
	            B2+=16;
	        }
	        if( k & 2 ){
	            for( size_t l=0; l<2; l++ ){
	              for( size_t j=0; j<4; j++ ){
	                for( size_t i=0; i<2; i++ ){
	                  c[i+2*j] += (*(A2+l*2+i))*(*(B2+l*4+j));
	                }
	              }
	            }
	            A2+=4;
	            B2+=8;
	        }
	        if( k & 1 ){
	            for( size_t l=0; l<1; l++ ){
	              for( size_t j=0; j<4; j++ ){
	                for( size_t i=0; i<2; i++ ){
	                  c[i+2*j] += (*(A2+l*2+i))*(*(B2+l*4+j));
	                }
	              }
	            }
	            A2+=2;
	            B2+=4;
	        }
	        for( size_t j=0; j<4; j++ ){
	          for( size_t i=0; i<2; i++ ){
	            *(C+i+j*ldc) += alpha*c[i+2*j];
	          }
	        }
	        //*C = (*C) + alpha*c00;
	        B2 = B2 - 4*K1; // N*K1
	        //C++;
	        C+=2;
	      //}

	    }
	    if( m & 1 ){

		for( size_t ii=0; ii<16; ii++ ){ c[ii]=0e0; }
	        size_t k = K1;
	        if( k >> 3 ){
	          size_t k8 = ( k >> 3 );
	          while( k8-- ){
	            for( size_t l=0; l<8; l++ ){
	              for( size_t j=0; j<4; j++ ){
	                for( size_t i=0; i<1; i++ ){
	                  c[i+1*j] += (*(A2+l*1+i))*(*(B2+l*4+j));
	                }
	              }
	            }
	            A2+=8;
	            B2+=32;
	          }
	        }
	        if( k & 4 ){
	            for( size_t l=0; l<4; l++ ){
	              for( size_t j=0; j<4; j++ ){
	                for( size_t i=0; i<1; i++ ){
	                  c[i+1*j] += (*(A2+l*1+i))*(*(B2+l*4+j));
	                }
	              }
	            }
	            A2+=4;
	            B2+=16;
	        }
	        if( k & 2 ){
	            for( size_t l=0; l<2; l++ ){
	              for( size_t j=0; j<4; j++ ){
	                for( size_t i=0; i<1; i++ ){
	                  c[i+1*j] += (*(A2+l*1+i))*(*(B2+l*4+j));
	                }
	              }
	            }
	            A2+=2;
	            B2+=8;
	        }
	        if( k & 1 ){
	            for( size_t l=0; l<1; l++ ){
	              for( size_t j=0; j<4; j++ ){
	                for( size_t i=0; i<1; i++ ){
	                  c[i+1*j] += (*(A2+l*1+i))*(*(B2+l*4+j));
	                }
	              }
	            }
	            A2+=1;
	            B2+=4;
	        }
	        for( size_t j=0; j<4; j++ ){
	          for( size_t i=0; i<1; i++ ){
	            *(C+i+j*ldc) += alpha*c[i+1*j];
	          }
	        }
	        //*C = (*C) + alpha*c00;
	        B2 = B2 - 4*K1; // N*K1
	        //C++;
	        C+=1;
	      //}

	    }
	    A2 = A2 - M1*K1;
	    B2 = B2 + 4*K1;
	    C  = C - M1 + 4*ldc;
	  }
	}
	if( n & 2 ){

	    size_t m = M1;
	    if( m >> 2 ){
	      size_t m4 = ( m >> 2 );
	      while( m4-- ){

		for( size_t ii=0; ii<16; ii++ ){ c[ii]=0e0; }
	        size_t k = K1;
	        if( k >> 3 ){
	          size_t k8 = ( k >> 3 );
	          while( k8-- ){
	            for( size_t l=0; l<8; l++ ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<4; i++ ){
	                  c[i+4*j] += (*(A2+l*4+i))*(*(B2+l*2+j));
	                }
	              }
	            }
	            A2+=32;
	            B2+=16;
	          }
	        }
	        if( k & 4 ){
	            for( size_t l=0; l<4; l++ ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<4; i++ ){
	                  c[i+4*j] += (*(A2+l*4+i))*(*(B2+l*2+j));
	                }
	              }
	            }
	            A2+=16;
	            B2+=8;
	        }
	        if( k & 2 ){
	            for( size_t l=0; l<2; l++ ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<4; i++ ){
	                  c[i+4*j] += (*(A2+l*4+i))*(*(B2+l*2+j));
	                }
	              }
	            }
	            A2+=8;
	            B2+=4;
	        }
	        if( k & 1 ){
	            for( size_t l=0; l<1; l++ ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<4; i++ ){
	                  c[i+4*j] += (*(A2+l*4+i))*(*(B2+l*2+j));
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
	        B2 = B2 - 2*K1;//N*K1
	        //C++;
	        C+=4;
	      }

	    }
	    if( m & 2 ){

		for( size_t ii=0; ii<16; ii++ ){ c[ii]=0e0; }
	        size_t k = K1;
	        if( k >> 3 ){
	          size_t k8 = ( k >> 3 );
	          while( k8-- ){
	            for( size_t l=0; l<8; l++ ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<2; i++ ){
	                  c[i+2*j] += (*(A2+l*2+i))*(*(B2+l*2+j));
	                }
	              }
	            }
	            A2+=16;
	            B2+=16;
	          }
	        }
	        if( k & 4 ){
	            for( size_t l=0; l<4; l++ ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<2; i++ ){
	                  c[i+2*j] += (*(A2+l*2+i))*(*(B2+l*2+j));
	                }
	              }
	            }
	            A2+=8;
	            B2+=8;
	        }
	        if( k & 2 ){
	            for( size_t l=0; l<2; l++ ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<2; i++ ){
	                  c[i+2*j] += (*(A2+l*2+i))*(*(B2+l*2+j));
	                }
	              }
	            }
	            A2+=4;
	            B2+=4;
	        }
	        if( k & 1 ){
	            for( size_t l=0; l<1; l++ ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<2; i++ ){
	                  c[i+2*j] += (*(A2+l*2+i))*(*(B2+l*2+j));
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
	        B2 = B2 - 2*K1; // N*K1
	        //C++;
	        C+=2;
	      //}

	    }
	    if( m & 1 ){

		for( size_t ii=0; ii<16; ii++ ){ c[ii]=0e0; }
	        size_t k = K1;
	        if( k >> 3 ){
	          size_t k8 = ( k >> 3 );
	          while( k8-- ){
	            for( size_t l=0; l<8; l++ ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<1; i++ ){
	                  c[i+1*j] += (*(A2+l*1+i))*(*(B2+l*2+j));
	                }
	              }
	            }
	            A2+=8;
	            B2+=16;
	          }
	        }
	        if( k & 4 ){
	            for( size_t l=0; l<4; l++ ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<1; i++ ){
	                  c[i+1*j] += (*(A2+l*1+i))*(*(B2+l*2+j));
	                }
	              }
	            }
	            A2+=4;
	            B2+=8;
	        }
	        if( k & 2 ){
	            for( size_t l=0; l<2; l++ ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<1; i++ ){
	                  c[i+1*j] += (*(A2+l*1+i))*(*(B2+l*2+j));
	                }
	              }
	            }
	            A2+=2;
	            B2+=4;
	        }
	        if( k & 1 ){
	            for( size_t l=0; l<1; l++ ){
	              for( size_t j=0; j<2; j++ ){
	                for( size_t i=0; i<1; i++ ){
	                  c[i+1*j] += (*(A2+l*1+i))*(*(B2+l*2+j));
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
	        B2 = B2 - 2*K1; // N*K1
	        //C++;
	        C+=1;
	      //}

	    }
	    A2 = A2 - M1*K1;
	    B2 = B2 + 2*K1;
	    C  = C - M1 + 2*ldc;
	  //}

	}
	if( n & 1 ){

	    size_t m = M1;
	    if( m >> 2 ){
	      size_t m4 = ( m >> 2 );
	      while( m4-- ){

		for( size_t ii=0; ii<16; ii++ ){ c[ii]=0e0; }
	        size_t k = K1;
	        if( k >> 3 ){
	          size_t k8 = ( k >> 3 );
	          while( k8-- ){
	            for( size_t l=0; l<8; l++ ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<4; i++ ){
	                  c[i+4*j] += (*(A2+l*4+i))*(*(B2+l*1+j));
	                }
	              }
	            }
	            A2+=32;
	            B2+=8;
	          }
	        }
	        if( k & 4 ){
	            for( size_t l=0; l<4; l++ ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<4; i++ ){
	                  c[i+4*j] += (*(A2+l*4+i))*(*(B2+l*1+j));
	                }
	              }
	            }
	            A2+=16;
	            B2+=4;
	        }
	        if( k & 2 ){
	            for( size_t l=0; l<2; l++ ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<4; i++ ){
	                  c[i+4*j] += (*(A2+l*4+i))*(*(B2+l*1+j));
	                }
	              }
	            }
	            A2+=8;
	            B2+=2;
	        }
	        if( k & 1 ){
	            for( size_t l=0; l<1; l++ ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<4; i++ ){
	                  c[i+4*j] += (*(A2+l*4+i))*(*(B2+l*1+j));
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
	        B2 = B2 - 1*K1;//N*K1
	        //C++;
	        C+=4;
	      }

	    }
	    if( m & 2 ){

		for( size_t ii=0; ii<16; ii++ ){ c[ii]=0e0; }
	        size_t k = K1;
	        if( k >> 3 ){
	          size_t k8 = ( k >> 3 );
	          while( k8-- ){
	            for( size_t l=0; l<8; l++ ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<2; i++ ){
	                  c[i+2*j] += (*(A2+l*2+i))*(*(B2+l*1+j));
	                }
	              }
	            }
	            A2+=16;
	            B2+=8;
	          }
	        }
	        if( k & 4 ){
	            for( size_t l=0; l<4; l++ ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<2; i++ ){
	                  c[i+2*j] += (*(A2+l*2+i))*(*(B2+l*1+j));
	                }
	              }
	            }
	            A2+=8;
	            B2+=4;
	        }
	        if( k & 2 ){
	            for( size_t l=0; l<2; l++ ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<2; i++ ){
	                  c[i+2*j] += (*(A2+l*2+i))*(*(B2+l*1+j));
	                }
	              }
	            }
	            A2+=4;
	            B2+=2;
	        }
	        if( k & 1 ){
	            for( size_t l=0; l<1; l++ ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<2; i++ ){
	                  c[i+2*j] += (*(A2+l*2+i))*(*(B2+l*1+j));
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
	        B2 = B2 - 1*K1; // N*K1
	        //C++;
	        C+=2;
	      //}

	    }
	    if( m & 1 ){

		for( size_t ii=0; ii<16; ii++ ){ c[ii]=0e0; }
	        size_t k = K1;
	        if( k >> 3 ){
	          size_t k8 = ( k >> 3 );
	          while( k8-- ){
	            for( size_t l=0; l<8; l++ ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<1; i++ ){
	                  c[i+1*j] += (*(A2+l*1+i))*(*(B2+l*1+j));
	                }
	              }
	            }
	            A2+=8;
	            B2+=8;
	          }
	        }
	        if( k & 4 ){
	            for( size_t l=0; l<4; l++ ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<1; i++ ){
	                  c[i+1*j] += (*(A2+l*1+i))*(*(B2+l*1+j));
	                }
	              }
	            }
	            A2+=4;
	            B2+=4;
	        }
	        if( k & 2 ){
	            for( size_t l=0; l<2; l++ ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<1; i++ ){
	                  c[i+1*j] += (*(A2+l*1+i))*(*(B2+l*1+j));
	                }
	              }
	            }
	            A2+=2;
	            B2+=2;
	        }
	        if( k & 1 ){
	            for( size_t l=0; l<1; l++ ){
	              for( size_t j=0; j<1; j++ ){
	                for( size_t i=0; i<1; i++ ){
	                  c[i+1*j] += (*(A2+l*1+i))*(*(B2+l*1+j));
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
	        B2 = B2 - 1*K1; // N*K1
	        //C++;
	        C+=1;
	      //}

	    }
	    A2 = A2 - M1*K1;
	    B2 = B2 + 1*K1;
	    C  = C - M1 + 1*ldc;
	  //}


	}

	A2 = A2 + M1*K1;
	B2 = B2 - K1*N1;
	C  = C - ldc*N1 + M1;
	// ---- Kernel


}

