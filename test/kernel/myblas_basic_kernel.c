#include "myblas_internal.h"
#include "kernel_test.h"
#include <stdio.h>

void myblas_basic_kernel(double alpha, const double *A2, const double *B2, 
                         double *C, size_t ldc, const block3d_info_t* info ){

	size_t M2     = info->M2    ;
	size_t N2     = info->N2    ;
	size_t K2     = info->K2    ;
	size_t tile_M = info->tile_M;
	size_t tile_N = info->tile_N;
	size_t tile_K = info->tile_K;

	size_t MQ = M2/tile_M;
	size_t MR = M2%tile_M;
	size_t NQ = N2/tile_N;
	size_t NR = N2%tile_N;
	size_t KQ = K2/tile_K;
	size_t KR = K2%tile_K;

	block3d_info_t tile = { 0, 0, 0, 0, 0, 0 };

	if( MR >  0 ){ MQ++; }
	if( MR == 0 ){ MR = tile_M; }
	if( NR >  0 ){ NQ++; }
	if( NR == 0 ){ NR = tile_N; }
	if( KR >  0 ){ KQ++; }
	if( KR == 0 ){ KR = tile_K; }

	// L1-cache blocking
	size_t k1 = KQ;
	while( k1-- ){
	  size_t K1 = tile_K; if( k1==0 ){ K1=KR; }

	  size_t n1 = NQ;
	  while( n1-- ){
	    size_t N1 = tile_N; if( n1==0 ){ N1=NR; }

	    size_t m1 = MQ;
	    while( m1-- ){
	      size_t M1 = tile_M; if( m1==0 ){ M1=MR; }

	      tile.M2 = M1; tile.N2 = N1; tile.K2 = K1;
	      myblas_basic_kernel_core( alpha, A2, B2, C, ldc, &tile );
	
              A2 = A2 + M1*K1;
	      C  = C  + M1;

	    } // end of m2-loop
	    A2 = A2 - M2*K1;
	    B2 = B2 + K1*N1;
	    C  = C - M2 + ldc*N1;

	  } // end of n2-loop
	  A2 = A2 + M2*K1;
	  C  = C - ldc*N2;

	} // end of k2-loop
	A2 = A2 - M2*K2;
	B2 = B2 - K2*N2;

}


void myblas_basic_kernel_core(double alpha, const double *A2, const double *B2, 
                              double *C, size_t ldc, const block3d_info_t* info ){

	size_t M1     = info->M2    ;
	size_t N1     = info->N2    ;
	size_t K1     = info->K2    ;

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
	            for( size_t j=0; j<4; j++ ){
	              for( size_t i=0; i<4; i++ ){
	                for( size_t l=0; l<8; l++ ){
	                  c[i+4*j] += (*(A2+l+i*8))*(*(B2+l+j*8));
	                }
	              }
	            }
	            A2+=32;
	            B2+=32;
	          }
	        }
	        if( k & 4 ){
	            for( size_t j=0; j<4; j++ ){
	              for( size_t i=0; i<4; i++ ){
	                for( size_t l=0; l<4; l++ ){
	                  c[i+4*j] += (*(A2+l+i*4))*(*(B2+l+j*4));
	                }
	              }
	            }
	            A2+=16;
	            B2+=16;
	        }
	        if( k & 2 ){
	            for( size_t j=0; j<4; j++ ){
	              for( size_t i=0; i<4; i++ ){
	                for( size_t l=0; l<2; l++ ){
	                  c[i+4*j] += (*(A2+l+i*2))*(*(B2+l+j*2));
	                }
	              }
	            }
	            A2+=8;
	            B2+=8;
	        }
	        if( k & 1 ){
	            for( size_t j=0; j<4; j++ ){
	              for( size_t i=0; i<4; i++ ){
	                for( size_t l=0; l<1; l++ ){
	                  c[i+4*j] += (*(A2+l+i*1))*(*(B2+l+j*1));
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
	            for( size_t j=0; j<4; j++ ){
	              for( size_t i=0; i<2; i++ ){
	                for( size_t l=0; l<8; l++ ){
	                  c[i+2*j] += (*(A2+l+i*8))*(*(B2+l+j*8));
	                }
	              }
	            }
	            A2+=16;
	            B2+=32;
	          }
	        }
	        if( k & 4 ){
	            for( size_t j=0; j<4; j++ ){
	              for( size_t i=0; i<2; i++ ){
	                for( size_t l=0; l<4; l++ ){
	                  c[i+2*j] += (*(A2+l+i*4))*(*(B2+l+j*4));
	                }
	              }
	            }
	            A2+=8;
	            B2+=16;
	        }
	        if( k & 2 ){
	            for( size_t j=0; j<4; j++ ){
	              for( size_t i=0; i<2; i++ ){
	                for( size_t l=0; l<2; l++ ){
	                  c[i+2*j] += (*(A2+l+i*2))*(*(B2+l+j*2));
	                }
	              }
	            }
	            A2+=4;
	            B2+=8;
	        }
	        if( k & 1 ){
	            for( size_t j=0; j<4; j++ ){
	              for( size_t i=0; i<2; i++ ){
	                for( size_t l=0; l<1; l++ ){
	                  c[i+2*j] += (*(A2+l+i*1))*(*(B2+l+j*1));
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
	            for( size_t j=0; j<4; j++ ){
	              for( size_t i=0; i<1; i++ ){
	                for( size_t l=0; l<8; l++ ){
	                  c[i+1*j] += (*(A2+l+i*8))*(*(B2+l+j*8));
	                }
	              }
	            }
	            A2+=8;
	            B2+=32;
	          }
	        }
	        if( k & 4 ){
	            for( size_t j=0; j<4; j++ ){
	              for( size_t i=0; i<1; i++ ){
	                for( size_t l=0; l<4; l++ ){
	                  c[i+1*j] += (*(A2+l+i*4))*(*(B2+l+j*4));
	                }
	              }
	            }
	            A2+=4;
	            B2+=16;
	        }
	        if( k & 2 ){
	            for( size_t j=0; j<4; j++ ){
	              for( size_t i=0; i<1; i++ ){
	                for( size_t l=0; l<2; l++ ){
	                  c[i+1*j] += (*(A2+l+i*2))*(*(B2+l+j*2));
	                }
	              }
	            }
	            A2+=2;
	            B2+=8;
	        }
	        if( k & 1 ){
	            for( size_t j=0; j<4; j++ ){
	              for( size_t i=0; i<1; i++ ){
	                for( size_t l=0; l<1; l++ ){
	                  c[i+1*j] += (*(A2+l+i*1))*(*(B2+l+j*1));
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
	        if( k & 4 ){
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
	        if( k & 2 ){
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
	        if( k & 1 ){
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
	        if( k & 4 ){
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
	        if( k & 2 ){
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
	        if( k & 1 ){
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
	        if( k & 4 ){
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
	        if( k & 2 ){
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
	        if( k & 1 ){
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
	        if( k & 4 ){
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
	        if( k & 2 ){
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
	        if( k & 1 ){
	            for( size_t j=0; j<1; j++ ){
	              for( size_t i=0; i<4; i++ ){
	                for( size_t l=0; l<1; l++ ){
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
	        if( k & 4 ){
	            for( size_t j=0; j<1; j++ ){
	              for( size_t i=0; i<2; i++ ){
	                for( size_t l=0; l<4; l++ ){
	                  c[i+2*j] += (*(A2+l+i*4))*(*(B2+l+j*4));
	                }
	              }
	            }
	            A2+=8;
	            B2+=4;
	        }
	        if( k & 2 ){
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
	        if( k & 1 ){
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
	        if( k & 4 ){
	            for( size_t j=0; j<1; j++ ){
	              for( size_t i=0; i<1; i++ ){
	                for( size_t l=0; l<4; l++ ){
	                  c[i+1*j] += (*(A2+l+i*4))*(*(B2+l+j*4));
	                }
	              }
	            }
	            A2+=4;
	            B2+=4;
	        }
	        if( k & 2 ){
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
	        if( k & 1 ){
	            for( size_t j=0; j<1; j++ ){
	              for( size_t i=0; i<1; i++ ){
	                for( size_t l=0; l<1; l++ ){
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

