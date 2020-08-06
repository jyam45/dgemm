#include "myblas_internal.h"

void myblas_dgemm_scale2d_detail(size_t M, size_t N, double beta, double *C, size_t ldc ){

	double ymm0 ,ymm1 ,ymm2 ,ymm3 ;
	double ymm4 ,ymm5 ,ymm6 ,ymm7 ;
	double ymm8 ,ymm9 ,ymm10,ymm11;
	double ymm12,ymm13,ymm14,ymm15;

	ymm15 = beta;

	// scaling beta*C
	size_t n = N;
	if( n >> 2 ){
	    size_t n4 = ( n >> 2 ); // unrolling with 4 elements
	    while( n4-- )
	    {
	        size_t m = M;
	        if( m >> 3 ){
	            size_t m8 = ( m >> 3 );
	            while( m8-- ){
	                size_t m8a = 4; // YMM Vector-length
	                size_t m8b = 4; // YMM Vector-length
	                while( m8a-- ){ // vectorizing
	                    ymm0 = *(C+0*ldc);
	                    ymm1 = *(C+1*ldc);
	                    ymm2 = *(C+2*ldc);
	                    ymm3 = *(C+3*ldc);
	                    ymm0 *= ymm15;
	                    ymm1 *= ymm15;
	                    ymm2 *= ymm15;
	                    ymm3 *= ymm15;
	                    *(C+0*ldc) = ymm0;
	                    *(C+1*ldc) = ymm1;
	                    *(C+2*ldc) = ymm2;
	                    *(C+3*ldc) = ymm3;
	                    C++;
	                }
	                while( m8b-- ){ // vectorizing
	                    ymm8 = *(C+0*ldc);
	                    ymm9 = *(C+1*ldc);
	                    ymm10= *(C+2*ldc);
	                    ymm11= *(C+3*ldc);
	                    ymm8  *= ymm15;
	                    ymm9  *= ymm15;
	                    ymm10 *= ymm15;
	                    ymm11 *= ymm15;
	                    *(C+0*ldc) = ymm8 ;
	                    *(C+1*ldc) = ymm9 ;
	                    *(C+2*ldc) = ymm10;
	                    *(C+3*ldc) = ymm11;
	                    C++;
	                }

	            }
	        }
	        if( m & 0x4 ){
	            size_t m4 = 4;
	            while( m4-- )
	            {
	                ymm0 = *(C+0*ldc);
	                ymm1 = *(C+1*ldc);
	                ymm2 = *(C+2*ldc);
	                ymm3 = *(C+3*ldc);
	                ymm0 *= ymm15;
	                ymm1 *= ymm15;
	                ymm2 *= ymm15;
	                ymm3 *= ymm15;
	                *(C+0*ldc) = ymm0;
	                *(C+1*ldc) = ymm1;
	                *(C+2*ldc) = ymm2;
	                *(C+3*ldc) = ymm3;
	                C++;
	            }
		}
	        if( m & 0x3 ){
	            size_t mr = ( m & 0x3 );
	            while( mr-- ){
	                ymm0 = *(C+0*ldc);
	                ymm1 = *(C+1*ldc);
	                ymm2 = *(C+2*ldc);
	                ymm3 = *(C+3*ldc);
	                ymm0 *= ymm15;
	                ymm1 *= ymm15;
	                ymm2 *= ymm15;
	                ymm3 *= ymm15;
	                *(C+0*ldc) = ymm0;
	                *(C+1*ldc) = ymm1;
	                *(C+2*ldc) = ymm2;
	                *(C+3*ldc) = ymm3;
	                C++;
	            }
		}
	        C = C - M + 4*ldc;
	    }
	}
	if( n & 0x3 ){
	  size_t nr = ( n & 0x3 );
	  while( nr-- ){
	      size_t m = M;
	      while( m-- ){
	  	*C=beta*(*C);
	         C++;
	      }
	      C = C - M + ldc;
	  }
	}
	C = C - ldc*N; // retern to head of pointer.


}

