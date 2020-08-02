#include "myblas_internal.h"

void myblas_dgemm_kernel(double alpha, const double *A2, const double *B2, 
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

	double ymm0 ,ymm1 ,ymm2 ,ymm3 ;
	double ymm4 ,ymm5 ,ymm6 ,ymm7 ;
	double ymm8 ,ymm9 ,ymm10,ymm11;
	double ymm12,ymm13,ymm14,ymm15;

	double a00,a01,a02,a03;
	double a10,a11,a12,a13;
	double a20,a21,a22,a23;
	double a30,a31,a32,a33;

	double b00,b01,b02,b03;
	double b10,b11,b12,b13;
	double b20,b21,b22,b23;
	double b30,b31,b32,b33;

	double c00,c01,c02,c03;
	double c10,c11,c12,c13;
	double c20,c21,c22,c23;
	double c30,c31,c32,c33;

	double       AB;

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

	      // Kernel ----
	      size_t n = N1;
	      if( n >> 2 ){
	        size_t n4 = ( n >> 2 ); // unrolling N
	        while( n4-- ){
	          size_t m = M1;
	          while( m-- ){
	              //AB=0e0;
	              c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	              size_t k = K1;
	              while( k-- ){
	                a00  = *(A2 + 0*K1 );
	                b00  = *(B2 + 0*K1 );
	                b01  = *(B2 + 1*K1 );
	                b02  = *(B2 + 2*K1 );
	                b03  = *(B2 + 3*K1 );
	                c00 += a00 * b00;
	                c01 += a00 * b01;
	                c02 += a00 * b02;
	                c03 += a00 * b03;
	                //AB = AB + (*A2)*(*B2);
	                A2++;
	                B2++;
	              }
	              //*C = (*C) + alpha*AB;
	              *(C+0+0*ldc) += alpha*c00;
	              *(C+0+1*ldc) += alpha*c01;
	              *(C+0+2*ldc) += alpha*c02;
	              *(C+0+3*ldc) += alpha*c03;
	              B2 = B2 - K1;
	              C++;
	          }
	          A2 = A2 - M1*K1;
	          B2 = B2 + 4*K1;
	          C  = C - M1 + 4*ldc;
	        }
	      }
	      if( n & 3 ){
	        size_t nr = ( n & 3 ); // unrolling N
	        while( nr-- ){
	          size_t m = M1;
	          while( m-- ){
	            AB=0e0;
	            size_t k = K1;
	            while( k-- ){
	              AB = AB + (*A2)*(*B2);
	              A2++;
	              B2++;
	            }
	            *C = (*C) + alpha*AB;
	            B2 = B2 - K1;
	            C++;
	          }
	          A2 = A2 - M1*K1;
	          B2 = B2 + K1;
	          C  = C - M1 + ldc;
	        }
	      }

	      A2 = A2 + M1*K1;
	      B2 = B2 - K1*N1;
	      C  = C - ldc*N1 + M1;
	      // ---- Kernel

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
