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
	      myblas_dgemm_kernel_core( alpha, A2, B2, C, ldc, &tile );
	
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


void myblas_dgemm_kernel_core(double alpha, const double *A2, const double *B2, 
                              double *C, size_t ldc, const block3d_info_t* info ){

	size_t M1     = info->M2    ;
	size_t N1     = info->N2    ;
	size_t K1     = info->K2    ;

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

	// Kernel ----
	size_t n = N1;
	if( n >> 2 ){
	  size_t n4 = ( n >> 2 ); // unrolling N
	  while( n4-- ){
	    size_t m = M1;
	    if( m >> 2 ){
	      size_t m4 = ( m >> 2 ); // unrolling M
	      while( m4-- ){
	        //AB=0e0;
	        c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        c10=0e0;c11=0e0;c12=0e0;c13=0e0;
	        c20=0e0;c21=0e0;c22=0e0;c23=0e0;
	        c30=0e0;c31=0e0;c32=0e0;c33=0e0;
	        size_t k = K1;
	        if( k >> 2 ){
	          size_t k4 = ( k >> 2 ); // Unrolling K
	          while( k4-- ){
	            a00 = *(A2 + 0 + 0*4 ); a10 = *(A2 + 1 + 0*4 ); a20 = *(A2 + 2 + 0*4 ); a30 = *(A2 + 3 + 0*4 ); // ymm0
	            a01 = *(A2 + 0 + 1*4 ); a11 = *(A2 + 1 + 1*4 ); a21 = *(A2 + 2 + 1*4 ); a31 = *(A2 + 3 + 1*4 ); // ymm1
	            a02 = *(A2 + 0 + 2*4 ); a12 = *(A2 + 1 + 2*4 ); a22 = *(A2 + 2 + 2*4 ); a32 = *(A2 + 3 + 2*4 ); // ymm2
	            a03 = *(A2 + 0 + 3*4 ); a13 = *(A2 + 1 + 3*4 ); a23 = *(A2 + 2 + 3*4 ); a33 = *(A2 + 3 + 3*4 ); // ymm3
	            b00 = *(B2 + 0 + 0*4 ); b10 = *(B2 + 1 + 0*4 ); b20 = *(B2 + 2 + 0*4 ); b30 = *(B2 + 3 + 0*4 ); // ymm4
	            b01 = *(B2 + 0 + 1*4 ); b11 = *(B2 + 1 + 1*4 ); b21 = *(B2 + 2 + 1*4 ); b31 = *(B2 + 3 + 1*4 ); // ymm5
	            b02 = *(B2 + 0 + 2*4 ); b12 = *(B2 + 1 + 2*4 ); b22 = *(B2 + 2 + 2*4 ); b32 = *(B2 + 3 + 2*4 ); // ymm6
	            b03 = *(B2 + 0 + 3*4 ); b13 = *(B2 + 1 + 3*4 ); b23 = *(B2 + 2 + 3*4 ); b33 = *(B2 + 3 + 3*4 ); // ymm7
	            c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            c02 += a00 * b02; c02 += a10 * b12; c02 += a20 * b22; c02 += a30 * b32; // ymm14
	            c03 += a00 * b03; c03 += a10 * b13; c03 += a20 * b23; c03 += a30 * b33; // ymm15
	            c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            c12 += a01 * b02; c12 += a11 * b12; c12 += a21 * b22; c12 += a31 * b32; // ymm14
	            c13 += a01 * b03; c13 += a11 * b13; c13 += a21 * b23; c13 += a31 * b33; // ymm15
	            c20 += a02 * b00; c20 += a12 * b10; c20 += a22 * b20; c20 += a32 * b30; // ymm12
	            c21 += a02 * b01; c21 += a12 * b11; c21 += a22 * b21; c21 += a32 * b31; // ymm13
	            c22 += a02 * b02; c22 += a12 * b12; c22 += a22 * b22; c22 += a32 * b32; // ymm14
	            c23 += a02 * b03; c23 += a12 * b13; c23 += a22 * b23; c23 += a32 * b33; // ymm15
	            c30 += a03 * b00; c30 += a13 * b10; c30 += a23 * b20; c30 += a33 * b30; // ymm12
	            c31 += a03 * b01; c31 += a13 * b11; c31 += a23 * b21; c31 += a33 * b31; // ymm13
	            c32 += a03 * b02; c32 += a13 * b12; c32 += a23 * b22; c32 += a33 * b32; // ymm14
	            c33 += a03 * b03; c33 += a13 * b13; c33 += a23 * b23; c33 += a33 * b33; // ymm15
	            //AB = AB + (*A2)*(*B2);
	            A2+=16;
	            B2+=16;
	          }
	        }
	        if( k & 3 ){
	          size_t kr = ( k & 3 ); // Unrolling K
	          while( kr-- ){
	            a00  = *(A2 + 0*4 );
	            a01  = *(A2 + 1*4 );
	            a02  = *(A2 + 2*4 );
	            a03  = *(A2 + 3*4 );
	            b00  = *(B2 + 0*4 );
	            b01  = *(B2 + 1*4 );
	            b02  = *(B2 + 2*4 );
	            b03  = *(B2 + 3*4 );
	            c00 += a00 * b00;
	            c01 += a00 * b01;
	            c02 += a00 * b02;
	            c03 += a00 * b03;
	            c10 += a01 * b00;
	            c11 += a01 * b01;
	            c12 += a01 * b02;
	            c13 += a01 * b03;
	            c20 += a02 * b00;
	            c21 += a02 * b01;
	            c22 += a02 * b02;
	            c23 += a02 * b03;
	            c30 += a03 * b00;
	            c31 += a03 * b01;
	            c32 += a03 * b02;
	            c33 += a03 * b03;
	            //AB = AB + (*A2)*(*B2);
	            A2++;
	            B2++;
	          }

	        }
	        //*C = (*C) + alpha*AB;
	        *(C+0+0*ldc) += alpha*c00;
	        *(C+0+1*ldc) += alpha*c01;
	        *(C+0+2*ldc) += alpha*c02;
	        *(C+0+3*ldc) += alpha*c03;
	        *(C+1+0*ldc) += alpha*c10;
	        *(C+1+1*ldc) += alpha*c11;
	        *(C+1+2*ldc) += alpha*c12;
	        *(C+1+3*ldc) += alpha*c13;
	        *(C+2+0*ldc) += alpha*c20;
	        *(C+2+1*ldc) += alpha*c21;
	        *(C+2+2*ldc) += alpha*c22;
	        *(C+2+3*ldc) += alpha*c23;
	        *(C+3+0*ldc) += alpha*c30;
	        *(C+3+1*ldc) += alpha*c31;
	        *(C+3+2*ldc) += alpha*c32;
	        *(C+3+3*ldc) += alpha*c33;
	        //A2 = A2 - K1 + 4*K1;
	        B2 = B2 - 4*K1;
	        C+=4;
	      }
	    }
	    if( m & 3 ){
	      size_t mr = ( m & 3 ); // unrolling M
	      while( mr-- ){
	    //  while( m-- ){
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
	      c00=0e0;
	      size_t k = K1;
	      while( k-- ){
	        c00 = c00 + (*A2)*(*B2);
	        A2++;
	        B2++;
	      }
	      *C = (*C) + alpha*c00;
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

}

