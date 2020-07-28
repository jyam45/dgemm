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

	double       AB;

	size_t k2 = KQ;
	while( k2-- ){
	  size_t K1 = tile_K;

	  size_t n2 = NQ;
	  while( n2-- ){
	    size_t N1 = tile_N;

	    size_t m2 = MQ;
	    while( m2-- ){
	      size_t M1 = tile_M;

	      // Kernel ----
	      size_t n1 = N1;
	      while( n1-- ){
	        size_t m1 = M1;
	        while( m1-- ){
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
	      A2 = A2 + M1*K1;
	      B2 = B2 - K1*N1;
	      C  = C - ldc*N1 + M1;
	      // ---- Kernel

	    }{ // rest of m2-loop
	      size_t M1 = MR;

	      // Kernel ----
	      size_t n1 = N1;
	      while( n1-- ){
	        size_t m1 = M1;
	        while( m1-- ){
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
	      A2 = A2 + M1*K1;
	      B2 = B2 - K1*N1;
	      C  = C - ldc*N1 + M1;
	      // ---- Kernel


	    } // end of m2-loop
	    A2 = A2 - M2*K1;
	    B2 = B2 + K1*N1;
	    C  = C - M2 + ldc*N1;

	  }{ // rest of n2-loop

	    size_t N1 = NR;

	    size_t m2 = MQ;
	    while( m2-- ){
	      size_t M1 = tile_M;

	      // Kernel ----
	      size_t n1 = N1;
	      while( n1-- ){
	        size_t m1 = M1;
	        while( m1-- ){
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
	      A2 = A2 + M1*K1;
	      B2 = B2 - K1*N1;
	      C  = C - ldc*N1 + M1;
	      // ---- Kernel

	    }{ // rest of m2-loop
	      size_t M1 = MR;

	      // Kernel ----
	      size_t n1 = N1;
	      while( n1-- ){
	        size_t m1 = M1;
	        while( m1-- ){
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

	}{// rest of k2-loop

	  size_t K1 = KR;

	  size_t n2 = NQ;
	  while( n2-- ){
	    size_t N1 = tile_N;

	    size_t m2 = MQ;
	    while( m2-- ){
	      size_t M1 = tile_M;

	      // Kernel ----
	      size_t n1 = N1;
	      while( n1-- ){
	        size_t m1 = M1;
	        while( m1-- ){
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
	      A2 = A2 + M1*K1;
	      B2 = B2 - K1*N1;
	      C  = C - ldc*N1 + M1;
	      // ---- Kernel

	    }{ // rest of m2-loop
	      size_t M1 = MR;

	      // Kernel ----
	      size_t n1 = N1;
	      while( n1-- ){
	        size_t m1 = M1;
	        while( m1-- ){
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
	      A2 = A2 + M1*K1;
	      B2 = B2 - K1*N1;
	      C  = C - ldc*N1 + M1;
	      // ---- Kernel


	    } // end of m2-loop
	    A2 = A2 - M2*K1;
	    B2 = B2 + K1*N1;
	    C  = C - M2 + ldc*N1;

	  }{ // rest of n2-loop

	    size_t N1 = NR;

	    size_t m2 = MQ;
	    while( m2-- ){
	      size_t M1 = tile_M;

	      // Kernel ----
	      size_t n1 = N1;
	      while( n1-- ){
	        size_t m1 = M1;
	        while( m1-- ){
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
	      A2 = A2 + M1*K1;
	      B2 = B2 - K1*N1;
	      C  = C - ldc*N1 + M1;
	      // ---- Kernel

	    }{ // rest of m2-loop
	      size_t M1 = MR;

	      // Kernel ----
	      size_t n1 = N1;
	      while( n1-- ){
	        size_t m1 = M1;
	        while( m1-- ){
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
