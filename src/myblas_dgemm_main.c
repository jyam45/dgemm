#include "myblas_internal.h"
#include <stdlib.h>
#include <stdio.h>

#define MYBLAS_PANEL_M  256
#define MYBLAS_PANEL_N  256
#define MYBLAS_PANEL_K  256

#define MYBLAS_BLOCK_M  128
#define MYBLAS_BLOCK_N   64
#define MYBLAS_BLOCK_K  128

#define MYBLAS_TILE_M    32
#define MYBLAS_TILE_N    32
#define MYBLAS_TILE_K    32

#define MIN(x,y)  (((x)<(y))?(x):(y))

void myblas_dgemm_main( gemm_args_t* args ){

	size_t       TransA = args->TransA;
	size_t       TransB = args->TransB;
	size_t       M      = args->M;
	size_t       N      = args->N;
	size_t       K      = args->K;
	double       alpha  = args->alpha;
	const double *A     = args->A;
	size_t       lda    = args->lda;
	const double *B     = args->B;
	size_t       ldb    = args->ldb;
	double       beta   = args->beta;
	double       *C     = args->C;
	size_t       ldc    = args->ldc;

	double       AB;

	if( TransA & MASK_TRANS ){	
	    if( TransB & MASK_TRANS ){

	        for( size_t j=0; j<N; j++ ){
	            for( size_t i=0; i<M; i++ ){
			AB=0e0;
	                for( size_t k=0; k<K; k++ ){
	                   AB = AB + (*A)*(*B);
	                   A++;
	                   B+=ldb;
	                }
			*C=beta*(*C) + alpha*AB;
	                A = A - K + lda;
	                B = B - ldb*K;
	                C++;
	            }
	            A = A - lda*M;
	            B = B + 1;
	            C = C - M + ldc;
	        }

	    }else{

	        for( size_t j=0; j<N; j++ ){
	            for( size_t i=0; i<M; i++ ){
			AB=0e0;
	                for( size_t k=0; k<K; k++ ){
	                   AB = AB + (*A)*(*B);
	                   A++;
	                   B++;
	                }
			*C=beta*(*C) + alpha*AB;
	                A = A - K + lda;
	                B = B - K;
	                C++;
	            }
	            A = A - lda*M;
	            B = B + ldb;
	            C = C - M + ldc;
	        }

	    }
	}else{
	    if( TransB & MASK_TRANS ){	

	        for( size_t j=0; j<N; j++ ){
	            for( size_t i=0; i<M; i++ ){
			AB=0e0;
	                for( size_t k=0; k<K; k++ ){
	                   AB = AB + (*A)*(*B);
	                   A += lda;
	                   B += ldb;
	                }
			*C=beta*(*C) + alpha*AB;
	                A = A - lda*K + 1;
	                B = B - ldb*K;
	                C++;
	            }
	            A = A - M;
	            B = B + 1;
	            C = C - M + ldc;
	        }

	    }else{

	        // scaling beta*C
	        for( size_t j=0; j<N; j++ ){
	            for( size_t i=0; i<M; i++ ){
			*C=beta*(*C);
	                C++;
	            }
	            C = C - M + ldc;
	        }
	        C = C - ldc*N; // retern to head of pointer.

		double*   A2 = calloc( MYBLAS_BLOCK_M*MYBLAS_BLOCK_K, sizeof(double) );
	        size_t  lda2 = MYBLAS_BLOCK_M;
		double*   B2 = calloc( MYBLAS_BLOCK_K*MYBLAS_BLOCK_N, sizeof(double) );
	        size_t  ldb2 = MYBLAS_BLOCK_K;

	        // L3 cache
	        for( size_t j3=0 ; j3<N; j3+=MIN(N-j3,MYBLAS_PANEL_N) ){
	        for( size_t i3=0 ; i3<M; i3+=MIN(M-i3,MYBLAS_PANEL_M) ){
	        for( size_t k3=0 ; k3<K; k3+=MIN(K-k3,MYBLAS_PANEL_K) ){
	            size_t M3 = MIN(MYBLAS_PANEL_M ,M-i3);
	            size_t N3 = MIN(MYBLAS_PANEL_N ,N-j3);
	            size_t K3 = MIN(MYBLAS_PANEL_K ,K-k3);
	            // L2 cache
	            for( size_t j2=j3; j2<j3+N3; j2+=MIN(N-j2,MYBLAS_BLOCK_N) ){
	            for( size_t i2=i3; i2<i3+M3; i2+=MIN(M-i2,MYBLAS_BLOCK_M) ){
	            for( size_t k2=k3; k2<k3+K3; k2+=MIN(K-k2,MYBLAS_BLOCK_K) ){
	                size_t M2 = MIN(MYBLAS_BLOCK_M ,M-i2);
	                size_t N2 = MIN(MYBLAS_BLOCK_N ,N-j2);
	                size_t K2 = MIN(MYBLAS_BLOCK_K ,K-k2);

	                // On L2-Cache Copy for A
	                A = A + lda*k2 + i2;
	                for( size_t i1=i2; i1<i2+M2; i1+=MIN(M-i1,MYBLAS_TILE_M ) ){
	                  size_t M1 = MIN(MYBLAS_TILE_M ,M-i1);
	                  for( size_t k1=k2; k1<k2+K2; k1+=MIN(K-k1,MYBLAS_TILE_K ) ){
	                    size_t K1 = MIN(MYBLAS_TILE_K ,K-k1);
	                    for( size_t i =i1; i < i1+M1; i++ ){
	                      for( size_t k =k1; k < k1+K1; k++ ){
	                        (*A2) = (*A);
	                        A += lda ;
	                        A2+= lda2;
	                      }
	                      A  = A  - lda *K1 + 1;
	                      A2 = A2 - lda2*K1 + 1;
	                    }
	                    A = A - M1 + lda *K1;
	                    A2= A2- M1 + lda2*K1;
	                  }
	                  A = A  - lda *K2 + M1;
	                  A2= A2 - lda2*K2 + M1;
	                }
	                A = A  - M2;
	                A2= A2 - M2;
	                A = A - lda*k2 - i2; // return to head

	                // On L2-Cache Copy for B
	                B = B + ldb*j2 + k2;
	                for( size_t j1=j2; j1<j2+N2; j1+=MIN(N-j1,MYBLAS_TILE_N ) ){
	                  size_t N1 = MIN(MYBLAS_TILE_N ,N-j1);
	                  for( size_t k1=k2; k1<k2+K2; k1+=MIN(K-k1,MYBLAS_TILE_K ) ){
	                    size_t K1 = MIN(MYBLAS_TILE_K ,K-k1);
	                    for( size_t j =j1; j < j1+N1; j++ ){
	                      for( size_t k =k1; k < k1+K1; k++ ){
	                        *B2=*B;
	                        B++;
	                        B2++;
	                      }
	                      B  = B  - K1 + ldb ;
	                      B2 = B2 - K1 + ldb2;
	                    }
	                    B  = B  - ldb *N1 + K1;
	                    B2 = B2 - ldb2*N1 + K1;
	                  }
	                  B  = B  - K2 + ldb *N1;
	                  B2 = B2 - K2 + ldb2*N1;
	                }
	                B  = B  - ldb *N2;
	                B2 = B2 - ldb2*N2;
	                B  = B  - ldb*j2 - k2; // return to head

	                // L1 cache
	                for( size_t j1=j2; j1<j2+N2; j1+=MIN(N-j1,MYBLAS_TILE_N ) ){
	                for( size_t i1=i2; i1<i2+M2; i1+=MIN(M-i1,MYBLAS_TILE_M ) ){
	                for( size_t k1=k2; k1<k2+K2; k1+=MIN(K-k1,MYBLAS_TILE_K ) ){

	                    C = C + ldc*j1 + i1;

	                    A2 = A2 + lda2*(k1-k2) + (i1-i2);
	                    B2 = B2 + ldb2*(j1-j2) + (k1-k2);

	                    size_t M1 = MIN(MYBLAS_TILE_M ,M-i1);
	                    size_t N1 = MIN(MYBLAS_TILE_N ,N-j1);
	                    size_t K1 = MIN(MYBLAS_TILE_K ,K-k1);
	                    for( size_t j =j1; j < j1+N1; j++ ){
	                      for( size_t i =i1; i < i1+M1; i++ ){
		                AB=0e0;
	                        for( size_t k =k1; k < k1+K1; k++ ){
	                          AB = AB + (*A2)*(*B2);
	                          //printf("A(%d,%d)=%G A2(%d,%d)=%G\n",i,k,*A,i,k,*A2);
	                          //printf("B(%d,%d)=%G B2(%d,%d)=%G\n",k,j,*B,k,j,*B2);
	                          A2+= lda2;
	                          B2++;
	                        }
			        *C = (*C) + alpha*AB;
	                        A2 = A2 - lda2*K1 + 1;
	                        B2 = B2 - K1;
	                        C++;
	                      }
	                      A2 = A2- M1;
	                      B2 = B2 + ldb2;
	                      C  = C - M1 + ldc;
	                    }
	                    B2 = B2 - ldb2*N1;
	                    C  = C - ldc*N1;

	                    C = C - ldc*j1 - i1;

	                    A2 = A2 - lda2*(k1-k2) - (i1-i2);
	                    B2 = B2 - ldb2*(j1-j2) - (k1-k2);

	                }}}

	            }}}

	        }}}

	        free(A2);
	        free(B2);

	    }
	}
}
