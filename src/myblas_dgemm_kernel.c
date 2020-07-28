#include "myblas_internal.h"

#define MIN(x,y)  (((x)<(y))?(x):(y))
void myblas_dgemm_kernel(double alpha, const double *A2, const double *B2, 
                         double *C, size_t ldc, const block3d_info_t* info ){

	size_t i2     = info->i2    ;
	size_t j2     = info->j2    ;
	size_t k2     = info->k2    ;
	size_t M      = info->M     ;
	size_t N      = info->N     ;
	size_t K      = info->K     ;
	size_t M2     = info->M2    ;
	size_t N2     = info->N2    ;
	size_t K2     = info->K2    ;
	size_t tile_M = info->tile_M;
	size_t tile_N = info->tile_N;
	size_t tile_K = info->tile_K;

	double       AB;

	for( size_t k1=k2; k1<k2+K2; k1+=MIN(K-k1,tile_K ) ){
	  size_t K1 = MIN(tile_K ,K-k1);
	  for( size_t j1=j2; j1<j2+N2; j1+=MIN(N-j1,tile_N ) ){
	    size_t N1 = MIN(tile_N ,N-j1);
	    for( size_t i1=i2; i1<i2+M2; i1+=MIN(M-i1,tile_M ) ){
	      size_t M1 = MIN(tile_M ,M-i1);

	      for( size_t j =j1; j < j1+N1; j++ ){
	        for( size_t i =i1; i < i1+M1; i++ ){
	          AB=0e0;
	          for( size_t k =k1; k < k1+K1; k++ ){
	            AB = AB + (*A2)*(*B2);
	            //printf("A2(%d,%d)=%G A2(%d,%d)=%G\n",i,k,*A2,i,k,*A2);
	            //printf("B2(%d,%d)=%G B2(%d,%d)=%G\n",k,j,*B2,k,j,*B2);
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

	    }// i1
	    A2 = A2 - M2*K1;
	    B2 = B2 + K1*N1;
	    C  = C - M2 + ldc*N1;

	  }// j1
	  A2 = A2 + M2*K1;
	  C  = C - ldc*N2;

	}// k1
	A2 = A2 - M2*K2;
	B2 = B2 - K2*N2;

}
