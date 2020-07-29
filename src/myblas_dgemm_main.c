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

static copy_func_t copy_funcs_A[] = {myblas_dgemm_copy_t,myblas_dgemm_copy_n};
static copy_func_t copy_funcs_B[] = {myblas_dgemm_copy_n,myblas_dgemm_copy_t};

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

	copy_func_t myblas_dgemm_copy_A = copy_funcs_A[TransA];
	copy_func_t myblas_dgemm_copy_B = copy_funcs_B[TransB];

	// scaling beta*C
	block2d_info_t infoC = {M,N,1,1};
	myblas_dgemm_scale2d(beta,C,ldc,&infoC);

	double*   A2 = calloc( MYBLAS_BLOCK_M*MYBLAS_BLOCK_K, sizeof(double) );
	double*   B2 = calloc( MYBLAS_BLOCK_K*MYBLAS_BLOCK_N, sizeof(double) );

	// L3 cache
	for( size_t j3=0 ; j3<N; j3+=MIN(N-j3,MYBLAS_PANEL_N) ){
	for( size_t i3=0 ; i3<M; i3+=MIN(M-i3,MYBLAS_PANEL_M) ){
	for( size_t k3=0 ; k3<K; k3+=MIN(K-k3,MYBLAS_PANEL_K) ){
	    size_t M3 = MIN(MYBLAS_PANEL_M ,M-i3);
	    size_t N3 = MIN(MYBLAS_PANEL_N ,N-j3);
	    size_t K3 = MIN(MYBLAS_PANEL_K ,K-k3);

	    // L2 cache
	    for( size_t k2=k3; k2<k3+K3; k2+=MIN(K-k2,MYBLAS_BLOCK_K) ){
	      size_t K2 = MIN(MYBLAS_BLOCK_K ,K-k2);

	      for( size_t i2=i3; i2<i3+M3; i2+=MIN(M-i2,MYBLAS_BLOCK_M) ){
	        size_t M2 = MIN(MYBLAS_BLOCK_M ,M-i2);

	        // On L2-Cache Copy for A
	        block2d_info_t infoA = {k2,i2,K2,M2,MYBLAS_TILE_K,MYBLAS_TILE_M};
	        myblas_dgemm_copy_A(A,lda,A2,&infoA);

	        for( size_t j2=j3; j2<j3+N3; j2+=MIN(N-j2,MYBLAS_BLOCK_N) ){
	          size_t N2 = MIN(MYBLAS_BLOCK_N ,N-j2);

	          // On L2-Cache Copy for B
	          block2d_info_t infoB = {k2,j2,K2,N2,MYBLAS_TILE_K,MYBLAS_TILE_N};
	          myblas_dgemm_copy_B(B,ldb,B2,&infoB);

	          // L1 cache
	          block3d_info_t info3d = {M2,N2,K2,MYBLAS_TILE_M,MYBLAS_TILE_N,MYBLAS_TILE_K};
	          myblas_dgemm_kernel(alpha,A2,B2,C+ldc*j2+i2,ldc,&info3d);

	        }
	      }
	    }

	}}}

	free(A2);
	free(B2);
}
