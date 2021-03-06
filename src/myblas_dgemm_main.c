#include "myblas_internal.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MYBLAS_PANEL_M  2048
#define MYBLAS_PANEL_N  2048
#define MYBLAS_PANEL_K  2048

#define MYBLAS_BLOCK_M  512
#define MYBLAS_BLOCK_N  512
#define MYBLAS_BLOCK_K  256

#define MYBLAS_TILE_M    16
#define MYBLAS_TILE_N     6
#define MYBLAS_TILE_K   256

#define ALIGNMENT_B      32  // for AVX

#define MIN(x,y)  (((x)<(y))?(x):(y))

static copy_func_t copy_funcs_A[] = {myblas_dgemm_copy_t,myblas_dgemm_copy_n};
static copy_func_t copy_funcs_B[] = {myblas_dgemm_copy_n,myblas_dgemm_copy_t};
//static copy_func_t copy_funcs_A[] = {myblas_dgemm_copy_t_core,myblas_dgemm_copy_n_core};
//static copy_func_t copy_funcs_B[] = {myblas_dgemm_copy_n_core,myblas_dgemm_copy_t_core};

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

	kernel_func_t myblas_dgemm_kernel_AB = myblas_dgemm_kernel;
	//kernel_func_t myblas_dgemm_kernel_AB = myblas_dgemm_kernel_core;

	copy_func_t myblas_dgemm_copy_A = copy_funcs_A[TransA];
	copy_func_t myblas_dgemm_copy_B = copy_funcs_B[TransB];


	#ifdef _OPENMP
	//#pragma omp parallel reduction(+:C[0:ldc*N-1])
	#pragma omp parallel
	#endif
	{

	    #ifdef _OPENMP
	    size_t num_threads = omp_get_num_threads();
	    size_t thread_id   = omp_get_thread_num();
	    #else
	    size_t num_threads = 1;
	    size_t thread_id   = 0;
	    #endif

	    size_t N_threads = myblas_num_col_threads(num_threads);
	    size_t M_threads = myblas_num_row_threads(num_threads);
	    size_t N_thread_id = thread_id%N_threads;
	    size_t M_thread_id = thread_id/N_threads;

	    //printf("N_threads=%d, M_threads=%d, N_thread_id=%d, M_thread_id=%d\n",N_threads,M_threads,N_thread_id,M_thread_id);

	    size_t Nth  = (N/N_threads)+(N%N_threads>N_thread_id?1:0); // K-size per a thread
	    size_t j3th = N_thread_id * (N/N_threads) + (N%N_threads<N_thread_id?N%N_threads:N_thread_id); // K start point in a thread
	    size_t Nend = j3th+Nth;

	    size_t Mth  = (M/M_threads)+(M%M_threads>M_thread_id?1:0); // K-size per a thread
	    size_t i3th = M_thread_id * (M/M_threads) + (M%M_threads<M_thread_id?M%M_threads:M_thread_id); // K start point in a thread
	    size_t Mend = i3th+Mth;

	    // scaling beta*C
	    block2d_info_t infoC = {Mth,Nth,1,1};
	    myblas_dgemm_scale2d(beta,C+i3th+j3th*ldc,ldc,&infoC);

	    //printf("Parallel=[%d,%d], thread_id=(%d,%d), sub-matrix size=[%d,%d], index=(%d:%d,%d:%d)\n",M_threads,N_threads,M_thread_id,N_thread_id,Mth,Nth,i3th,Mend,j3th,Nend);

	    //double*   A2 = calloc( MYBLAS_BLOCK_M*MYBLAS_BLOCK_K, sizeof(double) );
	    //double*   B2 = calloc( MYBLAS_BLOCK_K*MYBLAS_BLOCK_N, sizeof(double) );
	    double*   A2 = aligned_alloc( ALIGNMENT_B, MYBLAS_BLOCK_M*MYBLAS_BLOCK_K*sizeof(double) ); // C11 standard
	    double*   B2 = aligned_alloc( ALIGNMENT_B, MYBLAS_BLOCK_K*MYBLAS_BLOCK_N*sizeof(double) ); // C11 standard

	    // L3 cache
	    for( size_t k3=0 ; k3<K; k3+=MIN(K-k3,MYBLAS_PANEL_K) ){
	        size_t K3 = MIN(MYBLAS_PANEL_K ,K-k3);
	        size_t K3B = K3/MYBLAS_BLOCK_K; // number of full blocks
	        size_t K3R = K3%MYBLAS_BLOCK_K; // size of the final block
	        K3B = K3B + (K3R>0?1:0); // number of blocks
	        if( K3R==0 ) K3R = MYBLAS_BLOCK_K;

	    for( size_t j3=j3th ; j3<Nend; j3+=MIN(Nend-j3,MYBLAS_PANEL_N) ){
	        size_t N3 = MIN(MYBLAS_PANEL_N ,Nend-j3);
	        size_t N3B = N3/MYBLAS_BLOCK_N; // number of full blocks
	        size_t N3R = N3%MYBLAS_BLOCK_N; // size of the final block
	        N3B = N3B + (N3R>0?1:0); // number of blocks
	        if( N3R==0 ) N3R = MYBLAS_BLOCK_N;

	    for( size_t i3=i3th ; i3<Mend; i3+=MIN(Mend-i3,MYBLAS_PANEL_M) ){
	        size_t M3 = MIN(MYBLAS_PANEL_M ,Mend-i3);
	        size_t M3B = M3/MYBLAS_BLOCK_M; // number of full blocks
	        size_t M3R = M3%MYBLAS_BLOCK_M; // size of the final block
	        M3B = M3B + (M3R>0?1:0); // number of blocks
	        if( M3R==0 ) M3R = MYBLAS_BLOCK_M;

	        // L2 cache
	        size_t l2 = K3B; // block index
	        size_t k2 = k3;  // element index
	        while( l2-- ){
	          size_t K2 = MYBLAS_BLOCK_K; if( l2==0 ){ K2=K3R; }

	          size_t m2n2 = M3B*N3B; // m2n2=m2*N3B+n2
	          size_t m2 = M3B-1;
	          size_t n2 = N3B-1;
	          size_t i2 = i3;  // element index
	          size_t j2 = j3;  // element index
	          size_t M2 = MYBLAS_BLOCK_M; if( i2+M2 >= i3+M3 ){ M2=M3R; }
	          size_t N2 = MYBLAS_BLOCK_N; if( j2+N2 >= j3+N3 ){ N2=N3R; }

	          // First Blocks
	          {

	            //printf("m2=%d n2=%d  A(%d,%d)[%d,%d] x B(%d,%d)[%d,%d] = C(%d,%d)[%d,%d]\n",m2,n2,i2,k2,M2,K2,k2,j2,K2,N2,i2,j2,M2,N2); 
	            // On L2-Cache Copy for A
	            block2d_info_t infoA = {k2,i2,K2,M2,MYBLAS_TILE_K,MYBLAS_TILE_M,COPY_MK};
	            myblas_dgemm_copy_A(A,lda,A2,&infoA);

	            // On L2-Cache Copy for B
	            block2d_info_t infoB = {k2,j2,K2,N2,MYBLAS_TILE_K,MYBLAS_TILE_N,COPY_NK};
	            myblas_dgemm_copy_B(B,ldb,B2,&infoB);

	            // L1 cache
	            block3d_info_t info3d = {M2,N2,K2,MYBLAS_TILE_M,MYBLAS_TILE_N,MYBLAS_TILE_K};
	            myblas_dgemm_kernel_AB(alpha,A2,B2,C+ldc*j2+i2,ldc,&info3d);

	            //double* B1 = B2;
	            //for( size_t j1 = j2; j1 < j2+N2; j1+=4 ){ 
	            //  size_t N1 = 4; if( j1+4 >= j2+N2 ){ N1 = j2+N2 - j1; }

	            //  // On L2-Cache Copy for B
	            //  block2d_info_t infoB = {k2,j1,K2,N1,MYBLAS_TILE_K,MYBLAS_TILE_N};
	            //  myblas_dgemm_copy_B(B,ldb,B1,&infoB);

	            //  // L1 cache
	            //  block3d_info_t info3d = {M2,N1,K2,MYBLAS_TILE_M,MYBLAS_TILE_N,MYBLAS_TILE_K};
	            //  myblas_dgemm_kernel_AB(alpha,A2,B1,C+ldc*j1+i2,ldc,&info3d);

	            //  B1 += K2*N1;
	            //}

	            m2n2--;
	          }

	          while( m2n2-- ){

	            if( n2 > 0 ){ 
	              n2--; 
	              if( j2+N2 < j3+N3 ){ j2+=N2; }else{ j2=j3; }
	              N2 = MYBLAS_BLOCK_N; if( j2+N2 >= j3+N3 ){ N2=N3R; }

	              //printf("m2=%d n2=%d  A(%d,%d)[%d,%d] x B(%d,%d)[%d,%d] = C(%d,%d)[%d,%d]\n",m2,n2,i2,k2,M2,K2,k2,j2,K2,N2,i2,j2,M2,N2); 

	              // On L2-Cache Copy for B
	              block2d_info_t infoB = {k2,j2,K2,N2,MYBLAS_TILE_K,MYBLAS_TILE_N,COPY_NK};
	              myblas_dgemm_copy_B(B,ldb,B2,&infoB);

	              // L1 cache
	              block3d_info_t info3d = {M2,N2,K2,MYBLAS_TILE_M,MYBLAS_TILE_N,MYBLAS_TILE_K};
	              myblas_dgemm_kernel_AB(alpha,A2,B2,C+ldc*j2+i2,ldc,&info3d);

	              //double* B1 = B2;
	              //for( size_t j1 = j2; j1 < j2+N2; j1+=4 ){ 
	              //  size_t N1 = 4; if( j1+4 >= j2+N2 ){ N1 = j2+N2 - j1; }

	              //  // On L2-Cache Copy for B
	              //  block2d_info_t infoB = {k2,j1,K2,N1,MYBLAS_TILE_K,MYBLAS_TILE_N};
	              //  myblas_dgemm_copy_B(B,ldb,B1,&infoB);

	              //  // L1 cache
	              //  block3d_info_t info3d = {M2,N1,K2,MYBLAS_TILE_M,MYBLAS_TILE_N,MYBLAS_TILE_K};
	              //  myblas_dgemm_kernel_AB(alpha,A2,B1,C+ldc*j1+i2,ldc,&info3d);

	              //  B1 += K2*N1;
	              //}

	            }else{ 
	              m2--;
	              i2 += M2;
	              M2 = MYBLAS_BLOCK_M; if( i2+M2 >= i3+M3 ){ M2=M3R; }

	              //printf("m2=%d n2=%d  A(%d,%d)[%d,%d] x B(%d,%d)[%d,%d] = C(%d,%d)[%d,%d]\n",m2,n2,i2,k2,M2,K2,k2,j2,K2,N2,i2,j2,M2,N2); 

	              // On L2-Cache Copy for A
	              block2d_info_t infoA = {k2,i2,K2,M2,MYBLAS_TILE_K,MYBLAS_TILE_M,COPY_MK};
	              myblas_dgemm_copy_A(A,lda,A2,&infoA);

	              // L1 cache
	              block3d_info_t info3d = {M2,N2,K2,MYBLAS_TILE_M,MYBLAS_TILE_N,MYBLAS_TILE_K};
	              myblas_dgemm_kernel_AB(alpha,A2,B2,C+ldc*j2+i2,ldc,&info3d);

	              //double* A1 = A2;
	              //for( size_t i1 = i2; i1 < i2+M2; i1+=4 ){ 
	              //  size_t M1 = 4; if( i1+4 >= i2+M2 ){ M1 = i2+M2 - i1; }

	              //  // On L2-Cache Copy for A
	              //  block2d_info_t infoA = {k2,i1,K2,M1,MYBLAS_TILE_K,MYBLAS_TILE_M};
	              //  myblas_dgemm_copy_A(A,lda,A1,&infoA);

	              //  // L1 cache
	              //  block3d_info_t info3d = {M1,N2,K2,MYBLAS_TILE_M,MYBLAS_TILE_N,MYBLAS_TILE_K};
	              //  myblas_dgemm_kernel_AB(alpha,A1,B2,C+ldc*j2+i1,ldc,&info3d);

	              //  A1 += K2*M1;
	              //}

	              n2 = N3B-1; 
	            }

	          }
	          k2 += K2;
	        }

	    }}}

	    free(A2);
	    free(B2);

	}//openmp

}
