#include "myblas_internal.h"
#include "kernel_test.h"
#include <stdio.h>

#define  KERNEL(MU,NU,KU,LU) \
	            for( size_t k=0; k<KU; k++ ){\
	              for( size_t j=0; j<NU; j++ ){\
	                for( size_t i=0; i<MU; i++ ){\
  	                   for( size_t l=0; l<LU; l++ ){\
	                     c[i+MU*j] += (*(A2+l+i*LU+k*LU*MU))*(*(B2+l+j*LU+k*LU*NU));\
	                   }\
	                }\
	              }\
	            }\
	            A2+=MU*KU*LU;\
	            B2+=KU*LU*NU;

#define  CSTORE(MU,NU) \
	        for( size_t j=0; j<NU; j++ ){\
	          for( size_t i=0; i<MU; i++ ){\
	            *(C+i+j*ldc) += alpha*c[i+MU*j];\
	          }\
	        }\
	        B2 = B2 - NU*K;\
	        C+=MU;

#define KERNEL_K4(MU,NU) \
 	        for( size_t ii=0; ii<MU*NU; ii++ ){ c[ii]=0e0; }\
	        if( K >> 2 ){\
	          size_t k4 = ( K >> 2 );\
	          while( k4-- ){\
	            KERNEL(MU,NU,2,2);\
	          }\
	        }\
	        if( K & 2 ){\
	            KERNEL(MU,NU,1,2);\
	        }\
	        if( K & 1 ){\
	            KERNEL(MU,NU,1,1);\
	        }\
	        CSTORE(MU,NU);

#define KERNEL_K8(MU,NU) \
 	        for( size_t ii=0; ii<MU*NU; ii++ ){ c[ii]=0e0; }\
	        if( K >> 3 ){\
	          size_t k8 = ( K >> 3 );\
	          while( k8-- ){\
	            KERNEL(MU,NU,4,2);\
	          }\
	        }\
	        if( K & 4 ){\
	            KERNEL(MU,NU,2,2);\
	        }\
	        if( K & 2 ){\
	            KERNEL(MU,NU,1,2);\
	        }\
	        if( K & 1 ){\
	            KERNEL(MU,NU,1,1);\
	        }\
	        CSTORE(MU,NU);

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

	// 4x2x8

	size_t M     = info->M2    ;
	size_t N     = info->N2    ;
	size_t K     = info->K2    ;

	double c[4*6];

	size_t NQ = N/6;
	size_t NR = N%6;

	if( NQ ){
	  size_t n6 = NQ;
	  while( n6-- ){
	    
	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        KERNEL_K8(4,6);

	      }

	    }
	    if( M & 2 ){

	        KERNEL_K8(2,6);

	    }
	    if( M & 1 ){

	        KERNEL_K8(1,6);

	    }
	    A2 = A2 - M*K;
	    B2 = B2 + 6*K;
	    C  = C - M + 6*ldc;
	  }
	
	}
	if( NR & 4 ){
	    
	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        KERNEL_K8(4,4);

	      }

	    }
	    if( M & 2 ){

	        KERNEL_K8(2,4);

	    }
	    if( M & 1 ){

	        KERNEL_K8(1,4);

	    }
	    A2 = A2 - M*K;
	    B2 = B2 + 4*K;
	    C  = C - M + 4*ldc;

	}
	if( NR & 2 ){

	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        KERNEL_K8(4,2);

	      }

	    }
	    if( M & 2 ){

	        KERNEL_K8(2,2);

	    }
	    if( M & 1 ){

	        KERNEL_K8(1,2);

	    }

	    A2 = A2 - M*K;
	    B2 = B2 + 2*K;
	    C  = C - M + 2*ldc;
	}
	if( NR & 1 ){
	   
	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        KERNEL_K8(4,1);

	      }

	    }
	    if( M & 2 ){

	        KERNEL_K8(2,1);

	    }
	    if( M & 1 ){

	        KERNEL_K8(1,1);

	    }

	    A2 = A2 - M*K;
	    B2 = B2 + 1*K;
	    C  = C - M + 1*ldc;
	}

	A2 = A2 + M*K;
	B2 = B2 - K*N;
	C  = C - ldc*N + M;

}

