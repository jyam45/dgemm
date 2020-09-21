#include "copy2d_test.h"
#include <stdlib.h>

#define  COPY_T(MU,KU,LU) \
	        for( size_t k=0; k<KU; k++ ){\
	          for( size_t i=0; i<MU; i++ ){\
	            for( size_t l=0; l<LU; l++ ){\
	              (*A2) = *(A+i+l*lda+k*LU*lda);\
	              A2++;\
	            }\
	          }\
	        }\
	        A += KU*LU*lda ;

#define  COPY_T_K4(MU) \
	    if( K >> 2 ){\
	      size_t k4 = ( K >> 2 );\
	      while( k4-- ){\
	        COPY_T(MU,2,2);\
	      }\
	    }\
	    if( K & 2 ){\
	        COPY_T(MU,1,2);\
	    }\
	    if( K & 1 ){\
	        COPY_T(MU,1,1);\
	    }\
	    A  = A  - lda *K + MU;


#define  COPY_T_K8(MU) \
	    if( K >> 3 ){\
	      size_t k8 = ( K >> 3 );\
	      while( k8-- ){\
	        COPY_T(MU,4,2);\
	      }\
	    }\
	    if( K & 4 ){\
	        COPY_T(MU,2,2);\
	    }\
	    if( K & 2 ){\
	        COPY_T(MU,1,2);\
	    }\
	    if( K & 1 ){\
	        COPY_T(MU,1,1);\
	    }\
	    A  = A  - lda *K + MU;


void myblas_basic_copy_t_MxK(size_t K, size_t M, const double* A, size_t k1, size_t i1, size_t lda, double* A2 );
void myblas_basic_copy_t_NxK(size_t K, size_t M, const double* A, size_t k1, size_t i1, size_t lda, double* A2 );

static copy_detail_func_t copy_t_detail[]={ myblas_basic_copy_t_MxK, myblas_basic_copy_t_NxK };

// On L2-Cache Copy for A
void myblas_basic_copy_t(const double* A, size_t lda, double* A2, const block2d_info_t* info ){

	size_t k2     = info->i2    ;
	size_t i2     = info->j2    ;
	size_t K2     = info->M2    ;
	size_t M2     = info->N2    ;
	size_t tile_K = info->tile_M;
	size_t tile_M = info->tile_N;

	size_t MQ = M2/tile_M;
	size_t MR = M2%tile_M;
	size_t KQ = K2/tile_K;
	size_t KR = K2%tile_K;

	copy_detail_func_t myblas_basic_copy_t_detail = copy_t_detail[info->type];

	A = A + lda*k2 + i2; // start point

	if( MR >  0 ){ MQ++; }
	if( MR == 0 ){ MR = tile_M; }
	if( KR >  0 ){ KQ++; }
	if( KR == 0 ){ KR = tile_K; }

	// L1-cache blocking
	size_t k1 = KQ;
	while( k1-- ){
	  size_t K1 = tile_K; if( k1==0 ){ K1=KR; }
	  size_t m1 = MQ;
	  while( m1-- ){
	    size_t M1 = tile_M; if( m1==0 ){ M1=MR; }

	    myblas_basic_copy_t_detail( K1, M1, A, 0, 0, lda, A2 );

	    A  = A  + M1;
	    A2 = A2 + M1*K1;
	  }
	  A = A  - M2 + lda *K1;
	}

}


void myblas_basic_copy_t_core(const double* A, size_t lda, double* A2, const block2d_info_t* info ){

	copy_detail_func_t myblas_basic_copy_t_detail = copy_t_detail[info->type];

	myblas_basic_copy_t_detail( info->M2, info->N2, A, info->i2, info->j2, lda, A2 );

}

void myblas_basic_copy_t_MxK(size_t K, size_t M, const double* A, size_t k1, size_t i1, size_t lda, double* A2 ){

	A = A + i1 + k1*lda;

	if( M >> 2 ){
	  size_t m4 = ( M >> 2 );
	  while( m4-- ){

	    COPY_T_K8(4);

	  }
	}
	if( M & 2 ){

	    COPY_T_K8(2);

	}
	if( M & 1 ){

	    COPY_T_K8(1);

	}

}

void myblas_basic_copy_t_NxK(size_t K, size_t M, const double* A, size_t k1, size_t i1, size_t lda, double* A2 ){

	A = A + i1 + k1*lda;

	size_t MQ = M/6;
	size_t MR = M%6;

	if( MQ ){
	  size_t m6 = MQ;
	  while( m6-- ){

	    COPY_T_K8(6);

	  }
	}
	if( MR & 4 ){

	    COPY_T_K8(4);

	}
	if( MR & 2 ){

	    COPY_T_K8(2);

	}
	if( MR & 1 ){

	    COPY_T_K8(1);

	}

}


