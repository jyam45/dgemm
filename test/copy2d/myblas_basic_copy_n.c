#include "copy2d_test.h"

#define COPY_N(NU,KU,LU) \
	        for( size_t k=0; k<KU; k++ ){\
	          for( size_t j=0; j<NU; j++ ){\
	            for( size_t l=0; l<LU; l++ ){\
	              *B2=*(B+l+k*LU+j*ldb);\
	              B2++;\
	            }\
	          }\
	        }\
	        B+=KU*LU;

#define COPY_N_K4(NU) \
	    if( K >> 2 ){\
	      size_t k4 = ( K >> 2 );\
	      while( k4-- ){\
	        COPY_N(NU,2,2);\
	      }\
	    }\
	    if( K & 2 ){\
	        COPY_N(NU,1,2);\
	    }\
	    if( K & 1 ){\
	        COPY_N(NU,1,1);\
	    }\
	    B  = B  - K + NU*ldb ;


#define COPY_N_K8(NU) \
	    if( K >> 3 ){\
	      size_t k8 = ( K >> 3 );\
	      while( k8-- ){\
	        COPY_N(NU,4,2);\
	      }\
	    }\
	    if( K & 4 ){\
	        COPY_N(NU,2,2);\
	    }\
	    if( K & 2 ){\
	        COPY_N(NU,1,2);\
	    }\
	    if( K & 1 ){\
	        COPY_N(NU,1,1);\
	    }\
	    B  = B  - K + NU*ldb ;


void myblas_basic_copy_n_MxK(size_t K, size_t N, const double* B, size_t k, size_t j,  size_t ldb, double* B2 );
void myblas_basic_copy_n_NxK(size_t K, size_t N, const double* B, size_t k, size_t j,  size_t ldb, double* B2 );

static copy_detail_func_t copy_n_detail[]={ myblas_basic_copy_n_MxK, myblas_basic_copy_n_NxK };

// On L2-Cache Copy for B
void myblas_basic_copy_n(const double* B, size_t ldb, double* B2, const block2d_info_t* info ){

	size_t k2     = info->i2    ;
	size_t j2     = info->j2    ;
	size_t K2     = info->M2    ;
	size_t N2     = info->N2    ;
	size_t tile_K = info->tile_M;
	size_t tile_N = info->tile_N;

	size_t NQ = N2/tile_N;
	size_t NR = N2%tile_N;
	size_t KQ = K2/tile_K;
	size_t KR = K2%tile_K;

	copy_detail_func_t myblas_basic_copy_n_detail = copy_n_detail[info->type];

	B = B + k2 + ldb*j2; // start point

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

	    myblas_basic_copy_n_detail( K1, N1, B, 0, 0, ldb, B2 );

	    B  = B  + N1*ldb;
	    B2 = B2 + N1*K1;

	  }
	  B  = B  - ldb *N2 + K1;
	}

}

void myblas_basic_copy_n_core(const double* B, size_t ldb, double* B2, const block2d_info_t* info ){

	copy_detail_func_t myblas_basic_copy_n_detail = copy_n_detail[info->type];

	myblas_basic_copy_n_detail( info->M2, info->N2, B, info->i2, info->j2, ldb, B2 );

}

void myblas_basic_copy_n_MxK(size_t K, size_t N, const double* B, size_t k, size_t j,  size_t ldb, double* B2 ){

	B = B + k + ldb*j; // start point

	if( N >> 2 ){
	  size_t n4 = ( N >> 2 );
	  while( n4-- ){

	    COPY_N_K8(4);

	  }
	}
	if( N & 2 ){

	    COPY_N_K8(2);

	}
	if( N & 1 ){

	    COPY_N_K8(1);

	}

}

void myblas_basic_copy_n_NxK(size_t K, size_t N, const double* B, size_t k, size_t j,  size_t ldb, double* B2 ){

	B = B + k + ldb*j; // start point

	size_t NQ = N/6;
	size_t NR = N%6;

	if( NQ ){
	  size_t n6 = NQ;
	  while( n6-- ){

	    COPY_N_K8(6);

	  }
	}
	if( NR & 4 ){

	    COPY_N_K8(4);

	}
	if( NR & 2 ){

	    COPY_N_K8(2);

	}
	if( NR & 1 ){

	    COPY_N_K8(1);

	}

}

