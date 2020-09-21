#include "myblas_internal.h"

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


void myblas_dgemm_copy_n_MxK(size_t K, size_t N, const double* B, size_t k, size_t j,  size_t ldb, double* B2 ){

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

void myblas_dgemm_copy_n_NxK(size_t K, size_t N, const double* B, size_t k, size_t j,  size_t ldb, double* B2 ){

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


