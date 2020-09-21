#include "myblas_internal.h"

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


void myblas_dgemm_copy_t_MxK(size_t K, size_t M, const double* A, size_t k1, size_t i1, size_t lda, double* A2 ){

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

void myblas_dgemm_copy_t_NxK(size_t K, size_t M, const double* A, size_t k1, size_t i1, size_t lda, double* A2 ){

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

