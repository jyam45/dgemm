#include "myblas_internal.h"

#define  COPY_T(MU,KU,LU,A2) \
	        for( size_t k=0; k<KU; k++ ){\
	          for( size_t i=0; i<MU; i++ ){\
	            for( size_t l=0; l<LU; l++ ){\
	              (*A2) = *(A+i+l*lda+k*2*lda);\
	              A2++;\
	            }\
	          }\
	        }\
	        A  = A  + MU;

#define  COPY_T_M4(KU,LU) \
	    if( M >> 2 ){\
	      size_t m4 = ( M >> 2 );\
	      while( m4-- ){\
	        COPY_T(4,KU,LU,A2);\
	        A2 = A2 - 4*KU*LU + 4*K;\
	      }\
	      A2 = A2 - MM*K + 4*KU*LU; \
	    }\
	    if( M & 2 ){\
	        COPY_T(2,KU,LU,A2_2);\
	    }\
	    if( M & 1 ){\
	        COPY_T(1,KU,LU,A2_1);\
	    }\
	    A  = A - M + KU*LU*lda;

#define  COPY_T_M6(KU,LU) \
	    if( MQ ){\
	      size_t m6 = MQ;\
	      while( m6-- ){\
	        COPY_T(6,KU,LU,A2);\
	        A2 = A2 - 6*KU*LU + 6*K;\
	      }\
	      A2 = A2 - MM*K + 6*KU*LU; \
	    }\
	    if( MR & 4 ){\
	        COPY_T(4,KU,LU,A2_4);\
	    }\
	    if( MR & 2 ){\
	        COPY_T(2,KU,LU,A2_2);\
	    }\
	    if( MR & 1 ){\
	        COPY_T(1,KU,LU,A2_1);\
	    }\
	    A  = A - M + KU*LU*lda;



void myblas_dgemm_copy_t_MxK(size_t K, size_t M, const double* A, size_t k1, size_t i1, size_t lda, double* A2 ){

	A = A + i1 + k1*lda;

	double* A0 = A2;
	double* A2_2 = A2   + K*( M & ~3 );
	double* A2_1 = A2_2 + K*( M &  2 );;
	size_t  MM = ( M & ~3 ); // size M for Main Loop 

	if( K >> 3 ){
	  size_t k8 = ( K >> 3 );
	  while( k8-- ){

	    COPY_T_M4(4,2);

	  }
	}
	if( K & 4  ){

	    COPY_T_M4(2,2);

	}
	if( K & 2  ){

	    COPY_T_M4(1,2);

	}
	if( K & 1  ){

	    COPY_T_M4(1,1);

	}

}

void myblas_dgemm_copy_t_NxK(size_t K, size_t M, const double* A, size_t k1, size_t i1, size_t lda, double* A2 ){

	A = A + i1 + k1*lda;

	size_t MQ = M/6;
	size_t MR = M%6;
	size_t MM = M-MR;

	double* A2_4 = A2   + K*MM;
	double* A2_2 = A2_4 + K*( MR & 4 );
	double* A2_1 = A2_2 + K*( MR & 2 );;

	if( K >> 3 ){
	  size_t k8 = ( K >> 3 );
	  while( k8-- ){

	    COPY_T_M6(4,2);

	  }
	}
	if( K & 4  ){

	    COPY_T_M6(2,2);

	}
	if( K & 2  ){

	    COPY_T_M6(1,2);

	}
	if( K & 1  ){

	    COPY_T_M6(1,1);

	}

}

