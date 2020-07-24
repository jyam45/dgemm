#include "myblas_internal.h"

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

	        for( size_t j=0; j<N; j++ ){
	            for( size_t i=0; i<M; i++ ){
			AB=0e0;
	                for( size_t k=0; k<K; k++ ){
	                   AB = AB + (*A)*(*B);
	                   A += lda;
	                   B++;
	                }
			*C=beta*(*C) + alpha*AB;
	                A = A - lda*K + 1;
	                B = B - K;
	                C++;
	            }
	            A = A - M;
	            B = B + ldb;
	            C = C - M + ldc;
	        }

	    }
	}
}
