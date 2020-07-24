#include "myblas.h"
#include "myblas_internal.h"

void myblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                  const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                  const int K, const double alpha, const double  *A,
                  const int lda, const double  *B, const int ldb,
                  const double beta, double  *C, const int ldc)
{
	gemm_args_t args={0,0,0,0,0,0e0,NULL,0,NULL,0,0e0,NULL,0};
	
	int info = 0;

	if( Order == CblasColMajor ){

		// Transpose Set-up

		if( TransA == CblasNoTrans  ){ args.TransA = 0; } 
		if( TransA == CblasTrans    ){ args.TransA = MASK_TRANS; } 
		if( TransA == CblasConjTrans){ args.TransA = MASK_TRANS | MASK_CONJ; } 

		if( TransB == CblasNoTrans  ){ args.TransB = 0; } 
		if( TransB == CblasTrans    ){ args.TransB = MASK_TRANS; } 
		if( TransB == CblasConjTrans){ args.TransB = MASK_TRANS | MASK_CONJ; } 

		// Error Check

		if( C    == NULL ) info=13;
		if( B    == NULL ) info=10;
		if( A    == NULL ) info= 8;
		if( info ){ myblas_xerbla("myblas_dgemm",info); return; }

		int ma = ( (args.TransA & MASK_TRANS) ? K : M );
		int mb = ( (args.TransB & MASK_TRANS) ? N : K );
		if( ldc  < M     ) info=14;
		if( ldb  < mb    ) info=11;
		if( lda  < ma    ) info= 9;
		if( K    < 0     ) info= 6;
		if( N    < 0     ) info= 5;
		if( M    < 0     ) info= 4;
		if( info ){ myblas_xerbla("myblas_dgemm",info); return; }

		// No Computing 

		if( M    == 0    ) return;
		if( N    == 0    ) return;
		if( K    == 0   && beta == 1e0  ) return;
		if( alpha== 0e0 && beta == 1e0  ) return;

		// Set arguments

		args.M     = M;
		args.N     = N;
		args.K     = K;
		args.alpha = alpha;
		args.A     = A;
		args.lda   = lda;
		args.B     = B;
		args.ldb   = ldb;
		args.beta  = beta;
		args.C     = C;
		args.ldc   = ldc;
	
	}else if( Order == CblasRowMajor ){

		// Transpose Set-up

		if( TransA == CblasNoTrans  ){ args.TransB = 0; } 
		if( TransA == CblasTrans    ){ args.TransB = MASK_TRANS; } 
		if( TransA == CblasConjTrans){ args.TransB = MASK_TRANS | MASK_CONJ; } 

		if( TransB == CblasNoTrans  ){ args.TransA = 0; } 
		if( TransB == CblasTrans    ){ args.TransA = MASK_TRANS; } 
		if( TransB == CblasConjTrans){ args.TransA = MASK_TRANS | MASK_CONJ; } 

		// Error Check

		if( C    == NULL ) info=13;
		if( B    == NULL ) info=10;
		if( A    == NULL ) info= 8;
		if( info ){ myblas_xerbla("myblas_dgemm",info); return; }

		int ma = ( (args.TransB & MASK_TRANS) ? M : K );
		int mb = ( (args.TransA & MASK_TRANS) ? K : N );
		if( ldc  < N     ) info=14;
		if( ldb  < mb    ) info=11;
		if( lda  < ma    ) info= 9;
		if( K    < 0     ) info= 6;
		if( N    < 0     ) info= 5;
		if( M    < 0     ) info= 4;
		if( info ){ myblas_xerbla("myblas_dgemm",info); return; }

		// No Computing 

		if( M    == 0    ) return;
		if( N    == 0    ) return;
		if( K    == 0   && beta == 1e0  ) return;
		if( alpha== 0e0 && beta == 1e0  ) return;

		// Set arguments

		args.M     = N;
		args.N     = M;
		args.K     = K;
		args.alpha = alpha;
		args.A     = B;
		args.lda   = ldb;
		args.B     = A;
		args.ldb   = lda;
		args.beta  = beta;
		args.C     = C;
		args.ldc   = ldc;
	
	}

	myblas_dgemm_main( &args );

}
