#include "cblas.h"

#ifdef __cplusplus
extern "C" {
#endif


void myblas_xerbla( const char* name, int info );


void myblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                  const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                  const int K, const double alpha, const double  *A,
                  const int lda, const double  *B, const int ldb,
                  const double beta, double  *C, const int ldc);



#ifdef __cplusplus
}
#endif
