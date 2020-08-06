#include "myblas_internal.h"

void myblas_dgemm_scale2d(double beta, double *C, size_t ldc, const block2d_info_t* info ){

	myblas_dgemm_scale2d_detail( info->M2, info->N2, beta, C, ldc );

}

