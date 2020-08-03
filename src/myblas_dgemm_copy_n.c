#include "myblas_internal.h"

// On L2-Cache Copy for B
void myblas_dgemm_copy_n(const double* B, size_t ldb, double* B2, const block2d_info_t* info ){

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

	    block2d_info_t tile = { 0, 0, K1, N1 };
	    myblas_dgemm_copy_n_core( B, ldb, B2, &tile );

	    B = B + N1*ldb;
	    B2= B2+ N1*K1;
	  }
	  B  = B  - ldb *N2 + K1;
	}

}


void myblas_dgemm_copy_n_core(const double* B, size_t ldb, double* B2, const block2d_info_t* info ){

	size_t k      = info->i2    ;
	size_t j      = info->j2    ;
	size_t K1     = info->M2    ;
	size_t N1     = info->N2    ;

	double ymm0 ,ymm1 ,ymm2 ,ymm3 ;
	double ymm4 ,ymm5 ,ymm6 ,ymm7 ;
	double ymm8 ,ymm9 ,ymm10,ymm11;
	double ymm12,ymm13,ymm14,ymm15;

	B = B + k + ldb*j; // start point

	size_t n = N1;
	if( n >> 2 ){
	  size_t n4 = ( n >> 2 ); // unrolling with 8 elements
	  while( n4-- )
	  {
	    size_t k = K1;
	    if( k >> 4 ){
	      size_t k16 = ( k >> 4 ); // unrolling with 16 elements
	      while( k16-- ){
	        size_t k8a = 4; // Vector-lenght of YMM
	        size_t k8b = 4; // Vector-lenght of YMM
	        size_t k8c = 4; // Vector-lenght of YMM
	        size_t k8d = 4; // Vector-lenght of YMM
	        while( k8a-- ){
	          ymm0  = *(B +0*ldb);
	          ymm1  = *(B +1*ldb);
	          ymm2  = *(B +2*ldb);
	          ymm3  = *(B +3*ldb);
	          *(B2+0*K1) = ymm0 ;
	          *(B2+1*K1) = ymm1 ;
	          *(B2+2*K1) = ymm2 ;
	          *(B2+3*K1) = ymm3 ;
	          B++;
	          B2++;
	        }
	        while( k8b-- ){
	          ymm4  = *(B +0*ldb);
	          ymm5  = *(B +1*ldb);
	          ymm6  = *(B +2*ldb);
	          ymm7  = *(B +3*ldb);
	          *(B2+0*K1) = ymm4 ;
	          *(B2+1*K1) = ymm5 ;
	          *(B2+2*K1) = ymm6 ;
	          *(B2+3*K1) = ymm7 ;
	          B++;
	          B2++;
	        }
	        while( k8c-- ){
	          ymm8  = *(B +0*ldb);
	          ymm9  = *(B +1*ldb);
	          ymm10 = *(B +2*ldb);
	          ymm11 = *(B +3*ldb);
	          *(B2+0*K1) = ymm8 ;
	          *(B2+1*K1) = ymm9 ;
	          *(B2+2*K1) = ymm10;
	          *(B2+3*K1) = ymm11;
	          B++;
	          B2++;
	        }
	        while( k8d-- ){
	          ymm12 = *(B +0*ldb);
	          ymm13 = *(B +1*ldb);
	          ymm14 = *(B +2*ldb);
	          ymm15 = *(B +3*ldb);
	          *(B2+0*K1) = ymm12;
	          *(B2+1*K1) = ymm13;
	          *(B2+2*K1) = ymm14;
	          *(B2+3*K1) = ymm15;
	          B++;
	          B2++;
	        }

	      }
	    }
	    //if( k >> 3 ){
	    //  size_t k8 = ( k >> 3 ); // unrolling with 8 elements
	    //  while( k8-- ){
	    if( k & 8 ){
	      {
	        size_t k8a = 4; // Vector-lenght of YMM
	        size_t k8b = 4; // Vector-lenght of YMM
	        while( k8a-- ){
	          ymm0  = *(B +0*ldb);
	          ymm1  = *(B +1*ldb);
	          ymm2  = *(B +2*ldb);
	          ymm3  = *(B +3*ldb);
	          *(B2+0*K1) = ymm0 ;
	          *(B2+1*K1) = ymm1 ;
	          *(B2+2*K1) = ymm2 ;
	          *(B2+3*K1) = ymm3 ;
	          B++;
	          B2++;
	        }
	        while( k8b-- ){
	          ymm4  = *(B +0*ldb);
	          ymm5  = *(B +1*ldb);
	          ymm6  = *(B +2*ldb);
	          ymm7  = *(B +3*ldb);
	          *(B2+0*K1) = ymm4 ;
	          *(B2+1*K1) = ymm5 ;
	          *(B2+2*K1) = ymm6 ;
	          *(B2+3*K1) = ymm7 ;
	          B++;
	          B2++;
	        }
	      }
	    }
	    //if( k >> 2 ){
	    //  size_t k4 = ( k >> 2 ); // unrolling with 8 elements
	    //  while( k4-- ){
	    if( k & 4 ){
	        size_t k8a = 4; // Vector-lenght of YMM
	        while( k8a-- ){
	          ymm0  = *(B +0*ldb);
	          ymm1  = *(B +1*ldb);
	          ymm2  = *(B +2*ldb);
	          ymm3  = *(B +3*ldb);
	          *(B2+0*K1) = ymm0 ;
	          *(B2+1*K1) = ymm1 ;
	          *(B2+2*K1) = ymm2 ;
	          *(B2+3*K1) = ymm3 ;
	          B++;
	          B2++;
	        }
	    }
	    if( k & 3 ){
	      size_t kr = ( k & 3 ); // unrolling with 8 elements
	      while( kr-- ){
	        ymm0  = *(B +0*ldb);
	        ymm1  = *(B +1*ldb);
	        ymm2  = *(B +2*ldb);
	        ymm3  = *(B +3*ldb);
	        *(B2+0*K1) = ymm0 ;
	        *(B2+1*K1) = ymm1 ;
	        *(B2+2*K1) = ymm2 ;
	        *(B2+3*K1) = ymm3 ;
	        B++;
	        B2++;
	      }
	    }
	    B  = B  - K1 + 4*ldb ;
	    B2 = B2 - K1 + 4*K1;
	  }
	}
	//if( n & 2 ){
	////if( n >> 1 ){
	////  size_t n2 = ( n >> 1 ); // unrolling with 8 elements
	////  while( n2-- )
	//  {
	//    size_t k = K1;
	//    if( k >> 4 ){
	//      size_t k16 = ( k >> 4 ); // unrolling with 16 elements
	//      while( k16-- ){
	//        size_t k8a = 4; // Vector-lenght of YMM
	//        size_t k8b = 4; // Vector-lenght of YMM
	//        size_t k8c = 4; // Vector-lenght of YMM
	//        size_t k8d = 4; // Vector-lenght of YMM
	//        while( k8a-- ){
	//          ymm0  = *(B +0*ldb);
	//          ymm1  = *(B +1*ldb);
	//          *(B2+0*K1) = ymm0 ;
	//          *(B2+1*K1) = ymm1 ;
	//          B++;
	//          B2++;
	//        }
	//        while( k8b-- ){
	//          ymm4  = *(B +0*ldb);
	//          ymm5  = *(B +1*ldb);
	//          *(B2+0*K1) = ymm4 ;
	//          *(B2+1*K1) = ymm5 ;
	//          B++;
	//          B2++;
	//        }
	//        while( k8c-- ){
	//          ymm8  = *(B +0*ldb);
	//          ymm9  = *(B +1*ldb);
	//          *(B2+0*K1) = ymm8 ;
	//          *(B2+1*K1) = ymm9 ;
	//          B++;
	//          B2++;
	//        }
	//        while( k8d-- ){
	//          ymm12 = *(B +0*ldb);
	//          ymm13 = *(B +1*ldb);
	//          *(B2+0*K1) = ymm12;
	//          *(B2+1*K1) = ymm13;
	//          B++;
	//          B2++;
	//        }

	//      }
	//    }
	//    //if( k >> 3 ){
	//    //  size_t k8 = ( k >> 3 ); // unrolling with 8 elements
	//    //  while( k8-- ){
	//    if( k & 8 ){
	//      {
	//        size_t k8a = 4; // Vector-lenght of YMM
	//        size_t k8b = 4; // Vector-lenght of YMM
	//        while( k8a-- ){
	//          ymm0  = *(B +0*ldb);
	//          ymm1  = *(B +1*ldb);
	//          *(B2+0*K1) = ymm0 ;
	//          *(B2+1*K1) = ymm1 ;
	//          B++;
	//          B2++;
	//        }
	//        while( k8b-- ){
	//          ymm4  = *(B +0*ldb);
	//          ymm5  = *(B +1*ldb);
	//          *(B2+0*K1) = ymm4 ;
	//          *(B2+1*K1) = ymm5 ;
	//          B++;
	//          B2++;
	//        }
	//      }
	//    }
	//    //if( k >> 2 ){
	//    //  size_t k4 = ( k >> 2 ); // unrolling with 8 elements
	//    //  while( k4-- ){
	//    if( k & 4 ){
	//        size_t k8a = 4; // Vector-lenght of YMM
	//        while( k8a-- ){
	//          ymm0  = *(B +0*ldb);
	//          ymm1  = *(B +1*ldb);
	//          *(B2+0*K1) = ymm0 ;
	//          *(B2+1*K1) = ymm1 ;
	//          B++;
	//          B2++;
	//        }
	//    }
	//    if( k & 3 ){
	//      size_t kr = ( k & 3 ); // unrolling with 8 elements
	//      while( kr-- ){
	//        ymm0  = *(B +0*ldb);
	//        ymm1  = *(B +1*ldb);
	//        *(B2+0*K1) = ymm0 ;
	//        *(B2+1*K1) = ymm1 ;
	//        B++;
	//        B2++;
	//      }
	//    }
	//    B  = B  - K1 + 2*ldb ;
	//    B2 = B2 - K1 + 2*K1;
	//  }
	//}

	if( n & 3 ){
	  size_t nr = ( n & 3 );
	  while( nr-- ){
	    size_t k = K1;
	    while( k-- ){
	      *B2=*B;
	      B++;
	      B2++;
	    }
	    B  = B  - K1 + ldb ;
	  }

	}


}


