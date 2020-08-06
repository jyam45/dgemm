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

	block2d_info_t tile = { 0,0,0,0,0,0 };

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

	    tile.M2 = K1; tile.N2 = N1;
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
	  while( n4-- ) {
	    size_t k = K1;
/*
	    if( k >> 3 ){
	      size_t k8 = ( k >> 3 ); // unrolling with 8 elements
	      while( k8-- ){
	          ymm0  = *(B+0+0*ldb); ymm4  = *(B+1+0*ldb); ymm8  = *(B+2+0*ldb); ymm12 = *(B+3+0*ldb);
	          ymm1  = *(B+0+1*ldb); ymm5  = *(B+1+1*ldb); ymm9  = *(B+2+1*ldb); ymm13 = *(B+3+1*ldb);
	          ymm2  = *(B+0+2*ldb); ymm6  = *(B+1+2*ldb); ymm10 = *(B+2+2*ldb); ymm14 = *(B+3+2*ldb);
	          ymm3  = *(B+0+3*ldb); ymm7  = *(B+1+3*ldb); ymm11 = *(B+2+3*ldb); ymm15 = *(B+3+3*ldb);
	          *(B2+0+0*8) = ymm0 ; *(B2+1+0*8) = ymm4 ; *(B2+2+0*8) = ymm8 ; *(B2+3+0*8) = ymm12;
	          *(B2+0+1*8) = ymm1 ; *(B2+1+1*8) = ymm5 ; *(B2+2+1*8) = ymm9 ; *(B2+3+1*8) = ymm13;
	          *(B2+0+2*8) = ymm2 ; *(B2+1+2*8) = ymm6 ; *(B2+2+2*8) = ymm10; *(B2+3+2*8) = ymm14;
	          *(B2+0+3*8) = ymm3 ; *(B2+1+3*8) = ymm7 ; *(B2+2+3*8) = ymm11; *(B2+3+3*8) = ymm15;
	          ymm0  = *(B+4+0*ldb); ymm4  = *(B+5+0*ldb); ymm8  = *(B+6+0*ldb); ymm12 = *(B+7+0*ldb);
	          ymm1  = *(B+4+1*ldb); ymm5  = *(B+5+1*ldb); ymm9  = *(B+6+1*ldb); ymm13 = *(B+7+1*ldb);
	          ymm2  = *(B+4+2*ldb); ymm6  = *(B+5+2*ldb); ymm10 = *(B+6+2*ldb); ymm14 = *(B+7+2*ldb);
	          ymm3  = *(B+4+3*ldb); ymm7  = *(B+5+3*ldb); ymm11 = *(B+6+3*ldb); ymm15 = *(B+7+3*ldb);
	          *(B2+4+0*8) = ymm0 ; *(B2+5+0*8) = ymm4 ; *(B2+6+0*8) = ymm8 ; *(B2+7+0*8) = ymm12;
	          *(B2+4+1*8) = ymm1 ; *(B2+5+1*8) = ymm5 ; *(B2+6+1*8) = ymm9 ; *(B2+7+1*8) = ymm13;
	          *(B2+4+2*8) = ymm2 ; *(B2+5+2*8) = ymm6 ; *(B2+6+2*8) = ymm10; *(B2+7+2*8) = ymm14;
	          *(B2+4+3*8) = ymm3 ; *(B2+5+3*8) = ymm7 ; *(B2+6+3*8) = ymm11; *(B2+7+3*8) = ymm15;
	          B+=8;
	          B2+=32;
	      }
	    }
	    if( k & 4 ){
*/
	    if( k >> 2 ){
	      size_t k4 = ( k >> 2 ); // unrolling with 8 elements
	      while( k4-- ){
	          ymm0  = *(B+0+0*ldb); ymm4  = *(B+1+0*ldb); ymm8  = *(B+2+0*ldb); ymm12 = *(B+3+0*ldb);
	          ymm1  = *(B+0+1*ldb); ymm5  = *(B+1+1*ldb); ymm9  = *(B+2+1*ldb); ymm13 = *(B+3+1*ldb);
	          ymm2  = *(B+0+2*ldb); ymm6  = *(B+1+2*ldb); ymm10 = *(B+2+2*ldb); ymm14 = *(B+3+2*ldb);
	          ymm3  = *(B+0+3*ldb); ymm7  = *(B+1+3*ldb); ymm11 = *(B+2+3*ldb); ymm15 = *(B+3+3*ldb);
	          *(B2+0+0*4) = ymm0 ; *(B2+1+0*4) = ymm4 ; *(B2+2+0*4) = ymm8 ; *(B2+3+0*4) = ymm12;
	          *(B2+0+1*4) = ymm1 ; *(B2+1+1*4) = ymm5 ; *(B2+2+1*4) = ymm9 ; *(B2+3+1*4) = ymm13;
	          *(B2+0+2*4) = ymm2 ; *(B2+1+2*4) = ymm6 ; *(B2+2+2*4) = ymm10; *(B2+3+2*4) = ymm14;
	          *(B2+0+3*4) = ymm3 ; *(B2+1+3*4) = ymm7 ; *(B2+2+3*4) = ymm11; *(B2+3+3*4) = ymm15;
	          B+=4;
	          B2+=16;
	      }
	    }
	    if( k & 2 ){
	        ymm0  = *(B+0+0*ldb); ymm4  = *(B+1+0*ldb);
	        ymm1  = *(B+0+1*ldb); ymm5  = *(B+1+1*ldb);
	        ymm2  = *(B+0+2*ldb); ymm6  = *(B+1+2*ldb);
	        ymm3  = *(B+0+3*ldb); ymm7  = *(B+1+3*ldb);
	        *(B2+0+0*2) = ymm0 ; *(B2+1+0*2) = ymm4 ;
	        *(B2+0+1*2) = ymm1 ; *(B2+1+1*2) = ymm5 ;
	        *(B2+0+2*2) = ymm2 ; *(B2+1+2*2) = ymm6 ;
	        *(B2+0+3*2) = ymm3 ; *(B2+1+3*2) = ymm7 ;
	        B+=2;
	        B2+=8;
	    }
	    if( k & 1 ){
	        ymm0  = *(B+0+0*ldb); 
	        ymm1  = *(B+0+1*ldb); 
	        ymm2  = *(B+0+2*ldb); 
	        ymm3  = *(B+0+3*ldb); 
	        *(B2+0+0*1) = ymm0 ;
	        *(B2+0+1*1) = ymm1 ;
	        *(B2+0+2*1) = ymm2 ;
	        *(B2+0+3*1) = ymm3 ;
	        B+=1;
	        B2+=4;
	    }
	    //if( k & 3 ){
	    //  size_t kr = ( k & 3 ); // unrolling with 8 elements
	    //  while( kr-- ){
	    //    ymm0  = *(B +0*ldb);
	    //    ymm1  = *(B +1*ldb);
	    //    ymm2  = *(B +2*ldb);
	    //    ymm3  = *(B +3*ldb);
	    //    *(B2+0) = ymm0 ;
	    //    *(B2+1) = ymm1 ;
	    //    *(B2+2) = ymm2 ;
	    //    *(B2+3) = ymm3 ;
	    //    B++;
	    //    B2+=4;
	    //  }
	    //}
	    B  = B  - K1 + 4*ldb ;
	    //B2 = B2 - K1 + 4*K1;
	  }
	}
	if( n & 2 ){

	    size_t k = K1;
/*
	    if( k >> 3 ){
	      size_t k8 = ( k >> 3 ); // unrolling with 8 elements
	      while( k8-- ){
	          ymm0  = *(B+0+0*ldb); ymm4  = *(B+1+0*ldb); ymm8  = *(B+2+0*ldb); ymm12 = *(B+3+0*ldb);
	          ymm1  = *(B+0+1*ldb); ymm5  = *(B+1+1*ldb); ymm9  = *(B+2+1*ldb); ymm13 = *(B+3+1*ldb);
	          *(B2+0+0*8) = ymm0 ; *(B2+1+0*8) = ymm4 ; *(B2+2+0*8) = ymm8 ; *(B2+3+0*8) = ymm12;
	          *(B2+0+1*8) = ymm1 ; *(B2+1+1*8) = ymm5 ; *(B2+2+1*8) = ymm9 ; *(B2+3+1*8) = ymm13;
	          ymm0  = *(B+4+0*ldb); ymm4  = *(B+5+0*ldb); ymm8  = *(B+6+0*ldb); ymm12 = *(B+7+0*ldb);
	          ymm1  = *(B+4+1*ldb); ymm5  = *(B+5+1*ldb); ymm9  = *(B+6+1*ldb); ymm13 = *(B+7+1*ldb);
	          *(B2+4+0*8) = ymm0 ; *(B2+5+0*8) = ymm4 ; *(B2+6+0*8) = ymm8 ; *(B2+7+0*8) = ymm12;
	          *(B2+4+1*8) = ymm1 ; *(B2+5+1*8) = ymm5 ; *(B2+6+1*8) = ymm9 ; *(B2+7+1*8) = ymm13;
	          B+=8;
	          B2+=16;
	      }
	    }
	    if( k & 4 ){
*/
	    if( k >> 2 ){
	      size_t k4 = ( k >> 2 ); // unrolling with 8 elements
	      while( k4-- ){
	          ymm0  = *(B+0+0*ldb); ymm4  = *(B+1+0*ldb); ymm8  = *(B+2+0*ldb); ymm12 = *(B+3+0*ldb);
	          ymm1  = *(B+0+1*ldb); ymm5  = *(B+1+1*ldb); ymm9  = *(B+2+1*ldb); ymm13 = *(B+3+1*ldb);
	          *(B2+0+0*4) = ymm0 ; *(B2+1+0*4) = ymm4 ; *(B2+2+0*4) = ymm8 ; *(B2+3+0*4) = ymm12;
	          *(B2+0+1*4) = ymm1 ; *(B2+1+1*4) = ymm5 ; *(B2+2+1*4) = ymm9 ; *(B2+3+1*4) = ymm13;
	          B+=4;
	          B2+=8;
	      }
	    }
	    if( k & 2 ){
	        ymm0  = *(B+0+0*ldb); ymm4  = *(B+1+0*ldb);
	        ymm1  = *(B+0+1*ldb); ymm5  = *(B+1+1*ldb);
	        *(B2+0+0*2) = ymm0 ; *(B2+1+0*2) = ymm4 ;
	        *(B2+0+1*2) = ymm1 ; *(B2+1+1*2) = ymm5 ;
	        B+=2;
	        B2+=4;
	    }
	    if( k & 1 ){
	        ymm0  = *(B+0+0*ldb); 
	        ymm1  = *(B+0+1*ldb); 
	        *(B2+0+0*1) = ymm0 ;
	        *(B2+0+1*1) = ymm1 ;
	        B+=1;
	        B2+=2;
	    }
	    B  = B  - K1 + 2*ldb ;
	    //B2 = B2 - K1 + 4*K1;

	}
	if( n & 1 ){

	    size_t k = K1;
/*
	    if( k >> 3 ){
	      size_t k8 = ( k >> 3 ); // unrolling with 8 elements
	      while( k8-- ){
	          ymm0  = *(B+0+0*ldb); ymm4  = *(B+1+0*ldb); ymm8  = *(B+2+0*ldb); ymm12 = *(B+3+0*ldb);
	          *(B2+0+0*8) = ymm0 ; *(B2+1+0*8) = ymm4 ; *(B2+2+0*8) = ymm8 ; *(B2+3+0*8) = ymm12;
	          ymm0  = *(B+4+0*ldb); ymm4  = *(B+5+0*ldb); ymm8  = *(B+6+0*ldb); ymm12 = *(B+7+0*ldb);
	          *(B2+4+0*8) = ymm0 ; *(B2+5+0*8) = ymm4 ; *(B2+6+0*8) = ymm8 ; *(B2+7+0*8) = ymm12;
	          B+=8;
	          B2+=8;
	      }
	    }
	    if( k & 4 ){
*/
	    if( k >> 2 ){
	      size_t k4 = ( k >> 2 ); // unrolling with 8 elements
	      while( k4-- ){
	          ymm0  = *(B+0+0*ldb); ymm4  = *(B+1+0*ldb); ymm8  = *(B+2+0*ldb); ymm12 = *(B+3+0*ldb);
	          *(B2+0+0*4) = ymm0 ; *(B2+1+0*4) = ymm4 ; *(B2+2+0*4) = ymm8 ; *(B2+3+0*4) = ymm12;
	          B+=4;
	          B2+=4;
	      }
	    }
	    if( k & 2 ){
	        ymm0  = *(B+0+0*ldb); ymm4  = *(B+1+0*ldb);
	        *(B2+0+0*2) = ymm0 ; *(B2+1+0*2) = ymm4 ;
	        B+=2;
	        B2+=2;
	    }
	    if( k & 1 ){
	        ymm0  = *(B+0+0*ldb); 
	        *(B2+0+0*1) = ymm0 ;
	        B+=1;
	        B2+=1;
	    }
	    B  = B  - K1 + 1*ldb ;
	    //B2 = B2 - K1 + 4*K1;

	}

	//if( n & 3 ){
	//  size_t nr = ( n & 3 );
	//  while( nr-- ){
	//    size_t k = K1;
	//    while( k-- ){
	//      *B2=*B;
	//      B++;
	//      B2++;
	//    }
	//    B  = B  - K1 + ldb ;
	//  }
	//}


}


