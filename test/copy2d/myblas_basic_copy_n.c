#include "copy2d_test.h"

void myblas_basic_copy_n_4x8(size_t K, size_t N, const double* B, size_t k, size_t j,  size_t ldb, double* B2 );
void myblas_basic_copy_n_2x8(size_t K, size_t N, const double* B, size_t k, size_t j,  size_t ldb, double* B2 );
void myblas_basic_copy_n_4x4(size_t K, size_t N, const double* B, size_t k, size_t j,  size_t ldb, double* B2 );

static copy_detail_func_t copy_n_detail[]={ myblas_basic_copy_n_4x8, myblas_basic_copy_n_2x8 };

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

void myblas_basic_copy_n_4x8(size_t K, size_t N, const double* B, size_t k, size_t j,  size_t ldb, double* B2 ){

	B = B + k + ldb*j; // start point

	if( N >> 2 ){
	  size_t n4 = ( N >> 2 );
	  while( n4-- ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){
	          for( size_t j=0; j<4; j++ ){
	        for( size_t l=0; l<8; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=8;
	      }
	    }
	    if( K & 4 ){
	          for( size_t j=0; j<4; j++ ){
	        for( size_t l=0; l<4; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	    }
	    if( K & 2 ){
	          for( size_t j=0; j<4; j++ ){
	        for( size_t l=0; l<2; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( K & 1 ){
	          for( size_t j=0; j<4; j++ ){
	        for( size_t l=0; l<1; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K + 4*ldb ;

	  }
	}
	if( N & 2 ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<8; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=8;
	      }
	    }
	    if( K & 4 ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<4; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	    }
	    if( K & 2 ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<2; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( K & 1 ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<1; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K + 2*ldb ;

	}
	if( N & 1 ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<8; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=8;
	      }
	    }
	    if( K & 4 ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<4; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	    }
	    if( K & 2 ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<2; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( K & 1 ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<1; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K + ldb ;

	}

}

void myblas_basic_copy_n_2x8(size_t K, size_t N, const double* B, size_t k, size_t j,  size_t ldb, double* B2 ){

	B = B + k + ldb*j; // start point

	if( N >> 1 ){
	  size_t n2 = ( N >> 1 );
	  while( n2-- ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<8; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=8;
	      }
	    }
	    if( K & 4 ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<4; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	    }
	    if( K & 2 ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<2; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( K & 1 ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<1; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K + 2*ldb ;
	  }
	}
	if( N & 1 ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<8; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=8;
	      }
	    }
	    if( K & 4 ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<4; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	    }
	    if( K & 2 ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<2; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( K & 1 ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<1; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K + ldb ;

	}


}


void myblas_basic_copy_n_4x4(size_t K, size_t N, const double* B, size_t k, size_t j,  size_t ldb, double* B2 ){

	B = B + k + ldb*j; // start point

	if( N >> 2 ){
	  size_t n4 = ( N >> 2 );
	  while( n4-- ){

	    if( K >> 2 ){
	      size_t k4 = ( K >> 2 );
	      while( k4-- ){
	          for( size_t j=0; j<4; j++ ){
	        for( size_t l=0; l<4; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	      }
	    }
	    if( K & 2 ){
	          for( size_t j=0; j<4; j++ ){
	        for( size_t l=0; l<2; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( K & 1 ){
	          for( size_t j=0; j<4; j++ ){
	        for( size_t l=0; l<1; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K + 4*ldb ;

	  }
	}
	if( N & 2 ){

	    if( K >> 2 ){
	      size_t k4 = ( K >> 2 );
	      while( k4-- ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<4; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	      }
	    }
	    if( K & 2 ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<2; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( K & 1 ){
	          for( size_t j=0; j<2; j++ ){
	        for( size_t l=0; l<1; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K + 2*ldb ;

	}
	if( N & 1 ){

	    if( K >> 2 ){
	      size_t k4 = ( K >> 2 );
	      while( k4-- ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<4; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=4;
	      }
	    }
	    if( K & 2 ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<2; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=2;
	    }
	    if( K & 1 ){
	          for( size_t j=0; j<1; j++ ){
	        for( size_t l=0; l<1; l++ ){
	            *B2=*(B+l+j*ldb);
	            B2++;
	          }
	        }
	        B+=1;
	    }
	    B  = B  - K + ldb ;

	}


}


