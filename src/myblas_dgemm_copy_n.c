#include "myblas_internal.h"
//#include <stdio.h>

static void myblas_dgemm_copy_n_internal(size_t K1, size_t N1, const double* B, size_t k, size_t j, size_t ldb, double* B2 );

// On L2-Cache Copy for B
void myblas_dgemm_copy_n(const double* B, size_t ldb, double* B2, const block2d_info_t* info ){

	size_t k2     = info->i2    ;
	size_t j2     = info->j2    ;
	size_t K2     = info->M2    ;
	size_t N2     = info->N2    ;
	size_t tile_K = info->tile_M;
	size_t tile_N = info->tile_N;

	size_t NB = N2/tile_N;
	size_t NR = N2%tile_N;
	size_t KB = K2/tile_K;
	size_t KR = K2%tile_K;

	//block2d_info_t tile = { 0,0,0,0,0,0 };

	B = B + k2 + ldb*j2; // start point

	if( NR >  0 ){ NB++; }
	if( NR == 0 ){ NR = tile_N; }
	if( KR >  0 ){ KB++; }
	if( KR == 0 ){ KR = tile_K; }

	// L1-cache blocking
	size_t k1 = KB;
	while( k1-- ){
	  size_t K1 = tile_K; if( k1==0 ){ K1=KR; }
	  size_t n1 = NB;
	  while( n1-- ){
	    size_t N1 = tile_N; if( n1==0 ){ N1=NR; }

	    myblas_dgemm_copy_n_internal( K1, N1, B, 0, 0, ldb, B2 );


	    B = B + N1*ldb;
	    B2= B2+ N1*K1;
	  }
	  B = B - N2*ldb + K1;
	}

}


void myblas_dgemm_copy_n_core(const double* B, size_t ldb, double* B2, const block2d_info_t* info ){

	myblas_dgemm_copy_n_internal( info->M2, info->N2, B, info->i2, info->j2, ldb, B2 );

}

void myblas_dgemm_copy_n_internal(size_t K1, size_t N1, const double* B, size_t k, size_t j, size_t ldb, double* B2 ){

	const double *b0 = B  + k + ldb*j;
	const double *b1 = b0 + ldb;
	double       *c0 = B2;
	double       *c1 = c0 + K1;
	size_t      ldb2 = ldb * 2 * sizeof(double);
	size_t      ldc2 = K1  * 2 * sizeof(double);

	size_t n = N1;
	if( n >> 2 ){
	  size_t n4 = ( n >> 2 ); // unrolling with 8 elements
	  while( n4-- )
	  {
	    size_t k = K1;
	    if( k >> 4 ){
	      size_t k16 = ( k >> 4 ); // unrolling with 16 elements
	      while( k16-- ){

	        __asm__ __volatile__ (
	            "vmovupd   0*8(%[b0]        ), %%ymm0\n\t"
	            "vmovupd   0*8(%[b1]        ), %%ymm1\n\t"
	            "vmovupd   0*8(%[b0],%[ldb2]), %%ymm2\n\t"
	            "vmovupd   0*8(%[b1],%[ldb2]), %%ymm3\n\t"
	            "vmovupd   %%ymm0,   0*8(%[c0]        )\n\t"
	            "vmovupd   %%ymm1,   0*8(%[c1]        )\n\t"
	            "vmovupd   %%ymm2,   0*8(%[c0],%[ldc2])\n\t"
	            "vmovupd   %%ymm3,   0*8(%[c1],%[ldc2])\n\t"
	            "\n\t"
	            "vmovupd   4*8(%[b0]        ), %%ymm8 \n\t"
	            "vmovupd   4*8(%[b1]        ), %%ymm9 \n\t"
	            "vmovupd   4*8(%[b0],%[ldb2]), %%ymm10\n\t"
	            "vmovupd   4*8(%[b1],%[ldb2]), %%ymm11\n\t"
	            "vmovupd   %%ymm8 ,   4*8(%[c0]        )\n\t"
	            "vmovupd   %%ymm9 ,   4*8(%[c1]        )\n\t"
	            "vmovupd   %%ymm10,   4*8(%[c0],%[ldc2])\n\t"
	            "vmovupd   %%ymm11,   4*8(%[c1],%[ldc2])\n\t"
	            "\n\t"
	            "vmovupd   8*8(%[b0]        ), %%ymm4\n\t"
	            "vmovupd   8*8(%[b1]        ), %%ymm5\n\t"
	            "vmovupd   8*8(%[b0],%[ldb2]), %%ymm6\n\t"
	            "vmovupd   8*8(%[b1],%[ldb2]), %%ymm7\n\t"
	            "vmovupd   %%ymm4,   8*8(%[c0]        )\n\t"
	            "vmovupd   %%ymm5,   8*8(%[c1]        )\n\t"
	            "vmovupd   %%ymm6,   8*8(%[c0],%[ldc2])\n\t"
	            "vmovupd   %%ymm7,   8*8(%[c1],%[ldc2])\n\t"
	            "\n\t"
	            "vmovupd  12*8(%[b0]        ), %%ymm12\n\t"
	            "vmovupd  12*8(%[b1]        ), %%ymm13\n\t"
	            "vmovupd  12*8(%[b0],%[ldb2]), %%ymm14\n\t"
	            "vmovupd  12*8(%[b1],%[ldb2]), %%ymm15\n\t"
	            "vmovupd   %%ymm12,  12*8(%[c0]        )\n\t"
	            "vmovupd   %%ymm13,  12*8(%[c1]        )\n\t"
	            "vmovupd   %%ymm14,  12*8(%[c0],%[ldc2])\n\t"
	            "vmovupd   %%ymm15,  12*8(%[c1],%[ldc2])\n\t"
	            "\n\t"
	            "addq   $16*8, %[b0]\n\t"
	            "addq   $16*8, %[b1]\n\t"
	            "addq   $16*8, %[c0]\n\t"
	            "addq   $16*8, %[c1]\n\t"
	            "\n\t"
	            :[b0]"+r"(b0),[b1]"+r"(b1),[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldb2]"r"(ldb2),[ldc2]"r"(ldc2)
	            );

	      }
	    }
	    if( k & 8 ){

	        __asm__ __volatile__ (
	            "vmovupd   0*8(%[b0]        ), %%ymm0\n\t"
	            "vmovupd   0*8(%[b1]        ), %%ymm1\n\t"
	            "vmovupd   0*8(%[b0],%[ldb2]), %%ymm2\n\t"
	            "vmovupd   0*8(%[b1],%[ldb2]), %%ymm3\n\t"
	            "vmovupd   %%ymm0,   0*8(%[c0]        )\n\t"
	            "vmovupd   %%ymm1,   0*8(%[c1]        )\n\t"
	            "vmovupd   %%ymm2,   0*8(%[c0],%[ldc2])\n\t"
	            "vmovupd   %%ymm3,   0*8(%[c1],%[ldc2])\n\t"
	            "\n\t"
	            "vmovupd   4*8(%[b0]        ), %%ymm8 \n\t"
	            "vmovupd   4*8(%[b1]        ), %%ymm9 \n\t"
	            "vmovupd   4*8(%[b0],%[ldb2]), %%ymm10\n\t"
	            "vmovupd   4*8(%[b1],%[ldb2]), %%ymm11\n\t"
	            "vmovupd   %%ymm8 ,   4*8(%[c0]        )\n\t"
	            "vmovupd   %%ymm9 ,   4*8(%[c1]        )\n\t"
	            "vmovupd   %%ymm10,   4*8(%[c0],%[ldc2])\n\t"
	            "vmovupd   %%ymm11,   4*8(%[c1],%[ldc2])\n\t"
	            "\n\t"
	            "addq   $8*8, %[b0]\n\t"
	            "addq   $8*8, %[b1]\n\t"
	            "addq   $8*8, %[c0]\n\t"
	            "addq   $8*8, %[c1]\n\t"
	            "\n\t"
	            :[b0]"+r"(b0),[b1]"+r"(b1),[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldb2]"r"(ldb2),[ldc2]"r"(ldc2)
	            );

	    }
	    if( k & 4 ){

	        __asm__ __volatile__ (
	            "vmovupd   0*8(%[b0]        ), %%ymm0\n\t"
	            "vmovupd   0*8(%[b1]        ), %%ymm1\n\t"
	            "vmovupd   0*8(%[b0],%[ldb2]), %%ymm2\n\t"
	            "vmovupd   0*8(%[b1],%[ldb2]), %%ymm3\n\t"
	            "vmovupd   %%ymm0,   0*8(%[c0]        )\n\t"
	            "vmovupd   %%ymm1,   0*8(%[c1]        )\n\t"
	            "vmovupd   %%ymm2,   0*8(%[c0],%[ldc2])\n\t"
	            "vmovupd   %%ymm3,   0*8(%[c1],%[ldc2])\n\t"
	            "\n\t"
	            "addq   $4*8, %[b0]\n\t"
	            "addq   $4*8, %[b1]\n\t"
	            "addq   $4*8, %[c0]\n\t"
	            "addq   $4*8, %[c1]\n\t"
	            "\n\t"
	            :[b0]"+r"(b0),[b1]"+r"(b1),[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldb2]"r"(ldb2),[ldc2]"r"(ldc2)
	            );

	    }
	    if( k & 2 ){

	        __asm__ __volatile__ (
	            "movupd   0*8(%[b0]        ), %%xmm0\n\t"
	            "movupd   0*8(%[b1]        ), %%xmm1\n\t"
	            "movupd   0*8(%[b0],%[ldb2]), %%xmm2\n\t"
	            "movupd   0*8(%[b1],%[ldb2]), %%xmm3\n\t"
	            "movupd   %%xmm0,   0*8(%[c0]        )\n\t"
	            "movupd   %%xmm1,   0*8(%[c1]        )\n\t"
	            "movupd   %%xmm2,   0*8(%[c0],%[ldc2])\n\t"
	            "movupd   %%xmm3,   0*8(%[c1],%[ldc2])\n\t"
	            "\n\t"
	            "addq   $2*8, %[b0]\n\t"
	            "addq   $2*8, %[b1]\n\t"
	            "addq   $2*8, %[c0]\n\t"
	            "addq   $2*8, %[c1]\n\t"
	            "\n\t"
	            :[b0]"+r"(b0),[b1]"+r"(b1),[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldb2]"r"(ldb2),[ldc2]"r"(ldc2)
	            );

	    }
	    if( k & 1 ){

	        __asm__ __volatile__ (
	            "movsd   0*8(%[b0]        ), %%xmm0\n\t"
	            "movsd   0*8(%[b1]        ), %%xmm1\n\t"
	            "movsd   0*8(%[b0],%[ldb2]), %%xmm2\n\t"
	            "movsd   0*8(%[b1],%[ldb2]), %%xmm3\n\t"
	            "movlpd  %%xmm0,   0*8(%[c0]        )\n\t"
	            "movlpd  %%xmm1,   0*8(%[c1]        )\n\t"
	            "movlpd  %%xmm2,   0*8(%[c0],%[ldc2])\n\t"
	            "movlpd  %%xmm3,   0*8(%[c1],%[ldc2])\n\t"
	            "\n\t"
	            "addq   $1*8, %[b0]\n\t"
	            "addq   $1*8, %[b1]\n\t"
	            "addq   $1*8, %[c0]\n\t"
	            "addq   $1*8, %[c1]\n\t"
	            "\n\t"
	            :[b0]"+r"(b0),[b1]"+r"(b1),[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldb2]"r"(ldb2),[ldc2]"r"(ldc2)
	            );

	    }
	    b0 = b0 - K1 + 4*ldb;
	    b1 = b1 - K1 + 4*ldb;
	    c0 = c0 - K1 + 4*K1;
	    c1 = c1 - K1 + 4*K1;
	  }
	}
	if( n & 2 ){

	  size_t k = K1;
	  if( k >> 4 ){
	    size_t k16 = ( k >> 4 ); // unrolling with 16 elements
	    while( k16-- ){

	      __asm__ __volatile__ (
	          "vmovupd   0*8(%[b0]        ), %%ymm0\n\t"
	          "vmovupd   0*8(%[b1]        ), %%ymm1\n\t"
	          "vmovupd   %%ymm0,   0*8(%[c0]        )\n\t"
	          "vmovupd   %%ymm1,   0*8(%[c1]        )\n\t"
	          "\n\t"
	          "vmovupd   4*8(%[b0]        ), %%ymm8 \n\t"
	          "vmovupd   4*8(%[b1]        ), %%ymm9 \n\t"
	          "vmovupd   %%ymm8 ,   4*8(%[c0]        )\n\t"
	          "vmovupd   %%ymm9 ,   4*8(%[c1]        )\n\t"
	          "\n\t"
	          "vmovupd   8*8(%[b0]        ), %%ymm4\n\t"
	          "vmovupd   8*8(%[b1]        ), %%ymm5\n\t"
	          "vmovupd   %%ymm4,   8*8(%[c0]        )\n\t"
	          "vmovupd   %%ymm5,   8*8(%[c1]        )\n\t"
	          "\n\t"
	          "vmovupd  12*8(%[b0]        ), %%ymm12\n\t"
	          "vmovupd  12*8(%[b1]        ), %%ymm13\n\t"
	          "vmovupd   %%ymm12,  12*8(%[c0]        )\n\t"
	          "vmovupd   %%ymm13,  12*8(%[c1]        )\n\t"
	          "\n\t"
	          "addq   $16*8, %[b0]\n\t"
	          "addq   $16*8, %[b1]\n\t"
	          "addq   $16*8, %[c0]\n\t"
	          "addq   $16*8, %[c1]\n\t"
	          "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[c0]"+r"(c0),[c1]"+r"(c1)
	          :[ldb2]"r"(ldb2),[ldc2]"r"(ldc2)
	          );

	    }
	  }
	  if( k & 8 ){

	      __asm__ __volatile__ (
	          "vmovupd   0*8(%[b0]        ), %%ymm0\n\t"
	          "vmovupd   0*8(%[b1]        ), %%ymm1\n\t"
	          "vmovupd   %%ymm0,   0*8(%[c0]        )\n\t"
	          "vmovupd   %%ymm1,   0*8(%[c1]        )\n\t"
	          "\n\t"
	          "vmovupd   4*8(%[b0]        ), %%ymm8 \n\t"
	          "vmovupd   4*8(%[b1]        ), %%ymm9 \n\t"
	          "vmovupd   %%ymm8 ,   4*8(%[c0]        )\n\t"
	          "vmovupd   %%ymm9 ,   4*8(%[c1]        )\n\t"
	          "\n\t"
	          "addq   $8*8, %[b0]\n\t"
	          "addq   $8*8, %[b1]\n\t"
	          "addq   $8*8, %[c0]\n\t"
	          "addq   $8*8, %[c1]\n\t"
	          "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[c0]"+r"(c0),[c1]"+r"(c1)
	          :[ldb2]"r"(ldb2),[ldc2]"r"(ldc2)
	          );

	  }
	  if( k & 4 ){

	      __asm__ __volatile__ (
	          "vmovupd   0*8(%[b0]        ), %%ymm0\n\t"
	          "vmovupd   0*8(%[b1]        ), %%ymm1\n\t"
	          "vmovupd   %%ymm0,   0*8(%[c0]        )\n\t"
	          "vmovupd   %%ymm1,   0*8(%[c1]        )\n\t"
	          "\n\t"
	          "addq   $4*8, %[b0]\n\t"
	          "addq   $4*8, %[b1]\n\t"
	          "addq   $4*8, %[c0]\n\t"
	          "addq   $4*8, %[c1]\n\t"
	          "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[c0]"+r"(c0),[c1]"+r"(c1)
	          :[ldb2]"r"(ldb2),[ldc2]"r"(ldc2)
	          );

	  }
	  if( k & 2 ){

	      __asm__ __volatile__ (
	          "movupd   0*8(%[b0]        ), %%xmm0\n\t"
	          "movupd   0*8(%[b1]        ), %%xmm1\n\t"
	          "movupd   %%xmm0,   0*8(%[c0]        )\n\t"
	          "movupd   %%xmm1,   0*8(%[c1]        )\n\t"
	          "\n\t"
	          "addq   $2*8, %[b0]\n\t"
	          "addq   $2*8, %[b1]\n\t"
	          "addq   $2*8, %[c0]\n\t"
	          "addq   $2*8, %[c1]\n\t"
	          "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[c0]"+r"(c0),[c1]"+r"(c1)
	          :[ldb2]"r"(ldb2),[ldc2]"r"(ldc2)
	          );

	  }
	  if( k & 1 ){

	      __asm__ __volatile__ (
	          "movsd   0*8(%[b0]        ), %%xmm0\n\t"
	          "movsd   0*8(%[b1]        ), %%xmm1\n\t"
	          "movlpd  %%xmm0,   0*8(%[c0]        )\n\t"
	          "movlpd  %%xmm1,   0*8(%[c1]        )\n\t"
	          "\n\t"
	          "addq   $1*8, %[b0]\n\t"
	          "addq   $1*8, %[b1]\n\t"
	          "addq   $1*8, %[c0]\n\t"
	          "addq   $1*8, %[c1]\n\t"
	          "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[c0]"+r"(c0),[c1]"+r"(c1)
	          :[ldb2]"r"(ldb2),[ldc2]"r"(ldc2)
	          );

	  }
	  b0 = b0 - K1 + 2*ldb;
	  b1 = b1 - K1 + 2*ldb;
	  c0 = c0 - K1 + 2*K1;
	  c1 = c1 - K1 + 2*K1;

	}
	if( n & 1 ){

	  size_t k = K1;
	  if( k >> 4 ){
	    size_t k16 = ( k >> 4 ); // unrolling with 16 elements
	    while( k16-- ){

	      __asm__ __volatile__ (
	          "vmovupd   0*8(%[b0]        ), %%ymm0\n\t"
	          "vmovupd   %%ymm0,   0*8(%[c0]        )\n\t"
	          "\n\t"
	          "vmovupd   4*8(%[b0]        ), %%ymm8 \n\t"
	          "vmovupd   %%ymm8 ,   4*8(%[c0]        )\n\t"
	          "\n\t"
	          "vmovupd   8*8(%[b0]        ), %%ymm4\n\t"
	          "vmovupd   %%ymm4,   8*8(%[c0]        )\n\t"
	          "\n\t"
	          "vmovupd  12*8(%[b0]        ), %%ymm12\n\t"
	          "vmovupd   %%ymm12,  12*8(%[c0]        )\n\t"
	          "\n\t"
	          "addq   $16*8, %[b0]\n\t"
	          "addq   $16*8, %[c0]\n\t"
	          "\n\t"
	          :[b0]"+r"(b0),[c0]"+r"(c0)
	          );

	    }
	  }
	  if( k & 8 ){

	      __asm__ __volatile__ (
	          "vmovupd   0*8(%[b0]        ), %%ymm0\n\t"
	          "vmovupd   %%ymm0,   0*8(%[c0]        )\n\t"
	          "\n\t"
	          "vmovupd   4*8(%[b0]        ), %%ymm8 \n\t"
	          "vmovupd   %%ymm8 ,   4*8(%[c0]        )\n\t"
	          "\n\t"
	          "addq   $8*8, %[b0]\n\t"
	          "addq   $8*8, %[c0]\n\t"
	          "\n\t"
	          :[b0]"+r"(b0),[c0]"+r"(c0)
	          );

	  }
	  if( k & 4 ){

	      __asm__ __volatile__ (
	          "vmovupd   0*8(%[b0]        ), %%ymm0\n\t"
	          "vmovupd   %%ymm0,   0*8(%[c0]        )\n\t"
	          "\n\t"
	          "addq   $4*8, %[b0]\n\t"
	          "addq   $4*8, %[c0]\n\t"
	          "\n\t"
	          :[b0]"+r"(b0),[c0]"+r"(c0)
	          );

	  }
	  if( k & 2 ){

	      __asm__ __volatile__ (
	          "movupd   0*8(%[b0]        ), %%xmm0\n\t"
	          "movupd   %%xmm0,   0*8(%[c0]        )\n\t"
	          "\n\t"
	          "addq   $2*8, %[b0]\n\t"
	          "addq   $2*8, %[c0]\n\t"
	          "\n\t"
	          :[b0]"+r"(b0),[c0]"+r"(c0)
	          );

	  }
	  if( k & 1 ){

	      __asm__ __volatile__ (
	          "movsd   0*8(%[b0]        ), %%xmm0\n\t"
	          "movlpd  %%xmm0,   0*8(%[c0]        )\n\t"
	          "\n\t"
	          "addq   $1*8, %[b0]\n\t"
	          "addq   $1*8, %[c0]\n\t"
	          "\n\t"
	          :[b0]"+r"(b0),[c0]"+r"(c0)
	          );

	  }
	  b0 = b0 - K1 + ldb;
	  c0 = c0 - K1 + K1;

	}

}


