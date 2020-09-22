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
	        A  = A  + MU;\
	        a0 = a0 + MU;\
	        a1 = a1 + MU;

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
	    A  = A - M + KU*LU*lda;\
	    a0 = a0 - M + KU*LU*lda;\
	    a1 = a1 - M + KU*LU*lda;

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
	    A  = A - M + KU*LU*lda;\
	    a0 = a0 - M + KU*LU*lda;\
	    a1 = a1 - M + KU*LU*lda;



void myblas_dgemm_copy_t_MxK(size_t K, size_t M, const double* A, size_t k1, size_t i1, size_t lda, double* A2 ){

	A = A + i1 + k1*lda;

	double* A2_2 = A2   + K*( M & ~3 );
	double* A2_1 = A2_2 + K*( M &  2 );;
	size_t  MM = ( M & ~3 ); // size M for Main Loop 

	const double* a0 = A;
	const double* a1 = A + lda;
	size_t  lda2 = 2*lda*sizeof(double);
	size_t  lda3 = 3*lda*sizeof(double);
	size_t  KK   = K*sizeof(double);

	if( K >> 3 ){
	  size_t k8 = ( K >> 3 );
	  while( k8-- ){

	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        __asm__ __volatile__ (
	          "\n\t"
	          "vmovupd  0*8(%[a0]          ), %%ymm0 \n\t" // [a00,a10,a20,a30]
	          "vmovupd  0*8(%[a1]          ), %%ymm1 \n\t" // [a01,a11,a21,a31]
	          "vmovupd  0*8(%[a0],%[lda2]  ), %%ymm2 \n\t" // [a02,a12,a22,a32]
	          "vmovupd  0*8(%[a1],%[lda2]  ), %%ymm3 \n\t" // [a03,a13,a23,a33]
	          "\n\t"
	          "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm4 \n\t" // [a00,a01,a20,a21]
	          "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm5 \n\t" // [a10,a11,a30,a31]
	          "vshufpd    $0x00 , %%ymm3 , %%ymm2 , %%ymm6 \n\t" // [a02,a03,a22,a23]
	          "vshufpd    $0x0f , %%ymm3 , %%ymm2 , %%ymm7 \n\t" // [a12,a13,a32,a33]
	          "\n\t"
	          "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm12\n\t" // [a00,a01,a10,a11]
	          "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm13\n\t" // [a20,a21,a30,a31]
	          "vperm2f128 $0x20 , %%ymm7 , %%ymm6 , %%ymm14\n\t" // [a02,a03,a12,a13]
	          "vperm2f128 $0x31 , %%ymm7 , %%ymm6 , %%ymm15\n\t" // [a22,a23,a32,a33]
	          "\n\t"
	          "vmovapd  %%ymm12,  0*8(%[a2]) \n\t"
	          "vmovapd  %%ymm13,  4*8(%[a2]) \n\t"
	          "vmovapd  %%ymm14,  8*8(%[a2]) \n\t"
	          "vmovapd  %%ymm15, 12*8(%[a2]) \n\t"
	          "\n\t"
	          "vmovupd  0*8(%[a0],%[lda2],2), %%ymm0 \n\t" // [a04,a14,a24,a34]
	          "vmovupd  0*8(%[a1],%[lda2],2), %%ymm1 \n\t" // [a05,a15,a25,a35]
	          "vmovupd  0*8(%[a0],%[lda3],2), %%ymm2 \n\t" // [a06,a16,a26,a36]
	          "vmovupd  0*8(%[a1],%[lda3],2), %%ymm3 \n\t" // [a07,a17,a27,a37]
	          "\n\t"
	          "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm4 \n\t" // [a04,a05,a24,a25]
	          "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm5 \n\t" // [a14,a15,a34,a35]
	          "vshufpd    $0x00 , %%ymm3 , %%ymm2 , %%ymm6 \n\t" // [a06,a07,a26,a27]
	          "vshufpd    $0x0f , %%ymm3 , %%ymm2 , %%ymm7 \n\t" // [a16,a17,a36,a37]
	          "\n\t"
	          "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm12\n\t" // [a04,a05,a14,a15]
	          "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm13\n\t" // [a24,a25,a34,a35]
	          "vperm2f128 $0x20 , %%ymm7 , %%ymm6 , %%ymm14\n\t" // [a06,a07,a16,a17]
	          "vperm2f128 $0x31 , %%ymm7 , %%ymm6 , %%ymm15\n\t" // [a26,a27,a36,a37]
	          "\n\t"
	          "vmovapd  %%ymm12, 16*8(%[a2]) \n\t"
	          "vmovapd  %%ymm13, 20*8(%[a2]) \n\t"
	          "vmovapd  %%ymm14, 24*8(%[a2]) \n\t"
	          "vmovapd  %%ymm15, 28*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $4*8, %[a0]\n\t"
	          "addq  $4*8, %[a1]\n\t"
	          "leaq  0*8(%[a2],%[K],4), %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3),[K]"r"(KK));

	      }
	      A2 = A2 - MM*K + 4*4*2; 
	    }
	    if( M & 2 ){

	        __asm__ __volatile__ (
	          "movupd  0*8(%[a0]          ), %%xmm0 \n\t" // [a00,a10,---,---]
	          "movupd  0*8(%[a1]          ), %%xmm1 \n\t" // [a01,a11,---,---]
	          "movupd  0*8(%[a0],%[lda2]  ), %%xmm2 \n\t" // [a02,a12,---,---]
	          "movupd  0*8(%[a1],%[lda2]  ), %%xmm3 \n\t" // [a03,a13,---,---]
	          "\n\t"
	          "vshufpd    $0x00 , %%xmm1 , %%xmm0 , %%xmm12\n\t" // [a00,a01,---,---]
	          "vshufpd    $0x0f , %%xmm1 , %%xmm0 , %%xmm13\n\t" // [a10,a11,---,---]
	          "vshufpd    $0x00 , %%xmm3 , %%xmm2 , %%xmm14\n\t" // [a02,a03,---,---]
	          "vshufpd    $0x0f , %%xmm3 , %%xmm2 , %%xmm15\n\t" // [a12,a13,---,---]
	          "\n\t"
	          "movapd  %%xmm12,  0*8(%[a2]) \n\t"
	          "movapd  %%xmm13,  2*8(%[a2]) \n\t"
	          "movapd  %%xmm14,  4*8(%[a2]) \n\t"
	          "movapd  %%xmm15,  6*8(%[a2]) \n\t"
	          "\n\t"
	          "movupd  0*8(%[a0],%[lda2],2), %%xmm0 \n\t" // [a04,a14,---,---]
	          "movupd  0*8(%[a1],%[lda2],2), %%xmm1 \n\t" // [a05,a15,---,---]
	          "movupd  0*8(%[a0],%[lda3],2), %%xmm2 \n\t" // [a06,a16,---,---]
	          "movupd  0*8(%[a1],%[lda3],2), %%xmm3 \n\t" // [a07,a17,---,---]
	          "\n\t"
	          "vshufpd    $0x00 , %%xmm1 , %%xmm0 , %%xmm12\n\t" // [a04,a05,---,---]
	          "vshufpd    $0x0f , %%xmm1 , %%xmm0 , %%xmm13\n\t" // [a14,a15,---,---]
	          "vshufpd    $0x00 , %%xmm3 , %%xmm2 , %%xmm14\n\t" // [a06,a07,---,---]
	          "vshufpd    $0x0f , %%xmm3 , %%xmm2 , %%xmm15\n\t" // [a16,a17,---,---]
	          "\n\t"
	          "movapd  %%xmm12,  8*8(%[a2]) \n\t"
	          "movapd  %%xmm13, 10*8(%[a2]) \n\t"
	          "movapd  %%xmm14, 12*8(%[a2]) \n\t"
	          "movapd  %%xmm15, 14*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $2*8 , %[a0]\n\t"
	          "addq  $2*8 , %[a1]\n\t"
	          "addq  $16*8, %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    if( M & 1 ){

	        __asm__ __volatile__ (
	          "movsd  0*8(%[a0]          ), %%xmm0 \n\t" // [a00,  0,---,---]
	          "movsd  0*8(%[a1]          ), %%xmm1 \n\t" // [a01,  0,---,---]
	          "movsd  0*8(%[a0],%[lda2]  ), %%xmm2 \n\t" // [a02,  0,---,---]
	          "movsd  0*8(%[a1],%[lda2]  ), %%xmm3 \n\t" // [a03,  0,---,---]
	          "movsd  0*8(%[a0],%[lda2],2), %%xmm4 \n\t" // [a04,  0,---,---]
	          "movsd  0*8(%[a1],%[lda2],2), %%xmm5 \n\t" // [a05,  0,---,---]
	          "movsd  0*8(%[a0],%[lda3],2), %%xmm6 \n\t" // [a06,  0,---,---]
	          "movsd  0*8(%[a1],%[lda3],2), %%xmm7 \n\t" // [a07,  0,---,---]
	          "\n\t"
	          "movsd  %%xmm0 ,  0*8(%[a2]) \n\t"
	          "movsd  %%xmm1 ,  1*8(%[a2]) \n\t"
	          "movsd  %%xmm2 ,  2*8(%[a2]) \n\t"
	          "movsd  %%xmm3 ,  3*8(%[a2]) \n\t"
	          "movsd  %%xmm4 ,  4*8(%[a2]) \n\t"
	          "movsd  %%xmm5 ,  5*8(%[a2]) \n\t"
	          "movsd  %%xmm6 ,  6*8(%[a2]) \n\t"
	          "movsd  %%xmm7 ,  7*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $1*8 , %[a0]\n\t"
	          "addq  $1*8 , %[a1]\n\t"
	          "addq  $8*8 , %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_1)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    a0 = a0 - M + 4*2*lda;
	    a1 = a1 - M + 4*2*lda;

	  }
	}
	if( K & 4  ){

	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        __asm__ __volatile__ (
	          "vmovupd  0*8(%[a0]          ), %%ymm0 \n\t" // [a00,a10,a20,a30]
	          "vmovupd  0*8(%[a1]          ), %%ymm1 \n\t" // [a01,a11,a21,a31]
	          "vmovupd  0*8(%[a0],%[lda2]  ), %%ymm2 \n\t" // [a02,a12,a22,a32]
	          "vmovupd  0*8(%[a1],%[lda2]  ), %%ymm3 \n\t" // [a03,a13,a23,a33]
	          "\n\t"
	          "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm4 \n\t" // [a00,a01,a20,a21]
	          "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm5 \n\t" // [a10,a11,a30,a31]
	          "vshufpd    $0x00 , %%ymm3 , %%ymm2 , %%ymm6 \n\t" // [a02,a03,a22,a23]
	          "vshufpd    $0x0f , %%ymm3 , %%ymm2 , %%ymm7 \n\t" // [a12,a13,a32,a33]
	          "\n\t"
	          "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm12\n\t" // [a00,a01,a10,a11]
	          "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm13\n\t" // [a20,a21,a30,a31]
	          "vperm2f128 $0x20 , %%ymm7 , %%ymm6 , %%ymm14\n\t" // [a02,a03,a12,a13]
	          "vperm2f128 $0x31 , %%ymm7 , %%ymm6 , %%ymm15\n\t" // [a22,a23,a32,a33]
	          "\n\t"
	          "vmovapd  %%ymm12,  0*8(%[a2]) \n\t"
	          "vmovapd  %%ymm13,  4*8(%[a2]) \n\t"
	          "vmovapd  %%ymm14,  8*8(%[a2]) \n\t"
	          "vmovapd  %%ymm15, 12*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $4*8, %[a0]\n\t"
	          "addq  $4*8, %[a1]\n\t"
	          "leaq  0*8(%[a2],%[K],4), %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3),[K]"r"(KK));

	      }
	      A2 = A2 - MM*K + 4*2*2; 
	    }
	    if( M & 2 ){

	        __asm__ __volatile__ (
	          "movupd  0*8(%[a0]          ), %%xmm0 \n\t" // [a00,a10,---,---]
	          "movupd  0*8(%[a1]          ), %%xmm1 \n\t" // [a01,a11,---,---]
	          "movupd  0*8(%[a0],%[lda2]  ), %%xmm2 \n\t" // [a02,a12,---,---]
	          "movupd  0*8(%[a1],%[lda2]  ), %%xmm3 \n\t" // [a03,a13,---,---]
	          "\n\t"
	          "vshufpd    $0x00 , %%xmm1 , %%xmm0 , %%xmm12\n\t" // [a00,a01,---,---]
	          "vshufpd    $0x0f , %%xmm1 , %%xmm0 , %%xmm13\n\t" // [a10,a11,---,---]
	          "vshufpd    $0x00 , %%xmm3 , %%xmm2 , %%xmm14\n\t" // [a02,a03,---,---]
	          "vshufpd    $0x0f , %%xmm3 , %%xmm2 , %%xmm15\n\t" // [a12,a13,---,---]
	          "\n\t"
	          "movapd  %%xmm12,  0*8(%[a2]) \n\t"
	          "movapd  %%xmm13,  2*8(%[a2]) \n\t"
	          "movapd  %%xmm14,  4*8(%[a2]) \n\t"
	          "movapd  %%xmm15,  6*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $2*8 , %[a0]\n\t"
	          "addq  $2*8 , %[a1]\n\t"
	          "addq  $8*8 , %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));


	    }
	    if( M & 1 ){

	        __asm__ __volatile__ (
	          "movsd  0*8(%[a0]          ), %%xmm0 \n\t" // [a00,  0,---,---]
	          "movsd  0*8(%[a1]          ), %%xmm1 \n\t" // [a01,  0,---,---]
	          "movsd  0*8(%[a0],%[lda2]  ), %%xmm2 \n\t" // [a02,  0,---,---]
	          "movsd  0*8(%[a1],%[lda2]  ), %%xmm3 \n\t" // [a03,  0,---,---]
	          "\n\t"
	          "movsd  %%xmm0 ,  0*8(%[a2]) \n\t"
	          "movsd  %%xmm1 ,  1*8(%[a2]) \n\t"
	          "movsd  %%xmm2 ,  2*8(%[a2]) \n\t"
	          "movsd  %%xmm3 ,  3*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $1*8 , %[a0]\n\t"
	          "addq  $1*8 , %[a1]\n\t"
	          "addq  $4*8 , %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_1)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    a0 = a0 - M + 2*2*lda;
	    a1 = a1 - M + 2*2*lda;

	}
	if( K & 2  ){

	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        __asm__ __volatile__ (
	          "vmovupd  0*8(%[a0]          ), %%ymm0 \n\t" // [a00,a10,a20,a30]
	          "vmovupd  0*8(%[a1]          ), %%ymm1 \n\t" // [a01,a11,a21,a31]
	          "\n\t"
	          "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm4 \n\t" // [a00,a01,a20,a21]
	          "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm5 \n\t" // [a10,a11,a30,a31]
	          "\n\t"
	          "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm12\n\t" // [a00,a01,a10,a11]
	          "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm13\n\t" // [a20,a21,a30,a31]
	          "\n\t"
	          "vmovapd  %%ymm12,  0*8(%[a2]) \n\t"
	          "vmovapd  %%ymm13,  4*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $4*8, %[a0]\n\t"
	          "addq  $4*8, %[a1]\n\t"
	          "leaq  0*8(%[a2],%[K],4), %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3),[K]"r"(KK));

	      }
	      A2 = A2 - MM*K + 4*1*2; 
	    }
	    if( M & 2 ){

	        __asm__ __volatile__ (
	          "movupd  0*8(%[a0]          ), %%xmm0 \n\t" // [a00,a10,---,---]
	          "movupd  0*8(%[a1]          ), %%xmm1 \n\t" // [a01,a11,---,---]
	          "\n\t"
	          "vshufpd    $0x00 , %%xmm1 , %%xmm0 , %%xmm12\n\t" // [a00,a01,---,---]
	          "vshufpd    $0x0f , %%xmm1 , %%xmm0 , %%xmm13\n\t" // [a10,a11,---,---]
	          "\n\t"
	          "movapd  %%xmm12,  0*8(%[a2]) \n\t"
	          "movapd  %%xmm13,  2*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $2*8 , %[a0]\n\t"
	          "addq  $2*8 , %[a1]\n\t"
	          "addq  $4*8 , %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    if( M & 1 ){

	        __asm__ __volatile__ (
	          "movsd  0*8(%[a0]          ), %%xmm0 \n\t" // [a00,  0,---,---]
	          "movsd  0*8(%[a1]          ), %%xmm1 \n\t" // [a01,  0,---,---]
	          "\n\t"
	          "movsd  %%xmm0 ,  0*8(%[a2]) \n\t"
	          "movsd  %%xmm1 ,  1*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $1*8 , %[a0]\n\t"
	          "addq  $1*8 , %[a1]\n\t"
	          "addq  $2*8 , %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_1)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    a0 = a0 - M + 1*2*lda;
	    a1 = a1 - M + 1*2*lda;


	}
	if( K & 1  ){

	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        __asm__ __volatile__ (
	          "vmovupd  0*8(%[a0]          ), %%ymm0 \n\t" // [a00,a10,a20,a30]
	          "\n\t"
	          "vmovapd  %%ymm0 ,  0*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $4*8, %[a0]\n\t"
	          "addq  $4*8, %[a1]\n\t"
	          "leaq  0*8(%[a2],%[K],4), %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3),[K]"r"(KK));

	      }
	      A2 = A2 - MM*K + 4*1*1; 
	    }
	    if( M & 2 ){

	        __asm__ __volatile__ (
	          "movupd  0*8(%[a0]          ), %%xmm0 \n\t" // [a00,a10,---,---]
	          "\n\t"
	          "movapd  %%xmm0 ,  0*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $2*8 , %[a0]\n\t"
	          "addq  $2*8 , %[a1]\n\t"
	          "addq  $2*8 , %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    if( M & 1 ){

	        __asm__ __volatile__ (
	          "movsd  0*8(%[a0]          ), %%xmm0 \n\t" // [a00,  0,---,---]
	          "\n\t"
	          "movsd  %%xmm0 ,  0*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $1*8 , %[a0]\n\t"
	          "addq  $1*8 , %[a1]\n\t"
	          "addq  $1*8 , %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_1)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    a0 = a0 - M + 1*1*lda;
	    a1 = a1 - M + 1*1*lda;

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

	const double* a0 = A;
	const double* a1 = A + lda;
	size_t  lda2 = 2*lda*sizeof(double);
	size_t  lda3 = 3*lda*sizeof(double);
	size_t  K3   = 3*K*sizeof(double);

	if( K >> 3 ){
	  size_t k8 = ( K >> 3 );
	  while( k8-- ){

	    //COPY_T_M6(4,2);
	    if( MQ ){
	      size_t m6 = MQ;
	      while( m6-- ){
	        //COPY_T(6,4,2,A2);
	        __asm__ __volatile__ (
	          "vmovupd  0*8(%[a0]          ), %%ymm0 \n\t" // [a00,a10,a20,a30]
	          "vmovupd  0*8(%[a1]          ), %%ymm1 \n\t" // [a01,a11,a21,a31]
	          "vmovupd  4*8(%[a0]          ), %%xmm2 \n\t" // [a40,a50,---,---]
	          "vmovupd  4*8(%[a1]          ), %%xmm3 \n\t" // [a41,a51,---,---]
	          "\n\t"
	          "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm4 \n\t" // [a00,a01,a20,a21]
	          "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm5 \n\t" // [a10,a11,a30,a31]
	          "vshufpd    $0x00 , %%ymm3 , %%ymm2 , %%ymm6 \n\t" // [a40,a41,---,---]
	          "vshufpd    $0x0f , %%ymm3 , %%ymm2 , %%ymm7 \n\t" // [a50,a51,---,---]
	          "\n\t"
	          "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm10\n\t" // [a00,a01,a10,a11]
	          "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm11\n\t" // [a20,a21,a30,a31]
	          "vperm2f128 $0x20 , %%ymm7 , %%ymm6 , %%ymm12\n\t" // [a40,a41,a50,a51]
	          "\n\t"
	          "vmovupd  %%ymm10,  0*8(%[a2]) \n\t"
	          "vmovupd  %%ymm11,  4*8(%[a2]) \n\t"
	          "vmovupd  %%ymm12,  8*8(%[a2]) \n\t"
	          "\n\t"
	          "vmovupd  0*8(%[a0],%[lda2]  ), %%ymm0 \n\t" // [a02,a12,a22,a32]
	          "vmovupd  0*8(%[a1],%[lda2]  ), %%ymm1 \n\t" // [a03,a13,a23,a33]
	          "vmovupd  4*8(%[a0],%[lda2]  ), %%xmm2 \n\t" // [a42,a52,---,---]
	          "vmovupd  4*8(%[a1],%[lda2]  ), %%xmm3 \n\t" // [a43,a53,---,---]
	          "\n\t"
	          "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm4 \n\t" // [a02,a03,a22,a23]
	          "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm5 \n\t" // [a12,a13,a32,a33]
	          "vshufpd    $0x00 , %%ymm3 , %%ymm2 , %%ymm6 \n\t" // [a42,a43,---,---]
	          "vshufpd    $0x0f , %%ymm3 , %%ymm2 , %%ymm7 \n\t" // [a52,a53,---,---]
	          "\n\t"
	          "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm13\n\t" // [a02,a03,a12,a13]
	          "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm14\n\t" // [a22,a23,a32,a33]
	          "vperm2f128 $0x20 , %%ymm7 , %%ymm6 , %%ymm15\n\t" // [a42,a43,a52,a53]
	          "\n\t"
	          "vmovupd  %%ymm13, 12*8(%[a2]) \n\t"
	          "vmovupd  %%ymm14, 16*8(%[a2]) \n\t"
	          "vmovupd  %%ymm15, 20*8(%[a2]) \n\t"
	          "\n\t"
	          "vmovupd  0*8(%[a0],%[lda2],2), %%ymm0 \n\t" // [a04,a14,a24,a34]
	          "vmovupd  0*8(%[a1],%[lda2],2), %%ymm1 \n\t" // [a05,a15,a25,a35]
	          "vmovupd  4*8(%[a0],%[lda2],2), %%xmm2 \n\t" // [a44,a54,---,---]
	          "vmovupd  4*8(%[a1],%[lda2],2), %%xmm3 \n\t" // [a45,a55,---,---]
	          "\n\t"
	          "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm4 \n\t" // [a04,a05,a24,a25]
	          "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm5 \n\t" // [a14,a15,a34,a35]
	          "vshufpd    $0x00 , %%ymm3 , %%ymm2 , %%ymm6 \n\t" // [a44,a45,---,---]
	          "vshufpd    $0x0f , %%ymm3 , %%ymm2 , %%ymm7 \n\t" // [a54,a55,---,---]
	          "\n\t"
	          "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm10\n\t" // [a04,a05,a14,a15]
	          "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm11\n\t" // [a24,a25,a34,a35]
	          "vperm2f128 $0x20 , %%ymm7 , %%ymm6 , %%ymm12\n\t" // [a44,a45,a54,a55]
	          "\n\t"
	          "vmovupd  %%ymm10, 24*8(%[a2]) \n\t"
	          "vmovupd  %%ymm11, 28*8(%[a2]) \n\t"
	          "vmovupd  %%ymm12, 32*8(%[a2]) \n\t"
	          "\n\t"
	          "vmovupd  0*8(%[a0],%[lda3],2), %%ymm0 \n\t" // [a06,a16,a26,a36]
	          "vmovupd  0*8(%[a1],%[lda3],2), %%ymm1 \n\t" // [a07,a17,a27,a37]
	          "vmovupd  4*8(%[a0],%[lda3],2), %%xmm2 \n\t" // [a46,a56,---,---]
	          "vmovupd  4*8(%[a1],%[lda3],2), %%xmm3 \n\t" // [a47,a57,---,---]
	          "\n\t"
	          "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm4 \n\t" // [a06,a07,a26,a27]
	          "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm5 \n\t" // [a16,a17,a36,a37]
	          "vshufpd    $0x00 , %%ymm3 , %%ymm2 , %%ymm6 \n\t" // [a46,a47,---,---]
	          "vshufpd    $0x0f , %%ymm3 , %%ymm2 , %%ymm7 \n\t" // [a56,a57,---,---]
	          "\n\t"
	          "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm13\n\t" // [a06,a07,a16,a17]
	          "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm14\n\t" // [a26,a27,a36,a37]
	          "vperm2f128 $0x20 , %%ymm7 , %%ymm6 , %%ymm15\n\t" // [a46,a47,a56,a57]
	          "\n\t"
	          "vmovupd  %%ymm13, 36*8(%[a2]) \n\t"
	          "vmovupd  %%ymm14, 40*8(%[a2]) \n\t"
	          "vmovupd  %%ymm15, 44*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $6*8, %[a0]\n\t"
	          "addq  $6*8, %[a1]\n\t"
	          "leaq  0*8(%[a2],%[K3],2), %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3),[K3]"r"(K3));

	      }
	      A2 = A2 - MM*K + 6*4*2; 
	    }
	    if( MR & 4 ){

	        __asm__ __volatile__ (
	          "vmovupd  0*8(%[a0]          ), %%ymm0 \n\t" // [a00,a10,a20,a30]
	          "vmovupd  0*8(%[a1]          ), %%ymm1 \n\t" // [a01,a11,a21,a31]
	          "vmovupd  0*8(%[a0],%[lda2]  ), %%ymm2 \n\t" // [a02,a12,a22,a32]
	          "vmovupd  0*8(%[a1],%[lda2]  ), %%ymm3 \n\t" // [a03,a13,a23,a33]
	          "\n\t"
	          "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm4 \n\t" // [a00,a01,a20,a21]
	          "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm5 \n\t" // [a10,a11,a30,a31]
	          "vshufpd    $0x00 , %%ymm3 , %%ymm2 , %%ymm6 \n\t" // [a02,a03,a22,a23]
	          "vshufpd    $0x0f , %%ymm3 , %%ymm2 , %%ymm7 \n\t" // [a12,a13,a32,a33]
	          "\n\t"
	          "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm12\n\t" // [a00,a01,a10,a11]
	          "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm13\n\t" // [a20,a21,a30,a31]
	          "vperm2f128 $0x20 , %%ymm7 , %%ymm6 , %%ymm14\n\t" // [a02,a03,a12,a13]
	          "vperm2f128 $0x31 , %%ymm7 , %%ymm6 , %%ymm15\n\t" // [a22,a23,a32,a33]
	          "\n\t"
	          "vmovapd  %%ymm12,  0*8(%[a2]) \n\t"
	          "vmovapd  %%ymm13,  4*8(%[a2]) \n\t"
	          "vmovapd  %%ymm14,  8*8(%[a2]) \n\t"
	          "vmovapd  %%ymm15, 12*8(%[a2]) \n\t"
	          "\n\t"
	          "vmovupd  0*8(%[a0],%[lda2],2), %%ymm0 \n\t" // [a04,a14,a24,a34]
	          "vmovupd  0*8(%[a1],%[lda2],2), %%ymm1 \n\t" // [a05,a15,a25,a35]
	          "vmovupd  0*8(%[a0],%[lda3],2), %%ymm2 \n\t" // [a06,a16,a26,a36]
	          "vmovupd  0*8(%[a1],%[lda3],2), %%ymm3 \n\t" // [a07,a17,a27,a37]
	          "\n\t"
	          "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm4 \n\t" // [a04,a05,a24,a25]
	          "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm5 \n\t" // [a14,a15,a34,a35]
	          "vshufpd    $0x00 , %%ymm3 , %%ymm2 , %%ymm6 \n\t" // [a06,a07,a26,a27]
	          "vshufpd    $0x0f , %%ymm3 , %%ymm2 , %%ymm7 \n\t" // [a16,a17,a36,a37]
	          "\n\t"
	          "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm12\n\t" // [a04,a05,a14,a15]
	          "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm13\n\t" // [a24,a25,a34,a35]
	          "vperm2f128 $0x20 , %%ymm7 , %%ymm6 , %%ymm14\n\t" // [a06,a07,a16,a17]
	          "vperm2f128 $0x31 , %%ymm7 , %%ymm6 , %%ymm15\n\t" // [a26,a27,a36,a37]
	          "\n\t"
	          "vmovapd  %%ymm12, 16*8(%[a2]) \n\t"
	          "vmovapd  %%ymm13, 20*8(%[a2]) \n\t"
	          "vmovapd  %%ymm14, 24*8(%[a2]) \n\t"
	          "vmovapd  %%ymm15, 28*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $4*8 , %[a0]\n\t"
	          "addq  $4*8 , %[a1]\n\t"
	          "addq  $32*8, %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_4)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    if( MR & 2 ){

	        __asm__ __volatile__ (
	          "movupd  0*8(%[a0]          ), %%xmm0 \n\t" // [a00,a10,---,---]
	          "movupd  0*8(%[a1]          ), %%xmm1 \n\t" // [a01,a11,---,---]
	          "movupd  0*8(%[a0],%[lda2]  ), %%xmm2 \n\t" // [a02,a12,---,---]
	          "movupd  0*8(%[a1],%[lda2]  ), %%xmm3 \n\t" // [a03,a13,---,---]
	          "\n\t"
	          "vshufpd    $0x00 , %%xmm1 , %%xmm0 , %%xmm12\n\t" // [a00,a01,---,---]
	          "vshufpd    $0x0f , %%xmm1 , %%xmm0 , %%xmm13\n\t" // [a10,a11,---,---]
	          "vshufpd    $0x00 , %%xmm3 , %%xmm2 , %%xmm14\n\t" // [a02,a03,---,---]
	          "vshufpd    $0x0f , %%xmm3 , %%xmm2 , %%xmm15\n\t" // [a12,a13,---,---]
	          "\n\t"
	          "movapd  %%xmm12,  0*8(%[a2]) \n\t"
	          "movapd  %%xmm13,  2*8(%[a2]) \n\t"
	          "movapd  %%xmm14,  4*8(%[a2]) \n\t"
	          "movapd  %%xmm15,  6*8(%[a2]) \n\t"
	          "\n\t"
	          "movupd  0*8(%[a0],%[lda2],2), %%xmm0 \n\t" // [a04,a14,---,---]
	          "movupd  0*8(%[a1],%[lda2],2), %%xmm1 \n\t" // [a05,a15,---,---]
	          "movupd  0*8(%[a0],%[lda3],2), %%xmm2 \n\t" // [a06,a16,---,---]
	          "movupd  0*8(%[a1],%[lda3],2), %%xmm3 \n\t" // [a07,a17,---,---]
	          "\n\t"
	          "vshufpd    $0x00 , %%xmm1 , %%xmm0 , %%xmm12\n\t" // [a04,a05,---,---]
	          "vshufpd    $0x0f , %%xmm1 , %%xmm0 , %%xmm13\n\t" // [a14,a15,---,---]
	          "vshufpd    $0x00 , %%xmm3 , %%xmm2 , %%xmm14\n\t" // [a06,a07,---,---]
	          "vshufpd    $0x0f , %%xmm3 , %%xmm2 , %%xmm15\n\t" // [a16,a17,---,---]
	          "\n\t"
	          "movapd  %%xmm12,  8*8(%[a2]) \n\t"
	          "movapd  %%xmm13, 10*8(%[a2]) \n\t"
	          "movapd  %%xmm14, 12*8(%[a2]) \n\t"
	          "movapd  %%xmm15, 14*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $2*8 , %[a0]\n\t"
	          "addq  $2*8 , %[a1]\n\t"
	          "addq  $16*8, %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    if( MR & 1 ){

	        __asm__ __volatile__ (
	          "movsd  0*8(%[a0]          ), %%xmm0 \n\t" // [a00,  0,---,---]
	          "movsd  0*8(%[a1]          ), %%xmm1 \n\t" // [a01,  0,---,---]
	          "movsd  0*8(%[a0],%[lda2]  ), %%xmm2 \n\t" // [a02,  0,---,---]
	          "movsd  0*8(%[a1],%[lda2]  ), %%xmm3 \n\t" // [a03,  0,---,---]
	          "movsd  0*8(%[a0],%[lda2],2), %%xmm4 \n\t" // [a04,  0,---,---]
	          "movsd  0*8(%[a1],%[lda2],2), %%xmm5 \n\t" // [a05,  0,---,---]
	          "movsd  0*8(%[a0],%[lda3],2), %%xmm6 \n\t" // [a06,  0,---,---]
	          "movsd  0*8(%[a1],%[lda3],2), %%xmm7 \n\t" // [a07,  0,---,---]
	          "\n\t"
	          "movsd  %%xmm0 ,  0*8(%[a2]) \n\t"
	          "movsd  %%xmm1 ,  1*8(%[a2]) \n\t"
	          "movsd  %%xmm2 ,  2*8(%[a2]) \n\t"
	          "movsd  %%xmm3 ,  3*8(%[a2]) \n\t"
	          "movsd  %%xmm4 ,  4*8(%[a2]) \n\t"
	          "movsd  %%xmm5 ,  5*8(%[a2]) \n\t"
	          "movsd  %%xmm6 ,  6*8(%[a2]) \n\t"
	          "movsd  %%xmm7 ,  7*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $1*8 , %[a0]\n\t"
	          "addq  $1*8 , %[a1]\n\t"
	          "addq  $8*8 , %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_1)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    a0 = a0 - M + 4*2*lda;
	    a1 = a1 - M + 4*2*lda;


	  }
	}
	if( K & 4  ){

	    if( MQ ){
	      size_t m6 = MQ;
	      while( m6-- ){

	        __asm__ __volatile__ (
	          "vmovupd  0*8(%[a0]          ), %%ymm0 \n\t" // [a00,a10,a20,a30]
	          "vmovupd  0*8(%[a1]          ), %%ymm1 \n\t" // [a01,a11,a21,a31]
	          "vmovupd  4*8(%[a0]          ), %%xmm2 \n\t" // [a40,a50,---,---]
	          "vmovupd  4*8(%[a1]          ), %%xmm3 \n\t" // [a41,a51,---,---]
	          "\n\t"
	          "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm4 \n\t" // [a00,a01,a20,a21]
	          "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm5 \n\t" // [a10,a11,a30,a31]
	          "vshufpd    $0x00 , %%ymm3 , %%ymm2 , %%ymm6 \n\t" // [a40,a41,---,---]
	          "vshufpd    $0x0f , %%ymm3 , %%ymm2 , %%ymm7 \n\t" // [a50,a51,---,---]
	          "\n\t"
	          "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm10\n\t" // [a00,a01,a10,a11]
	          "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm11\n\t" // [a20,a21,a30,a31]
	          "vperm2f128 $0x20 , %%ymm7 , %%ymm6 , %%ymm12\n\t" // [a40,a41,a50,a51]
	          "\n\t"
	          "vmovupd  %%ymm10,  0*8(%[a2]) \n\t"
	          "vmovupd  %%ymm11,  4*8(%[a2]) \n\t"
	          "vmovupd  %%ymm12,  8*8(%[a2]) \n\t"
	          "\n\t"
	          "vmovupd  0*8(%[a0],%[lda2]  ), %%ymm0 \n\t" // [a02,a12,a22,a32]
	          "vmovupd  0*8(%[a1],%[lda2]  ), %%ymm1 \n\t" // [a03,a13,a23,a33]
	          "vmovupd  4*8(%[a0],%[lda2]  ), %%xmm2 \n\t" // [a42,a52,---,---]
	          "vmovupd  4*8(%[a1],%[lda2]  ), %%xmm3 \n\t" // [a43,a53,---,---]
	          "\n\t"
	          "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm4 \n\t" // [a02,a03,a22,a23]
	          "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm5 \n\t" // [a12,a13,a32,a33]
	          "vshufpd    $0x00 , %%ymm3 , %%ymm2 , %%ymm6 \n\t" // [a42,a43,---,---]
	          "vshufpd    $0x0f , %%ymm3 , %%ymm2 , %%ymm7 \n\t" // [a52,a53,---,---]
	          "\n\t"
	          "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm13\n\t" // [a02,a03,a12,a13]
	          "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm14\n\t" // [a22,a23,a32,a33]
	          "vperm2f128 $0x20 , %%ymm7 , %%ymm6 , %%ymm15\n\t" // [a42,a43,a52,a53]
	          "\n\t"
	          "vmovupd  %%ymm13, 12*8(%[a2]) \n\t"
	          "vmovupd  %%ymm14, 16*8(%[a2]) \n\t"
	          "vmovupd  %%ymm15, 20*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $6*8, %[a0]\n\t"
	          "addq  $6*8, %[a1]\n\t"
	          "leaq  0*8(%[a2],%[K3],2), %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3),[K3]"r"(K3));

	      }
	      A2 = A2 - MM*K + 6*2*2; 
	    }
	    if( MR & 4 ){

	        __asm__ __volatile__ (
	          "vmovupd  0*8(%[a0]          ), %%ymm0 \n\t" // [a00,a10,a20,a30]
	          "vmovupd  0*8(%[a1]          ), %%ymm1 \n\t" // [a01,a11,a21,a31]
	          "vmovupd  0*8(%[a0],%[lda2]  ), %%ymm2 \n\t" // [a02,a12,a22,a32]
	          "vmovupd  0*8(%[a1],%[lda2]  ), %%ymm3 \n\t" // [a03,a13,a23,a33]
	          "\n\t"
	          "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm4 \n\t" // [a00,a01,a20,a21]
	          "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm5 \n\t" // [a10,a11,a30,a31]
	          "vshufpd    $0x00 , %%ymm3 , %%ymm2 , %%ymm6 \n\t" // [a02,a03,a22,a23]
	          "vshufpd    $0x0f , %%ymm3 , %%ymm2 , %%ymm7 \n\t" // [a12,a13,a32,a33]
	          "\n\t"
	          "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm12\n\t" // [a00,a01,a10,a11]
	          "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm13\n\t" // [a20,a21,a30,a31]
	          "vperm2f128 $0x20 , %%ymm7 , %%ymm6 , %%ymm14\n\t" // [a02,a03,a12,a13]
	          "vperm2f128 $0x31 , %%ymm7 , %%ymm6 , %%ymm15\n\t" // [a22,a23,a32,a33]
	          "\n\t"
	          "vmovapd  %%ymm12,  0*8(%[a2]) \n\t"
	          "vmovapd  %%ymm13,  4*8(%[a2]) \n\t"
	          "vmovapd  %%ymm14,  8*8(%[a2]) \n\t"
	          "vmovapd  %%ymm15, 12*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $4*8 , %[a0]\n\t"
	          "addq  $4*8 , %[a1]\n\t"
	          "addq  $16*8, %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_4)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    if( MR & 2 ){

	        __asm__ __volatile__ (
	          "movupd  0*8(%[a0]          ), %%xmm0 \n\t" // [a00,a10,---,---]
	          "movupd  0*8(%[a1]          ), %%xmm1 \n\t" // [a01,a11,---,---]
	          "movupd  0*8(%[a0],%[lda2]  ), %%xmm2 \n\t" // [a02,a12,---,---]
	          "movupd  0*8(%[a1],%[lda2]  ), %%xmm3 \n\t" // [a03,a13,---,---]
	          "\n\t"
	          "vshufpd    $0x00 , %%xmm1 , %%xmm0 , %%xmm12\n\t" // [a00,a01,---,---]
	          "vshufpd    $0x0f , %%xmm1 , %%xmm0 , %%xmm13\n\t" // [a10,a11,---,---]
	          "vshufpd    $0x00 , %%xmm3 , %%xmm2 , %%xmm14\n\t" // [a02,a03,---,---]
	          "vshufpd    $0x0f , %%xmm3 , %%xmm2 , %%xmm15\n\t" // [a12,a13,---,---]
	          "\n\t"
	          "movapd  %%xmm12,  0*8(%[a2]) \n\t"
	          "movapd  %%xmm13,  2*8(%[a2]) \n\t"
	          "movapd  %%xmm14,  4*8(%[a2]) \n\t"
	          "movapd  %%xmm15,  6*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $2*8 , %[a0]\n\t"
	          "addq  $2*8 , %[a1]\n\t"
	          "addq  $8*8 , %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    if( MR & 1 ){

	        __asm__ __volatile__ (
	          "movsd  0*8(%[a0]          ), %%xmm0 \n\t" // [a00,  0,---,---]
	          "movsd  0*8(%[a1]          ), %%xmm1 \n\t" // [a01,  0,---,---]
	          "movsd  0*8(%[a0],%[lda2]  ), %%xmm2 \n\t" // [a02,  0,---,---]
	          "movsd  0*8(%[a1],%[lda2]  ), %%xmm3 \n\t" // [a03,  0,---,---]
	          "\n\t"
	          "movsd  %%xmm0 ,  0*8(%[a2]) \n\t"
	          "movsd  %%xmm1 ,  1*8(%[a2]) \n\t"
	          "movsd  %%xmm2 ,  2*8(%[a2]) \n\t"
	          "movsd  %%xmm3 ,  3*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $1*8 , %[a0]\n\t"
	          "addq  $1*8 , %[a1]\n\t"
	          "addq  $4*8 , %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_1)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    a0 = a0 - M + 2*2*lda;
	    a1 = a1 - M + 2*2*lda;


	}
	if( K & 2  ){

	    if( MQ ){
	      size_t m6 = MQ;
	      while( m6-- ){

	        __asm__ __volatile__ (
	          "vmovupd  0*8(%[a0]          ), %%ymm0 \n\t" // [a00,a10,a20,a30]
	          "vmovupd  0*8(%[a1]          ), %%ymm1 \n\t" // [a01,a11,a21,a31]
	          "vmovupd  4*8(%[a0]          ), %%xmm2 \n\t" // [a40,a50,---,---]
	          "vmovupd  4*8(%[a1]          ), %%xmm3 \n\t" // [a41,a51,---,---]
	          "\n\t"
	          "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm4 \n\t" // [a00,a01,a20,a21]
	          "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm5 \n\t" // [a10,a11,a30,a31]
	          "vshufpd    $0x00 , %%ymm3 , %%ymm2 , %%ymm6 \n\t" // [a40,a41,---,---]
	          "vshufpd    $0x0f , %%ymm3 , %%ymm2 , %%ymm7 \n\t" // [a50,a51,---,---]
	          "\n\t"
	          "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm10\n\t" // [a00,a01,a10,a11]
	          "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm11\n\t" // [a20,a21,a30,a31]
	          "vperm2f128 $0x20 , %%ymm7 , %%ymm6 , %%ymm12\n\t" // [a40,a41,a50,a51]
	          "\n\t"
	          "vmovupd  %%ymm10,  0*8(%[a2]) \n\t"
	          "vmovupd  %%ymm11,  4*8(%[a2]) \n\t"
	          "vmovupd  %%ymm12,  8*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $6*8, %[a0]\n\t"
	          "addq  $6*8, %[a1]\n\t"
	          "leaq  0*8(%[a2],%[K3],2), %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3),[K3]"r"(K3));

	      }
	      A2 = A2 - MM*K + 6*1*2; 
	    }
	    if( MR & 4 ){

	        __asm__ __volatile__ (
	          "vmovupd  0*8(%[a0]          ), %%ymm0 \n\t" // [a00,a10,a20,a30]
	          "vmovupd  0*8(%[a1]          ), %%ymm1 \n\t" // [a01,a11,a21,a31]
	          "\n\t"
	          "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm4 \n\t" // [a00,a01,a20,a21]
	          "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm5 \n\t" // [a10,a11,a30,a31]
	          "\n\t"
	          "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm12\n\t" // [a00,a01,a10,a11]
	          "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm13\n\t" // [a20,a21,a30,a31]
	          "\n\t"
	          "vmovapd  %%ymm12,  0*8(%[a2]) \n\t"
	          "vmovapd  %%ymm13,  4*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $4*8 , %[a0]\n\t"
	          "addq  $4*8 , %[a1]\n\t"
	          "addq  $8*8 , %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_4)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    if( MR & 2 ){

	        __asm__ __volatile__ (
	          "movupd  0*8(%[a0]          ), %%xmm0 \n\t" // [a00,a10,---,---]
	          "movupd  0*8(%[a1]          ), %%xmm1 \n\t" // [a01,a11,---,---]
	          "\n\t"
	          "vshufpd    $0x00 , %%xmm1 , %%xmm0 , %%xmm12\n\t" // [a00,a01,---,---]
	          "vshufpd    $0x0f , %%xmm1 , %%xmm0 , %%xmm13\n\t" // [a10,a11,---,---]
	          "\n\t"
	          "movapd  %%xmm12,  0*8(%[a2]) \n\t"
	          "movapd  %%xmm13,  2*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $2*8 , %[a0]\n\t"
	          "addq  $2*8 , %[a1]\n\t"
	          "addq  $4*8 , %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    if( MR & 1 ){

	        __asm__ __volatile__ (
	          "movsd  0*8(%[a0]          ), %%xmm0 \n\t" // [a00,  0,---,---]
	          "movsd  0*8(%[a1]          ), %%xmm1 \n\t" // [a01,  0,---,---]
	          "\n\t"
	          "movsd  %%xmm0 ,  0*8(%[a2]) \n\t"
	          "movsd  %%xmm1 ,  1*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $1*8 , %[a0]\n\t"
	          "addq  $1*8 , %[a1]\n\t"
	          "addq  $2*8 , %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_1)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    a0 = a0 - M + 1*2*lda;
	    a1 = a1 - M + 1*2*lda;


	}
	if( K & 1  ){

	    if( MQ ){
	      size_t m6 = MQ;
	      while( m6-- ){

	        __asm__ __volatile__ (
	          "vmovupd  0*8(%[a0]          ), %%ymm0 \n\t" // [a00,a10,a20,a30]
	          "vmovupd  4*8(%[a0]          ), %%xmm2 \n\t" // [a40,a50,---,---]
	          "\n\t"
	          "vmovupd  %%ymm0 ,  0*8(%[a2]) \n\t"
	          "vmovupd  %%xmm2 ,  4*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $6*8, %[a0]\n\t"
	          "addq  $6*8, %[a1]\n\t"
	          "leaq  0*8(%[a2],%[K3],2), %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3),[K3]"r"(K3));

	      }
	      A2 = A2 - MM*K + 6*1*1; 
	    }
	    if( MR & 4 ){

	        __asm__ __volatile__ (
	          "vmovupd  0*8(%[a0]          ), %%ymm0 \n\t" // [a00,a10,a20,a30]
	          "\n\t"
	          "vmovapd  %%ymm0 ,  0*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $4*8 , %[a0]\n\t"
	          "addq  $4*8 , %[a1]\n\t"
	          "addq  $4*8 , %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_4)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    if( MR & 2 ){

	        __asm__ __volatile__ (
	          "movupd  0*8(%[a0]          ), %%xmm0 \n\t" // [a00,a10,---,---]
	          "\n\t"
	          "movapd  %%xmm0 ,  0*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $2*8 , %[a0]\n\t"
	          "addq  $2*8 , %[a1]\n\t"
	          "addq  $2*8 , %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    if( MR & 1 ){

	        __asm__ __volatile__ (
	          "movsd  0*8(%[a0]          ), %%xmm0 \n\t" // [a00,  0,---,---]
	          "\n\t"
	          "movsd  %%xmm0 ,  0*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $1*8 , %[a0]\n\t"
	          "addq  $1*8 , %[a1]\n\t"
	          "addq  $1*8 , %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_1)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	    }
	    a0 = a0 - M + 1*1*lda;
	    a1 = a1 - M + 1*1*lda;


	}

}

