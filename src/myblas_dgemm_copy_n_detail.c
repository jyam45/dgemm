#include "myblas_internal.h"

#define COPY_N(NU,KU,LU) \
	        for( size_t k=0; k<KU; k++ ){\
	          for( size_t j=0; j<NU; j++ ){\
	            for( size_t l=0; l<LU; l++ ){\
	              *B2=*(B+l+k*LU+j*ldb);\
	              B2++;\
	            }\
	          }\
	        }\
	        B+=KU*LU;\
	        b0+=KU*LU;\
	        b1+=KU*LU;

#define COPY_N_K4(NU) \
	    if( K >> 2 ){\
	      size_t k4 = ( K >> 2 );\
	      while( k4-- ){\
	        COPY_N(NU,2,2);\
	      }\
	    }\
	    if( K & 2 ){\
	        COPY_N(NU,1,2);\
	    }\
	    if( K & 1 ){\
	        COPY_N(NU,1,1);\
	    }\
	    B  = B  - K + NU*ldb ;\
	    b0 = b0 - K + NU*ldb ;\
	    b1 = b1 - K + NU*ldb ;


#define COPY_N_K8(NU) \
	    if( K >> 3 ){\
	      size_t k8 = ( K >> 3 );\
	      while( k8-- ){\
	        COPY_N(NU,4,2);\
	      }\
	    }\
	    if( K & 4 ){\
	        COPY_N(NU,2,2);\
	    }\
	    if( K & 2 ){\
	        COPY_N(NU,1,2);\
	    }\
	    if( K & 1 ){\
	        COPY_N(NU,1,1);\
	    }\
	    B  = B  - K + NU*ldb ;\
	    b0 = b0 - K + NU*ldb ;\
	    b1 = b1 - K + NU*ldb ;


void myblas_dgemm_copy_n_MxK(size_t K, size_t N, const double* B, size_t k, size_t j,  size_t ldb, double* B2 ){

	B = B + k + ldb*j; // start point

	const double* b0 = B;
	const double* b1 = B + ldb;
	size_t        ldb2 = ldb * 2 * sizeof(double);
	size_t        ldb3 = ldb * 3 * sizeof(double);

	if( N >> 2 ){
	  size_t n4 = ( N >> 2 );
	  while( n4-- ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "prefetcht0  0*8(%[b0],%[ldb2],2) \n\t"
	            "prefetcht0  0*8(%[b1],%[ldb2],2) \n\t"
	            "prefetcht0  0*8(%[b0],%[ldb3],2) \n\t"
	            "prefetcht0  0*8(%[b1],%[ldb3],2) \n\t"
	            "\n\t"
	            "vmovupd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	            "vmovupd  0*8(%[b1]        ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	            "vmovupd  0*8(%[b0],%[ldb2]), %%ymm2 \n\t" // [x02,x12,x22,x32]
	            "vmovupd  0*8(%[b1],%[ldb2]), %%ymm3 \n\t" // [x03,x13,x23,x33]
	            "vmovupd  4*8(%[b0]        ), %%ymm4 \n\t" // [x40,x50,x60,x70]
	            "vmovupd  4*8(%[b1]        ), %%ymm5 \n\t" // [x41,x51,x61,x71]
	            "vmovupd  4*8(%[b0],%[ldb2]), %%ymm6 \n\t" // [x42,x52,x62,x72]
	            "vmovupd  4*8(%[b1],%[ldb2]), %%ymm7 \n\t" // [x43,x53,x63,x73]
	            "\n\t"
	            "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [x00,x10,x01,x11]
	            "vperm2f128  $0x20, %%ymm3 , %%ymm2 , %%ymm9 \n\t" // [x02,x12,x03,x13]
	            "vperm2f128  $0x31, %%ymm1 , %%ymm0 , %%ymm10\n\t" // [x20,x30,x21,x31]
	            "vperm2f128  $0x31, %%ymm3 , %%ymm2 , %%ymm11\n\t" // [x22,x32,x23,x33]
	            "vperm2f128  $0x20, %%ymm5 , %%ymm4 , %%ymm12\n\t" // [x40,x50,x41,x51]
	            "vperm2f128  $0x20, %%ymm7 , %%ymm6 , %%ymm13\n\t" // [x42,x52,x43,x53]
	            "vperm2f128  $0x31, %%ymm5 , %%ymm4 , %%ymm14\n\t" // [x60,x70,x61,x71]
	            "vperm2f128  $0x31, %%ymm7 , %%ymm6 , %%ymm15\n\t" // [x62,x72,x63,x73]
	            "\n\t"
	            "vmovapd  %%ymm8 ,   0*8(%[b2])\n\t"
	            "vmovapd  %%ymm9 ,   4*8(%[b2])\n\t"
	            "vmovapd  %%ymm10,   8*8(%[b2])\n\t"
	            "vmovapd  %%ymm11,  12*8(%[b2])\n\t"
	            "vmovapd  %%ymm12,  16*8(%[b2])\n\t"
	            "vmovapd  %%ymm13,  20*8(%[b2])\n\t"
	            "vmovapd  %%ymm14,  24*8(%[b2])\n\t"
	            "vmovapd  %%ymm15,  28*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $8*8 , %[b0]\n\t"
	            "addq  $8*8 , %[b1]\n\t"
	            "addq  $32*8, %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2),[ldb3]"r"(ldb3));

	      }
	    }
	    if( K & 4 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "prefetcht0  0*8(%[b0],%[ldb2],2) \n\t"
	            "prefetcht0  0*8(%[b1],%[ldb2],2) \n\t"
	            "prefetcht0  0*8(%[b0],%[ldb3],2) \n\t"
	            "prefetcht0  0*8(%[b1],%[ldb3],2) \n\t"
	            "\n\t"
	            "vmovupd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	            "vmovupd  0*8(%[b1]        ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	            "vmovupd  0*8(%[b0],%[ldb2]), %%ymm2 \n\t" // [x02,x12,x22,x32]
	            "vmovupd  0*8(%[b1],%[ldb2]), %%ymm3 \n\t" // [x03,x13,x23,x33]
	            "\n\t"
	            "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [x00,x10,x01,x11]
	            "vperm2f128  $0x20, %%ymm3 , %%ymm2 , %%ymm9 \n\t" // [x02,x12,x03,x13]
	            "vperm2f128  $0x31, %%ymm1 , %%ymm0 , %%ymm10\n\t" // [x20,x30,x21,x31]
	            "vperm2f128  $0x31, %%ymm3 , %%ymm2 , %%ymm11\n\t" // [x22,x32,x23,x33]
	            "\n\t"
	            "vmovapd  %%ymm8 ,   0*8(%[b2])\n\t"
	            "vmovapd  %%ymm9 ,   4*8(%[b2])\n\t"
	            "vmovapd  %%ymm10,   8*8(%[b2])\n\t"
	            "vmovapd  %%ymm11,  12*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $4*8 , %[b0]\n\t"
	            "addq  $4*8 , %[b1]\n\t"
	            "addq  $16*8, %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2),[ldb3]"r"(ldb3));

	    }
	    if( K & 2 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd  0*8(%[b0]        ), %%xmm0 \n\t" // [x00,x10,---,---]
	            "vmovupd  0*8(%[b1]        ), %%xmm1 \n\t" // [x01,x11,---,---]
	            "vmovupd  0*8(%[b0],%[ldb2]), %%xmm2 \n\t" // [x02,x12,---,---]
	            "vmovupd  0*8(%[b1],%[ldb2]), %%xmm3 \n\t" // [x03,x13,---,---]
	            "\n\t"
	            "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [x00,x10,x01,x11]
	            "vperm2f128  $0x20, %%ymm3 , %%ymm2 , %%ymm9 \n\t" // [x02,x12,x03,x13]
	            "\n\t"
	            "vmovapd  %%ymm8 ,   0*8(%[b2])\n\t"
	            "vmovapd  %%ymm9 ,   4*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $2*8 , %[b0]\n\t"
	            "addq  $2*8 , %[b1]\n\t"
	            "addq  $8*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	    }
	    if( K & 1 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "movlpd  0*8(%[b0]        ), %%xmm0 \n\t" // [x00,---,---,---]
	            "movhpd  0*8(%[b1]        ), %%xmm0 \n\t" // [x00,x01,---,---]
	            "movlpd  0*8(%[b0],%[ldb2]), %%xmm2 \n\t" // [x02,---,---,---]
	            "movhpd  0*8(%[b1],%[ldb2]), %%xmm2 \n\t" // [x02,x03,---,---]
	            "\n\t"
	            "vperm2f128  $0x20, %%ymm2 , %%ymm0 , %%ymm8 \n\t" // [x00,x01,x02,x03]
	            "\n\t"
	            "vmovapd  %%ymm8 ,   0*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $1*8 , %[b0]\n\t"
	            "addq  $1*8 , %[b1]\n\t"
	            "addq  $4*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	    }
	    b0 = b0 - K + 4*ldb ;
	    b1 = b1 - K + 4*ldb ;

	  }
	}
	if( N & 2 ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	            "vmovupd  0*8(%[b1]        ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	            "vmovupd  4*8(%[b0]        ), %%ymm4 \n\t" // [x40,x50,x60,x70]
	            "vmovupd  4*8(%[b1]        ), %%ymm5 \n\t" // [x41,x51,x61,x71]
	            "\n\t"
	            "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [x00,x10,x01,x11]
	            "vperm2f128  $0x31, %%ymm1 , %%ymm0 , %%ymm10\n\t" // [x20,x30,x21,x31]
	            "vperm2f128  $0x20, %%ymm5 , %%ymm4 , %%ymm12\n\t" // [x40,x50,x41,x51]
	            "vperm2f128  $0x31, %%ymm5 , %%ymm4 , %%ymm14\n\t" // [x60,x70,x61,x71]
	            "\n\t"
	            "vmovapd  %%ymm8 ,   0*8(%[b2])\n\t"
	            "vmovapd  %%ymm10,   4*8(%[b2])\n\t"
	            "vmovapd  %%ymm12,   8*8(%[b2])\n\t"
	            "vmovapd  %%ymm14,  12*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $8*8 , %[b0]\n\t"
	            "addq  $8*8 , %[b1]\n\t"
	            "addq  $16*8, %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	      }
	    }
	    if( K & 4 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	            "vmovupd  0*8(%[b1]        ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	            "\n\t"
	            "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [x00,x10,x01,x11]
	            "vperm2f128  $0x31, %%ymm1 , %%ymm0 , %%ymm10\n\t" // [x20,x30,x21,x31]
	            "\n\t"
	            "vmovapd  %%ymm8 ,   0*8(%[b2])\n\t"
	            "vmovapd  %%ymm10,   4*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $4*8 , %[b0]\n\t"
	            "addq  $4*8 , %[b1]\n\t"
	            "addq  $8*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	    }
	    if( K & 2 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd  0*8(%[b0]        ), %%xmm0 \n\t" // [x00,x10,---,---]
	            "vmovupd  0*8(%[b1]        ), %%xmm1 \n\t" // [x01,x11,---,---]
	            "\n\t"
	            "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [x00,x10,x01,x11]
	            "\n\t"
	            "vmovapd  %%ymm8 ,   0*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $2*8 , %[b0]\n\t"
	            "addq  $2*8 , %[b1]\n\t"
	            "addq  $4*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	    }
	    if( K & 1 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "movlpd  0*8(%[b0]        ), %%xmm0 \n\t" // [x00,---,---,---]
	            "movhpd  0*8(%[b1]        ), %%xmm0 \n\t" // [x00,x01,---,---]
	            "\n\t"
	            "movapd  %%xmm0 ,   0*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $1*8 , %[b0]\n\t"
	            "addq  $1*8 , %[b1]\n\t"
	            "addq  $2*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	    }
	    b0 = b0 - K + 2*ldb ;
	    b1 = b1 - K + 2*ldb ;

	}
	if( N & 1 ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	            "vmovupd  4*8(%[b0]        ), %%ymm4 \n\t" // [x40,x50,x60,x70]
	            "\n\t"
	            "vmovupd  %%ymm0 ,   0*8(%[b2])\n\t"
	            "vmovupd  %%ymm4 ,   4*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $8*8 , %[b0]\n\t"
	            "addq  $8*8 , %[b1]\n\t"
	            "addq  $8*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	      }
	    }
	    if( K & 4 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	            "\n\t"
	            "vmovupd  %%ymm0 ,   0*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $4*8 , %[b0]\n\t"
	            "addq  $4*8 , %[b1]\n\t"
	            "addq  $4*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	    }
	    if( K & 2 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd  0*8(%[b0]        ), %%xmm0 \n\t" // [x00,x10,---,---]
	            "\n\t"
	            "movupd  %%xmm0 ,   0*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $2*8 , %[b0]\n\t"
	            "addq  $2*8 , %[b1]\n\t"
	            "addq  $2*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));


	    }
	    if( K & 1 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "movlpd  0*8(%[b0]        ), %%xmm0 \n\t" // [x00,---,---,---]
	            "\n\t"
	            "movlpd  %%xmm0 ,   0*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $1*8 , %[b0]\n\t"
	            "addq  $1*8 , %[b1]\n\t"
	            "addq  $1*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	    }
	    b0 = b0 - K + 1*ldb ;
	    b1 = b1 - K + 1*ldb ;

	}

}

void myblas_dgemm_copy_n_NxK(size_t K, size_t N, const double* B, size_t k, size_t j,  size_t ldb, double* B2 ){

	B = B + k + ldb*j; // start point

	size_t NQ = N/6;
	size_t NR = N%6;

	const double* b0 = B;
	const double* b1 = B + ldb;
	size_t        ldb2 = ldb * 2 * sizeof(double);
	size_t        ldb3 = ldb * 3 * sizeof(double);
	size_t        ldb5 = ldb * 5 * sizeof(double);

	if( NQ ){
	  size_t n6 = NQ;
	  while( n6-- ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "prefetcht0  0*8(%[b0],%[ldb3],2) \n\t"
	            "prefetcht0  0*8(%[b1],%[ldb3],2) \n\t"
	            "prefetcht0  0*8(%[b0],%[ldb2],4) \n\t"
	            "prefetcht0  0*8(%[b1],%[ldb2],4) \n\t"
	            "prefetcht0  0*8(%[b0],%[ldb5],2) \n\t"
	            "prefetcht0  0*8(%[b1],%[ldb5],2) \n\t"
	            "\n\t"
	            "vmovupd  0*8(%[b0]          ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	            "vmovupd  0*8(%[b1]          ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	            "vmovupd  0*8(%[b0],%[ldb2]  ), %%ymm2 \n\t" // [x02,x12,x22,x32]
	            "vmovupd  0*8(%[b1],%[ldb2]  ), %%ymm3 \n\t" // [x03,x13,x23,x33]
	            "vmovupd  0*8(%[b0],%[ldb2],2), %%ymm4 \n\t" // [x04,x14,x24,x34]
	            "vmovupd  0*8(%[b1],%[ldb2],2), %%ymm5 \n\t" // [x05,x15,x25,x35]
	            "\n\t"
	            "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm10\n\t" // [x00,x10,x01,x11]
	            "vperm2f128  $0x20, %%ymm3 , %%ymm2 , %%ymm11\n\t" // [x02,x12,x03,x13]
	            "vperm2f128  $0x20, %%ymm5 , %%ymm4 , %%ymm12\n\t" // [x04,x14,x05,x15]
	            "vperm2f128  $0x31, %%ymm1 , %%ymm0 , %%ymm13\n\t" // [x20,x30,x21,x31]
	            "vperm2f128  $0x31, %%ymm3 , %%ymm2 , %%ymm14\n\t" // [x22,x32,x23,x33]
	            "vperm2f128  $0x31, %%ymm5 , %%ymm4 , %%ymm15\n\t" // [x24,x34,x25,x35]
	            "\n\t"
	            "vmovupd  %%ymm10,   0*8(%[b2])\n\t"
	            "vmovupd  %%ymm11,   4*8(%[b2])\n\t"
	            "vmovupd  %%ymm12,   8*8(%[b2])\n\t"
	            "vmovupd  %%ymm13,  12*8(%[b2])\n\t"
	            "vmovupd  %%ymm14,  16*8(%[b2])\n\t"
	            "vmovupd  %%ymm15,  20*8(%[b2])\n\t"
	            "\n\t"
	            "vmovupd  4*8(%[b0]          ), %%ymm0 \n\t" // [x40,x50,x60,x70]
	            "vmovupd  4*8(%[b1]          ), %%ymm1 \n\t" // [x41,x51,x61,x71]
	            "vmovupd  4*8(%[b0],%[ldb2]  ), %%ymm2 \n\t" // [x42,x52,x62,x72]
	            "vmovupd  4*8(%[b1],%[ldb2]  ), %%ymm3 \n\t" // [x43,x53,x63,x73]
	            "vmovupd  4*8(%[b0],%[ldb2],2), %%ymm4 \n\t" // [x44,x54,x64,x74]
	            "vmovupd  4*8(%[b1],%[ldb2],2), %%ymm5 \n\t" // [x45,x55,x65,x75]
	            "\n\t"
	            "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm10\n\t" // [x40,x50,x41,x51]
	            "vperm2f128  $0x20, %%ymm3 , %%ymm2 , %%ymm11\n\t" // [x42,x52,x43,x53]
	            "vperm2f128  $0x20, %%ymm5 , %%ymm4 , %%ymm12\n\t" // [x42,x52,x43,x53]
	            "vperm2f128  $0x31, %%ymm1 , %%ymm0 , %%ymm13\n\t" // [x60,x70,x61,x71]
	            "vperm2f128  $0x31, %%ymm3 , %%ymm2 , %%ymm14\n\t" // [x62,x72,x63,x73]
	            "vperm2f128  $0x31, %%ymm5 , %%ymm4 , %%ymm15\n\t" // [x62,x72,x63,x73]
	            "\n\t"
	            "vmovupd  %%ymm10,  24*8(%[b2])\n\t"
	            "vmovupd  %%ymm11,  28*8(%[b2])\n\t"
	            "vmovupd  %%ymm12,  32*8(%[b2])\n\t"
	            "vmovupd  %%ymm13,  36*8(%[b2])\n\t"
	            "vmovupd  %%ymm14,  40*8(%[b2])\n\t"
	            "vmovupd  %%ymm15,  44*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $8*8 , %[b0]\n\t"
	            "addq  $8*8 , %[b1]\n\t"
	            "addq  $48*8, %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2),[ldb3]"r"(ldb3),[ldb5]"r"(ldb5));

	      }
	    }
	    if( K & 4 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "prefetcht0  0*8(%[b0],%[ldb3],2) \n\t"
	            "prefetcht0  0*8(%[b1],%[ldb3],2) \n\t"
	            "prefetcht0  0*8(%[b0],%[ldb2],4) \n\t"
	            "prefetcht0  0*8(%[b1],%[ldb2],4) \n\t"
	            "prefetcht0  0*8(%[b0],%[ldb5],2) \n\t"
	            "prefetcht0  0*8(%[b1],%[ldb5],2) \n\t"
	            "\n\t"
	            "vmovupd  0*8(%[b0]          ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	            "vmovupd  0*8(%[b1]          ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	            "vmovupd  0*8(%[b0],%[ldb2]  ), %%ymm2 \n\t" // [x02,x12,x22,x32]
	            "vmovupd  0*8(%[b1],%[ldb2]  ), %%ymm3 \n\t" // [x03,x13,x23,x33]
	            "vmovupd  0*8(%[b0],%[ldb2],2), %%ymm4 \n\t" // [x04,x14,x24,x34]
	            "vmovupd  0*8(%[b1],%[ldb2],2), %%ymm5 \n\t" // [x05,x15,x25,x35]
	            "\n\t"
	            "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm10\n\t" // [x00,x10,x01,x11]
	            "vperm2f128  $0x20, %%ymm3 , %%ymm2 , %%ymm11\n\t" // [x02,x12,x03,x13]
	            "vperm2f128  $0x20, %%ymm5 , %%ymm4 , %%ymm12\n\t" // [x04,x14,x05,x15]
	            "vperm2f128  $0x31, %%ymm1 , %%ymm0 , %%ymm13\n\t" // [x20,x30,x21,x31]
	            "vperm2f128  $0x31, %%ymm3 , %%ymm2 , %%ymm14\n\t" // [x22,x32,x23,x33]
	            "vperm2f128  $0x31, %%ymm5 , %%ymm4 , %%ymm15\n\t" // [x24,x34,x25,x35]
	            "\n\t"
	            "vmovupd  %%ymm10,   0*8(%[b2])\n\t"
	            "vmovupd  %%ymm11,   4*8(%[b2])\n\t"
	            "vmovupd  %%ymm12,   8*8(%[b2])\n\t"
	            "vmovupd  %%ymm13,  12*8(%[b2])\n\t"
	            "vmovupd  %%ymm14,  16*8(%[b2])\n\t"
	            "vmovupd  %%ymm15,  20*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $4*8 , %[b0]\n\t"
	            "addq  $4*8 , %[b1]\n\t"
	            "addq  $24*8, %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2),[ldb3]"r"(ldb3),[ldb5]"r"(ldb5));

	    }
	    if( K & 2 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd  0*8(%[b0]          ), %%xmm0 \n\t" // [x00,x10,---,---]
	            "vmovupd  0*8(%[b1]          ), %%xmm1 \n\t" // [x01,x11,---,---]
	            "vmovupd  0*8(%[b0],%[ldb2]  ), %%xmm2 \n\t" // [x02,x12,---,---]
	            "vmovupd  0*8(%[b1],%[ldb2]  ), %%xmm3 \n\t" // [x03,x13,---,---]
	            "vmovupd  0*8(%[b0],%[ldb2],2), %%xmm4 \n\t" // [x04,x14,---,---]
	            "vmovupd  0*8(%[b1],%[ldb2],2), %%xmm5 \n\t" // [x05,x15,---,---]
	            "\n\t"
	            "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm10\n\t" // [x00,x10,x01,x11]
	            "vperm2f128  $0x20, %%ymm3 , %%ymm2 , %%ymm11\n\t" // [x02,x12,x03,x13]
	            "vperm2f128  $0x20, %%ymm5 , %%ymm4 , %%ymm12\n\t" // [x04,x14,x05,x15]
	            "\n\t"
	            "vmovupd  %%ymm10,   0*8(%[b2])\n\t"
	            "vmovupd  %%ymm11,   4*8(%[b2])\n\t"
	            "vmovupd  %%ymm12,   8*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $2*8 , %[b0]\n\t"
	            "addq  $2*8 , %[b1]\n\t"
	            "addq  $12*8, %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2),[ldb3]"r"(ldb3),[ldb5]"r"(ldb5));

	    }
	    if( K & 1 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "movlpd  0*8(%[b0]          ), %%xmm0 \n\t" // [x00,  0,  0,  0]
	            "movhpd  0*8(%[b1]          ), %%xmm0 \n\t" // [x00,x01,  0,  0]
	            "movlpd  0*8(%[b0],%[ldb2]  ), %%xmm2 \n\t" // [x02,  0,  0,  0]
	            "movhpd  0*8(%[b1],%[ldb2]  ), %%xmm2 \n\t" // [x02,x03,  0,  0]
	            "movlpd  0*8(%[b0],%[ldb2],2), %%xmm4 \n\t" // [x04,  0,  0,  0]
	            "movhpd  0*8(%[b1],%[ldb2],2), %%xmm4 \n\t" // [x04,x05,  0,  0]
	            "\n\t"
	            "vmovupd  %%xmm0 ,   0*8(%[b2])\n\t"
	            "vmovupd  %%xmm2 ,   2*8(%[b2])\n\t"
	            "vmovupd  %%xmm4 ,   4*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $1*8 , %[b0]\n\t"
	            "addq  $1*8 , %[b1]\n\t"
	            "addq  $6*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2),[ldb3]"r"(ldb3),[ldb5]"r"(ldb5));

	    }
	    b0 = b0 - K + 6*ldb ;
	    b1 = b1 - K + 6*ldb ;

	  }
	}
	if( NR & 4 ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	            "vmovupd  0*8(%[b1]        ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	            "vmovupd  0*8(%[b0],%[ldb2]), %%ymm2 \n\t" // [x02,x12,x22,x32]
	            "vmovupd  0*8(%[b1],%[ldb2]), %%ymm3 \n\t" // [x03,x13,x23,x33]
	            "vmovupd  4*8(%[b0]        ), %%ymm4 \n\t" // [x40,x50,x60,x70]
	            "vmovupd  4*8(%[b1]        ), %%ymm5 \n\t" // [x41,x51,x61,x71]
	            "vmovupd  4*8(%[b0],%[ldb2]), %%ymm6 \n\t" // [x42,x52,x62,x72]
	            "vmovupd  4*8(%[b1],%[ldb2]), %%ymm7 \n\t" // [x43,x53,x63,x73]
	            "\n\t"
	            "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [x00,x10,x01,x11]
	            "vperm2f128  $0x20, %%ymm3 , %%ymm2 , %%ymm9 \n\t" // [x02,x12,x03,x13]
	            "vperm2f128  $0x31, %%ymm1 , %%ymm0 , %%ymm10\n\t" // [x20,x30,x21,x31]
	            "vperm2f128  $0x31, %%ymm3 , %%ymm2 , %%ymm11\n\t" // [x22,x32,x23,x33]
	            "vperm2f128  $0x20, %%ymm5 , %%ymm4 , %%ymm12\n\t" // [x40,x50,x41,x51]
	            "vperm2f128  $0x20, %%ymm7 , %%ymm6 , %%ymm13\n\t" // [x42,x52,x43,x53]
	            "vperm2f128  $0x31, %%ymm5 , %%ymm4 , %%ymm14\n\t" // [x60,x70,x61,x71]
	            "vperm2f128  $0x31, %%ymm7 , %%ymm6 , %%ymm15\n\t" // [x62,x72,x63,x73]
	            "\n\t"
	            "vmovapd  %%ymm8 ,   0*8(%[b2])\n\t"
	            "vmovapd  %%ymm9 ,   4*8(%[b2])\n\t"
	            "vmovapd  %%ymm10,   8*8(%[b2])\n\t"
	            "vmovapd  %%ymm11,  12*8(%[b2])\n\t"
	            "vmovapd  %%ymm12,  16*8(%[b2])\n\t"
	            "vmovapd  %%ymm13,  20*8(%[b2])\n\t"
	            "vmovapd  %%ymm14,  24*8(%[b2])\n\t"
	            "vmovapd  %%ymm15,  28*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $8*8 , %[b0]\n\t"
	            "addq  $8*8 , %[b1]\n\t"
	            "addq  $32*8, %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	      }
	    }
	    if( K & 4 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	            "vmovupd  0*8(%[b1]        ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	            "vmovupd  0*8(%[b0],%[ldb2]), %%ymm2 \n\t" // [x02,x12,x22,x32]
	            "vmovupd  0*8(%[b1],%[ldb2]), %%ymm3 \n\t" // [x03,x13,x23,x33]
	            "\n\t"
	            "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [x00,x10,x01,x11]
	            "vperm2f128  $0x20, %%ymm3 , %%ymm2 , %%ymm9 \n\t" // [x02,x12,x03,x13]
	            "vperm2f128  $0x31, %%ymm1 , %%ymm0 , %%ymm10\n\t" // [x20,x30,x21,x31]
	            "vperm2f128  $0x31, %%ymm3 , %%ymm2 , %%ymm11\n\t" // [x22,x32,x23,x33]
	            "\n\t"
	            "vmovapd  %%ymm8 ,   0*8(%[b2])\n\t"
	            "vmovapd  %%ymm9 ,   4*8(%[b2])\n\t"
	            "vmovapd  %%ymm10,   8*8(%[b2])\n\t"
	            "vmovapd  %%ymm11,  12*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $4*8 , %[b0]\n\t"
	            "addq  $4*8 , %[b1]\n\t"
	            "addq  $16*8, %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	    }
	    if( K & 2 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd  0*8(%[b0]        ), %%xmm0 \n\t" // [x00,x10,---,---]
	            "vmovupd  0*8(%[b1]        ), %%xmm1 \n\t" // [x01,x11,---,---]
	            "vmovupd  0*8(%[b0],%[ldb2]), %%xmm2 \n\t" // [x02,x12,---,---]
	            "vmovupd  0*8(%[b1],%[ldb2]), %%xmm3 \n\t" // [x03,x13,---,---]
	            "\n\t"
	            "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [x00,x10,x01,x11]
	            "vperm2f128  $0x20, %%ymm3 , %%ymm2 , %%ymm9 \n\t" // [x02,x12,x03,x13]
	            "\n\t"
	            "vmovapd  %%ymm8 ,   0*8(%[b2])\n\t"
	            "vmovapd  %%ymm9 ,   4*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $2*8 , %[b0]\n\t"
	            "addq  $2*8 , %[b1]\n\t"
	            "addq  $8*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	    }
	    if( K & 1 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "movlpd  0*8(%[b0]        ), %%xmm0 \n\t" // [x00,---,---,---]
	            "movhpd  0*8(%[b1]        ), %%xmm0 \n\t" // [x00,x01,---,---]
	            "movlpd  0*8(%[b0],%[ldb2]), %%xmm2 \n\t" // [x02,---,---,---]
	            "movhpd  0*8(%[b1],%[ldb2]), %%xmm2 \n\t" // [x02,x03,---,---]
	            "\n\t"
	            "vperm2f128  $0x20, %%ymm2 , %%ymm0 , %%ymm8 \n\t" // [x00,x01,x02,x03]
	            "\n\t"
	            "vmovapd  %%ymm8 ,   0*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $1*8 , %[b0]\n\t"
	            "addq  $1*8 , %[b1]\n\t"
	            "addq  $4*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	    }
	    b0 = b0 - K + 4*ldb ;
	    b1 = b1 - K + 4*ldb ;

	}
	if( NR & 2 ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	            "vmovupd  0*8(%[b1]        ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	            "vmovupd  4*8(%[b0]        ), %%ymm4 \n\t" // [x40,x50,x60,x70]
	            "vmovupd  4*8(%[b1]        ), %%ymm5 \n\t" // [x41,x51,x61,x71]
	            "\n\t"
	            "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [x00,x10,x01,x11]
	            "vperm2f128  $0x31, %%ymm1 , %%ymm0 , %%ymm10\n\t" // [x20,x30,x21,x31]
	            "vperm2f128  $0x20, %%ymm5 , %%ymm4 , %%ymm12\n\t" // [x40,x50,x41,x51]
	            "vperm2f128  $0x31, %%ymm5 , %%ymm4 , %%ymm14\n\t" // [x60,x70,x61,x71]
	            "\n\t"
	            "vmovupd  %%ymm8 ,   0*8(%[b2])\n\t"
	            "vmovupd  %%ymm10,   4*8(%[b2])\n\t"
	            "vmovupd  %%ymm12,   8*8(%[b2])\n\t"
	            "vmovupd  %%ymm14,  12*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $8*8 , %[b0]\n\t"
	            "addq  $8*8 , %[b1]\n\t"
	            "addq  $16*8, %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	      }
	    }
	    if( K & 4 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	            "vmovupd  0*8(%[b1]        ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	            "\n\t"
	            "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [x00,x10,x01,x11]
	            "vperm2f128  $0x31, %%ymm1 , %%ymm0 , %%ymm10\n\t" // [x20,x30,x21,x31]
	            "\n\t"
	            "vmovupd  %%ymm8 ,   0*8(%[b2])\n\t"
	            "vmovupd  %%ymm10,   4*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $4*8 , %[b0]\n\t"
	            "addq  $4*8 , %[b1]\n\t"
	            "addq  $8*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	    }
	    if( K & 2 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd  0*8(%[b0]        ), %%xmm0 \n\t" // [x00,x10,---,---]
	            "vmovupd  0*8(%[b1]        ), %%xmm1 \n\t" // [x01,x11,---,---]
	            "\n\t"
	            "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [x00,x10,x01,x11]
	            "\n\t"
	            "vmovupd  %%ymm8 ,   0*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $2*8 , %[b0]\n\t"
	            "addq  $2*8 , %[b1]\n\t"
	            "addq  $4*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	    }
	    if( K & 1 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "movlpd  0*8(%[b0]        ), %%xmm0 \n\t" // [x00,---,---,---]
	            "movhpd  0*8(%[b1]        ), %%xmm0 \n\t" // [x00,x01,---,---]
	            "\n\t"
	            "movapd  %%xmm0 ,   0*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $1*8 , %[b0]\n\t"
	            "addq  $1*8 , %[b1]\n\t"
	            "addq  $2*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	    }
	    b0 = b0 - K + 2*ldb ;
	    b1 = b1 - K + 2*ldb ;


	}
	if( NR & 1 ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	            "vmovupd  4*8(%[b0]        ), %%ymm4 \n\t" // [x40,x50,x60,x70]
	            "\n\t"
	            "vmovupd  %%ymm0 ,   0*8(%[b2])\n\t"
	            "vmovupd  %%ymm4 ,   4*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $8*8 , %[b0]\n\t"
	            "addq  $8*8 , %[b1]\n\t"
	            "addq  $8*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	      }
	    }
	    if( K & 4 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	            "\n\t"
	            "vmovupd  %%ymm0 ,   0*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $4*8 , %[b0]\n\t"
	            "addq  $4*8 , %[b1]\n\t"
	            "addq  $4*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	    }
	    if( K & 2 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd  0*8(%[b0]        ), %%xmm0 \n\t" // [x00,x10,---,---]
	            "\n\t"
	            "movupd  %%xmm0 ,   0*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $2*8 , %[b0]\n\t"
	            "addq  $2*8 , %[b1]\n\t"
	            "addq  $2*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));


	    }
	    if( K & 1 ){

	          __asm__ __volatile__ (
	            "\n\t"
	            "movlpd  0*8(%[b0]        ), %%xmm0 \n\t" // [x00,---,---,---]
	            "\n\t"
	            "movlpd  %%xmm0 ,   0*8(%[b2])\n\t"
	            "\n\t"
	            "addq  $1*8 , %[b0]\n\t"
	            "addq  $1*8 , %[b1]\n\t"
	            "addq  $1*8 , %[b2]\n\t"
	            "\n\t"
	          :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	          :[ldb2]"r"(ldb2));

	    }
	    b0 = b0 - K + 1*ldb ;
	    b1 = b1 - K + 1*ldb ;

	}


}


