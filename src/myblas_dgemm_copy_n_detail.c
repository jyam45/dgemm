#include "myblas_internal.h"
#include <stdlib.h>

void myblas_dgemm_copy_n_detail(size_t K, size_t N, const double* B, size_t k, size_t j,  size_t ldb, double* B2 ){

	double x00,x01,x02,x03;
	double x10,x11,x12,x13;
	double x20,x21,x22,x23;
	double x30,x31,x32,x33;

	B = B + k + ldb*j; // start point

	const double* b0 = B;
	const double* b1 = B + ldb;
	size_t        ldb2 = ldb * 2 * sizeof(double);
	size_t        ldb3 = ldb * 3 * sizeof(double);

	size_t aligned = ((((uint64_t)B)&0x1f)?0:1)&((ldb&0x02)?0:1);

	if( aligned ){

	  if( N >> 2 ){
	    size_t n4 = ( N >> 2 );
	    while( n4-- ){

	      //__asm__ __volatile__ (
	      //  "\n\t"
	      //  "prefetcht1  0*8(%[b0],%[ldb2],2)\n\t"
	      //  "prefetcht1  0*8(%[b1],%[ldb2],2)\n\t"
	      //  "prefetcht1  0*8(%[b0],%[ldb3],2)\n\t"
	      //  "prefetcht1  0*8(%[b1],%[ldb3],2)\n\t"
	      //  "\n\t"
	      //:[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	      //:[ldb2]"r"(ldb2),[ldb3]"r"(ldb3));

	      if( K >> 3 ){
	        size_t k8 = ( K >> 3 );
	        k8--;

	        __asm__ __volatile__ (
	          "\n\t"
	          "vmovapd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	          "vmovapd  0*8(%[b1]        ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	          "vmovapd  0*8(%[b0],%[ldb2]), %%ymm2 \n\t" // [x02,x12,x22,x32]
	          "vmovapd  0*8(%[b1],%[ldb2]), %%ymm3 \n\t" // [x03,x13,x23,x33]
	          "\n\t"
	          "vshufpd  $0x00, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [x00,x01,x20,x21]
	          "vshufpd  $0x0f, %%ymm1 , %%ymm0 , %%ymm9 \n\t" // [x10,x11,x30,x31]
	          "vshufpd  $0x00, %%ymm3 , %%ymm2 , %%ymm10\n\t" // [x02,x03,x22,x23]
	          "vshufpd  $0x0f, %%ymm3 , %%ymm2 , %%ymm11\n\t" // [x12,x13,x32,x33]
	          "\n\t"
	        :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	        :[ldb2]"r"(ldb2));


	        while( k8-- ){
	            //x00 = *(B+0+0*ldb); x10 = *(B+1+0*ldb); x20 = *(B+2+0*ldb); x30 = *(B+3+0*ldb);
	            //x01 = *(B+0+1*ldb); x11 = *(B+1+1*ldb); x21 = *(B+2+1*ldb); x31 = *(B+3+1*ldb);
	            //x02 = *(B+0+2*ldb); x12 = *(B+1+2*ldb); x22 = *(B+2+2*ldb); x32 = *(B+3+2*ldb);
	            //x03 = *(B+0+3*ldb); x13 = *(B+1+3*ldb); x23 = *(B+2+3*ldb); x33 = *(B+3+3*ldb);
	            //*(B2+0*4+0) = x00; *(B2+1*4+0) = x10; *(B2+2*4+0) = x20; *(B2+3*4+0) = x30;
	            //*(B2+0*4+1) = x01; *(B2+1*4+1) = x11; *(B2+2*4+1) = x21; *(B2+3*4+1) = x31;
	            //*(B2+0*4+2) = x02; *(B2+1*4+2) = x12; *(B2+2*4+2) = x22; *(B2+3*4+2) = x32;
	            //*(B2+0*4+3) = x03; *(B2+1*4+3) = x13; *(B2+2*4+3) = x23; *(B2+3*4+3) = x33;
	            //x00 = *(B+4+0*ldb); x10 = *(B+5+0*ldb); x20 = *(B+6+0*ldb); x30 = *(B+7+0*ldb);
	            //x01 = *(B+4+1*ldb); x11 = *(B+5+1*ldb); x21 = *(B+6+1*ldb); x31 = *(B+7+1*ldb);
	            //x02 = *(B+4+2*ldb); x12 = *(B+5+2*ldb); x22 = *(B+6+2*ldb); x32 = *(B+7+2*ldb);
	            //x03 = *(B+4+3*ldb); x13 = *(B+5+3*ldb); x23 = *(B+6+3*ldb); x33 = *(B+7+3*ldb);
	            //*(B2+4*4+0) = x00; *(B2+5*4+0) = x10; *(B2+6*4+0) = x20; *(B2+7*4+0) = x30;
	            //*(B2+4*4+1) = x01; *(B2+5*4+1) = x11; *(B2+6*4+1) = x21; *(B2+7*4+1) = x31;
	            //*(B2+4*4+2) = x02; *(B2+5*4+2) = x12; *(B2+6*4+2) = x22; *(B2+7*4+2) = x32;
	            //*(B2+4*4+3) = x03; *(B2+5*4+3) = x13; *(B2+6*4+3) = x23; *(B2+7*4+3) = x33;
	            //B+=8;
	            //B2+=32;

	            __asm__ __volatile__ (
	              "\n\t"
	              "prefetcht0  0*8(%[b0],%[ldb2],2)\n\t"
	              "prefetcht0  0*8(%[b1],%[ldb2],2)\n\t"
	              "prefetcht0  0*8(%[b0],%[ldb3],2)\n\t"
	              "prefetcht0  0*8(%[b1],%[ldb3],2)\n\t"
	              "\n\t"
	              "vmovapd  4*8(%[b0]        ), %%ymm4 \n\t" // [x00,x10,x20,x30]
	              "vmovapd  4*8(%[b1]        ), %%ymm5 \n\t" // [x01,x11,x21,x31]
	              "vperm2f128  $0x20, %%ymm10, %%ymm8 , %%ymm12\n\t" // [x00,x01,x02,x03]
	              "vperm2f128  $0x20, %%ymm11, %%ymm9 , %%ymm13\n\t" // [x10,x11,x12,x13]
	              "\n\t"
	              "vmovapd  4*8(%[b0],%[ldb2]), %%ymm6 \n\t" // [x02,x12,x22,x32]
	              "vmovapd  4*8(%[b1],%[ldb2]), %%ymm7 \n\t" // [x03,x13,x23,x33]
	              "vperm2f128  $0x31, %%ymm10, %%ymm8 , %%ymm14\n\t" // [x20,x21,x22,x23]
	              "vperm2f128  $0x31, %%ymm11, %%ymm9 , %%ymm15\n\t" // [x30,x31,x32,x33]
	              "\n\t"
	              "vshufpd  $0x00, %%ymm5 , %%ymm4 , %%ymm8 \n\t" // [x00,x01,x20,x21]
	              "vshufpd  $0x0f, %%ymm5 , %%ymm4 , %%ymm9 \n\t" // [x10,x11,x30,x31]
	              "vmovapd  %%ymm12,   0*8(%[b2])\n\t"
	              "vmovapd  %%ymm13,   4*8(%[b2])\n\t"
	              "\n\t"
	              "vshufpd  $0x00, %%ymm7 , %%ymm6 , %%ymm10\n\t" // [x02,x03,x22,x23]
	              "vshufpd  $0x0f, %%ymm7 , %%ymm6 , %%ymm11\n\t" // [x12,x13,x32,x33]
	              "vmovapd  %%ymm14,   8*8(%[b2])\n\t"
	              "vmovapd  %%ymm15,  12*8(%[b2])\n\t"
	              "\n\t"
	              "vmovapd  8*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	              "vmovapd  8*8(%[b1]        ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	              "vperm2f128  $0x20, %%ymm10, %%ymm8 , %%ymm12\n\t" // [x00,x01,x02,x03]
	              "vperm2f128  $0x20, %%ymm11, %%ymm9 , %%ymm13\n\t" // [x10,x11,x12,x13]
	              "\n\t"
	              "vmovapd  8*8(%[b0],%[ldb2]), %%ymm2 \n\t" // [x02,x12,x22,x32]
	              "vmovapd  8*8(%[b1],%[ldb2]), %%ymm3 \n\t" // [x03,x13,x23,x33]
	              "vperm2f128  $0x31, %%ymm10, %%ymm8 , %%ymm14\n\t" // [x20,x21,x22,x23]
	              "vperm2f128  $0x31, %%ymm11, %%ymm9 , %%ymm15\n\t" // [x30,x31,x32,x33]
	              "\n\t"
	              "vshufpd  $0x00, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [x00,x01,x20,x21]
	              "vshufpd  $0x0f, %%ymm1 , %%ymm0 , %%ymm9 \n\t" // [x10,x11,x30,x31]
	              "vmovapd  %%ymm12,  16*8(%[b2])\n\t"
	              "vmovapd  %%ymm13,  20*8(%[b2])\n\t"
	              "\n\t"
	              "vshufpd  $0x00, %%ymm3 , %%ymm2 , %%ymm10\n\t" // [x02,x03,x22,x23]
	              "vshufpd  $0x0f, %%ymm3 , %%ymm2 , %%ymm11\n\t" // [x12,x13,x32,x33]
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

	        __asm__ __volatile__ (
	          "\n\t"
	          "vmovapd  4*8(%[b0]        ), %%ymm4 \n\t" // [x00,x10,x20,x30]
	          "vmovapd  4*8(%[b1]        ), %%ymm5 \n\t" // [x01,x11,x21,x31]
	          "vperm2f128  $0x20, %%ymm10, %%ymm8 , %%ymm12\n\t" // [x00,x01,x02,x03]
	          "vperm2f128  $0x20, %%ymm11, %%ymm9 , %%ymm13\n\t" // [x10,x11,x12,x13]
	          "\n\t"
	          "vmovapd  4*8(%[b0],%[ldb2]), %%ymm6 \n\t" // [x02,x12,x22,x32]
	          "vmovapd  4*8(%[b1],%[ldb2]), %%ymm7 \n\t" // [x03,x13,x23,x33]
	          "vperm2f128  $0x31, %%ymm10, %%ymm8 , %%ymm14\n\t" // [x20,x21,x22,x23]
	          "vperm2f128  $0x31, %%ymm11, %%ymm9 , %%ymm15\n\t" // [x30,x31,x32,x33]
	          "\n\t"
	          "vshufpd  $0x00, %%ymm5 , %%ymm4 , %%ymm8 \n\t" // [x00,x01,x20,x21]
	          "vshufpd  $0x0f, %%ymm5 , %%ymm4 , %%ymm9 \n\t" // [x10,x11,x30,x31]
	          "vmovapd  %%ymm12,   0*8(%[b2])\n\t"
	          "vmovapd  %%ymm13,   4*8(%[b2])\n\t"
	          "\n\t"
	          "vshufpd  $0x00, %%ymm7 , %%ymm6 , %%ymm10\n\t" // [x02,x03,x22,x23]
	          "vshufpd  $0x0f, %%ymm7 , %%ymm6 , %%ymm11\n\t" // [x12,x13,x32,x33]
	          "vmovapd  %%ymm14,   8*8(%[b2])\n\t"
	          "vmovapd  %%ymm15,  12*8(%[b2])\n\t"
	          "\n\t"
	          "vperm2f128  $0x20, %%ymm10, %%ymm8 , %%ymm12\n\t" // [x00,x01,x02,x03]
	          "vperm2f128  $0x31, %%ymm10, %%ymm8 , %%ymm14\n\t" // [x20,x21,x22,x23]
	          "vperm2f128  $0x20, %%ymm11, %%ymm9 , %%ymm13\n\t" // [x10,x11,x12,x13]
	          "vperm2f128  $0x31, %%ymm11, %%ymm9 , %%ymm15\n\t" // [x30,x31,x32,x33]
	          "\n\t"
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
	      if( K & 4  ){
	      //if( K >> 2 ){
	      //  size_t k4 = ( K >> 2 ); // unrolling with 8 elements
	      //  while( k4-- ){
	            //x00 = *(B+0+0*ldb); x10 = *(B+1+0*ldb); x20 = *(B+2+0*ldb); x30 = *(B+3+0*ldb);
	            //x01 = *(B+0+1*ldb); x11 = *(B+1+1*ldb); x21 = *(B+2+1*ldb); x31 = *(B+3+1*ldb);
	            //x02 = *(B+0+2*ldb); x12 = *(B+1+2*ldb); x22 = *(B+2+2*ldb); x32 = *(B+3+2*ldb);
	            //x03 = *(B+0+3*ldb); x13 = *(B+1+3*ldb); x23 = *(B+2+3*ldb); x33 = *(B+3+3*ldb);
	            //*(B2+0*4+0) = x00; *(B2+1*4+0) = x10; *(B2+2*4+0) = x20; *(B2+3*4+0) = x30;
	            //*(B2+0*4+1) = x01; *(B2+1*4+1) = x11; *(B2+2*4+1) = x21; *(B2+3*4+1) = x31;
	            //*(B2+0*4+2) = x02; *(B2+1*4+2) = x12; *(B2+2*4+2) = x22; *(B2+3*4+2) = x32;
	            //*(B2+0*4+3) = x03; *(B2+1*4+3) = x13; *(B2+2*4+3) = x23; *(B2+3*4+3) = x33;
	            //B+=4;
	            //B2+=16;

	            __asm__ __volatile__ (
	              "\n\t"
	              "vmovapd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	              "vmovapd  0*8(%[b1]        ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	              "vmovapd  0*8(%[b0],%[ldb2]), %%ymm2 \n\t" // [x02,x12,x22,x32]
	              "vmovapd  0*8(%[b1],%[ldb2]), %%ymm3 \n\t" // [x03,x13,x23,x33]
	              "\n\t"
	              "vshufpd  $0x00, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [x00,x01,x20,x21]
	              "vshufpd  $0x0f, %%ymm1 , %%ymm0 , %%ymm9 \n\t" // [x10,x11,x30,x31]
	              "vshufpd  $0x00, %%ymm3 , %%ymm2 , %%ymm10\n\t" // [x02,x03,x22,x23]
	              "vshufpd  $0x0f, %%ymm3 , %%ymm2 , %%ymm11\n\t" // [x12,x13,x32,x33]
	              "\n\t"
	              "vperm2f128  $0x20, %%ymm10, %%ymm8 , %%ymm12\n\t" // [x00,x01,x02,x03]
	              "vperm2f128  $0x31, %%ymm10, %%ymm8 , %%ymm14\n\t" // [x20,x21,x22,x23]
	              "vperm2f128  $0x20, %%ymm11, %%ymm9 , %%ymm13\n\t" // [x10,x11,x12,x13]
	              "vperm2f128  $0x31, %%ymm11, %%ymm9 , %%ymm15\n\t" // [x30,x31,x32,x33]
	              "\n\t"
	              "vmovapd  %%ymm12,   0*8(%[b2])\n\t"
	              "vmovapd  %%ymm13,   4*8(%[b2])\n\t"
	              "vmovapd  %%ymm14,   8*8(%[b2])\n\t"
	              "vmovapd  %%ymm15,  12*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $4*8 , %[b0]\n\t"
	              "addq  $4*8 , %[b1]\n\t"
	              "addq  $16*8, %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	            :[ldb2]"r"(ldb2));

	      //  }
	      }
	      if( K & 2 ){
	            //x00 = *(B+0+0*ldb); x10 = *(B+1+0*ldb);
	            //x01 = *(B+0+1*ldb); x11 = *(B+1+1*ldb);
	            //x02 = *(B+0+2*ldb); x12 = *(B+1+2*ldb);
	            //x03 = *(B+0+3*ldb); x13 = *(B+1+3*ldb);
	            //*(B2+0*4+0) = x00; *(B2+1*4+0) = x10;
	            //*(B2+0*4+1) = x01; *(B2+1*4+1) = x11;
	            //*(B2+0*4+2) = x02; *(B2+1*4+2) = x12;
	            //*(B2+0*4+3) = x03; *(B2+1*4+3) = x13;
	            //B+=2;
	            //B2+=8;

	            __asm__ __volatile__ (
	              "\n\t"
	              "vmovapd  0*8(%[b0]        ), %%xmm0 \n\t" // [x00,x10]
	              "vmovapd  0*8(%[b1]        ), %%xmm1 \n\t" // [x01,x11]
	              "vmovapd  0*8(%[b0],%[ldb2]), %%xmm2 \n\t" // [x02,x12]
	              "vmovapd  0*8(%[b1],%[ldb2]), %%xmm3 \n\t" // [x03,x13]
	              "\n\t"
	              "vshufpd  $0x00, %%xmm1 , %%xmm0 , %%xmm12\n\t" // [x00,x01]
	              "vshufpd  $0x0f, %%xmm1 , %%xmm0 , %%xmm14\n\t" // [x10,x11]
	              "vshufpd  $0x00, %%xmm3 , %%xmm2 , %%xmm13\n\t" // [x02,x03]
	              "vshufpd  $0x0f, %%xmm3 , %%xmm2 , %%xmm15\n\t" // [x12,x13]
	              "\n\t"
	              "vmovapd  %%xmm12,   0*8(%[b2])\n\t"
	              "vmovapd  %%xmm13,   2*8(%[b2])\n\t"
	              "vmovapd  %%xmm14,   4*8(%[b2])\n\t"
	              "vmovapd  %%xmm15,   6*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $2*8 , %[b0]\n\t"
	              "addq  $2*8 , %[b1]\n\t"
	              "addq  $8*8 , %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	            :[ldb2]"r"(ldb2));

	      }
	      if( K & 1 ){
	            //x00 = *(B+0+0*ldb);
	            //x01 = *(B+0+1*ldb);
	            //x02 = *(B+0+2*ldb);
	            //x03 = *(B+0+3*ldb);
	            //*(B2+0*4+0) = x00;
	            //*(B2+0*4+1) = x01;
	            //*(B2+0*4+2) = x02;
	            //*(B2+0*4+3) = x03;
	            //B+=1;
	            //B2+=4;

	            __asm__ __volatile__ (
	              "\n\t"
	              "movsd  0*8(%[b0]        ), %%xmm12\n\t" // [x00]
	              "movsd  0*8(%[b1]        ), %%xmm13\n\t" // [x01]
	              "movsd  0*8(%[b0],%[ldb2]), %%xmm14\n\t" // [x02]
	              "movsd  0*8(%[b1],%[ldb2]), %%xmm15\n\t" // [x03]
	              "\n\t"
	              "movlpd  %%xmm12,   0*8(%[b2])\n\t"
	              "movlpd  %%xmm13,   1*8(%[b2])\n\t"
	              "movlpd  %%xmm14,   2*8(%[b2])\n\t"
	              "movlpd  %%xmm15,   3*8(%[b2])\n\t"
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
	        k8--;

	        __asm__ __volatile__ (
	          "\n\t"
	          "movapd  0*8(%[b0]), %%xmm0 \n\t" // [x00,x10]
	          "movapd  2*8(%[b0]), %%xmm1 \n\t" // [x20,x30]
	          "movapd  0*8(%[b1]), %%xmm2 \n\t" // [x01,x11]
	          "movapd  2*8(%[b1]), %%xmm3 \n\t" // [x21,x31]
	          "\n\t"
	        :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	        :);


	        while( k8-- ){
	            //x00 = *(B+0+0*ldb); x10 = *(B+1+0*ldb); x20 = *(B+2+0*ldb); x30 = *(B+3+0*ldb);
	            //x01 = *(B+0+1*ldb); x11 = *(B+1+1*ldb); x21 = *(B+2+1*ldb); x31 = *(B+3+1*ldb);
	            //*(B2+0*2+0) = x00; *(B2+1*2+0) = x10; *(B2+2*2+0) = x20; *(B2+3*2+0) = x30;
	            //*(B2+0*2+1) = x01; *(B2+1*2+1) = x11; *(B2+2*2+1) = x21; *(B2+3*2+1) = x31;
	            //x00 = *(B+4+0*ldb); x10 = *(B+5+0*ldb); x20 = *(B+6+0*ldb); x30 = *(B+7+0*ldb);
	            //x01 = *(B+4+1*ldb); x11 = *(B+5+1*ldb); x21 = *(B+6+1*ldb); x31 = *(B+7+1*ldb);
	            //*(B2+4*2+0) = x00; *(B2+5*2+0) = x10; *(B2+6*2+0) = x20; *(B2+7*2+0) = x30;
	            //*(B2+4*2+1) = x01; *(B2+5*2+1) = x11; *(B2+6*2+1) = x21; *(B2+7*2+1) = x31;
	            //B+=8;
	            //B2+=16;

	            __asm__ __volatile__ (
	              "\n\t"
	              "vshufpd  $0x00, %%xmm2 , %%xmm0 , %%xmm8 \n\t" // [x00,x01]
	              "vshufpd  $0x03, %%xmm2 , %%xmm0 , %%xmm9 \n\t" // [x10,x11]
	              "movapd  4*8(%[b0]), %%xmm4 \n\t" // [x00,x10]
	              "movapd  4*8(%[b1]), %%xmm6 \n\t" // [x01,x11]
	              "movapd  %%xmm8 ,   0*8(%[b2])\n\t"
	              "movapd  %%xmm9 ,   2*8(%[b2])\n\t"
	              "\n\t"
	              "vshufpd  $0x00, %%xmm3 , %%xmm1 , %%xmm10\n\t" // [x20,x21]
	              "vshufpd  $0x03, %%xmm3 , %%xmm1 , %%xmm11\n\t" // [x30,x31]
	              "movapd  6*8(%[b0]), %%xmm5 \n\t" // [x20,x30]
	              "movapd  6*8(%[b1]), %%xmm7 \n\t" // [x21,x31]
	              "movapd  %%xmm10,   4*8(%[b2])\n\t"
	              "movapd  %%xmm11,   6*8(%[b2])\n\t"
	              "\n\t"
	              "vshufpd  $0x00, %%xmm6 , %%xmm4 , %%xmm12\n\t" // [x00,x01]
	              "vshufpd  $0x03, %%xmm6 , %%xmm4 , %%xmm13\n\t" // [x10,x11]
	              "movapd  8*8(%[b0]), %%xmm0 \n\t" // [x00,x10]
	              "movapd  8*8(%[b1]), %%xmm2 \n\t" // [x01,x11]
	              "movapd  %%xmm12,   8*8(%[b2])\n\t"
	              "movapd  %%xmm13,  10*8(%[b2])\n\t"
	              "\n\t"
	              "vshufpd  $0x00, %%xmm7 , %%xmm5 , %%xmm14\n\t" // [x20,x21]
	              "vshufpd  $0x03, %%xmm7 , %%xmm5 , %%xmm15\n\t" // [x30,x31]
	              "movapd 10*8(%[b0]), %%xmm1 \n\t" // [x20,x30]
	              "movapd 10*8(%[b1]), %%xmm3 \n\t" // [x21,x31]
	              "movapd  %%xmm14,  12*8(%[b2])\n\t"
	              "movapd  %%xmm15,  14*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $8*8 , %[b0]\n\t"
	              "addq  $8*8 , %[b1]\n\t"
	              "addq  $16*8, %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	            :);

	        }

	        __asm__ __volatile__ (
	          "\n\t"
	          "movapd  4*8(%[b0]), %%xmm4 \n\t" // [x00,x10]
	          "movapd  6*8(%[b0]), %%xmm5 \n\t" // [x20,x30]
	          "vshufpd  $0x00, %%xmm2 , %%xmm0 , %%xmm8 \n\t" // [x00,x01]
	          "vshufpd  $0x03, %%xmm2 , %%xmm0 , %%xmm9 \n\t" // [x10,x11]
	          "\n\t"
	          "movapd  4*8(%[b1]), %%xmm6 \n\t" // [x01,x11]
	          "movapd  6*8(%[b1]), %%xmm7 \n\t" // [x21,x31]
	          "vshufpd  $0x00, %%xmm3 , %%xmm1 , %%xmm10\n\t" // [x20,x21]
	          "vshufpd  $0x03, %%xmm3 , %%xmm1 , %%xmm11\n\t" // [x30,x31]
	          "\n\t"
	          "movapd  %%xmm8 ,   0*8(%[b2])\n\t"
	          "movapd  %%xmm9 ,   2*8(%[b2])\n\t"
	          "vshufpd  $0x00, %%xmm6 , %%xmm4 , %%xmm12\n\t" // [x00,x01]
	          "vshufpd  $0x03, %%xmm6 , %%xmm4 , %%xmm13\n\t" // [x10,x11]
	          "\n\t"
	          "movapd  %%xmm10,   4*8(%[b2])\n\t"
	          "movapd  %%xmm11,   6*8(%[b2])\n\t"
	          "vshufpd  $0x00, %%xmm7 , %%xmm5 , %%xmm14\n\t" // [x20,x21]
	          "vshufpd  $0x03, %%xmm7 , %%xmm5 , %%xmm15\n\t" // [x30,x31]
	          "\n\t"
	          "movapd  %%xmm12,   8*8(%[b2])\n\t"
	          "movapd  %%xmm13,  10*8(%[b2])\n\t"
	          "movapd  %%xmm14,  12*8(%[b2])\n\t"
	          "movapd  %%xmm15,  14*8(%[b2])\n\t"
	          "\n\t"
	          "addq  $8*8 , %[b0]\n\t"
	          "addq  $8*8 , %[b1]\n\t"
	          "addq  $16*8, %[b2]\n\t"
	          "\n\t"
	        :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	        :);
	      }
	      if( K & 4  ){
	      //if( K >> 2 ){
	      //  size_t k4 = ( K >> 2 ); // unrolling with 8 elements
	      //  while( k4-- ){
	            //x00 = *(B+0+0*ldb); x10 = *(B+1+0*ldb); x20 = *(B+2+0*ldb); x30 = *(B+3+0*ldb);
	            //x01 = *(B+0+1*ldb); x11 = *(B+1+1*ldb); x21 = *(B+2+1*ldb); x31 = *(B+3+1*ldb);
	            //*(B2+0*2+0) = x00; *(B2+1*2+0) = x10; *(B2+2*2+0) = x20; *(B2+3*2+0) = x30;
	            //*(B2+0*2+1) = x01; *(B2+1*2+1) = x11; *(B2+2*2+1) = x21; *(B2+3*2+1) = x31;
	            //B+=4;
	            //B2+=8;

	            __asm__ __volatile__ (
	              "\n\t"
	              "movapd  0*8(%[b0]), %%xmm0 \n\t" // [x00,x10]
	              "movapd  2*8(%[b0]), %%xmm1 \n\t" // [x20,x30]
	              "movapd  0*8(%[b1]), %%xmm2 \n\t" // [x01,x11]
	              "movapd  2*8(%[b1]), %%xmm3 \n\t" // [x21,x31]
	              "\n\t"
	              "vshufpd  $0x00, %%xmm2 , %%xmm0 , %%xmm8 \n\t" // [x00,x01]
	              "vshufpd  $0x03, %%xmm2 , %%xmm0 , %%xmm9 \n\t" // [x10,x11]
	              "vshufpd  $0x00, %%xmm3 , %%xmm1 , %%xmm10\n\t" // [x20,x21]
	              "vshufpd  $0x03, %%xmm3 , %%xmm1 , %%xmm11\n\t" // [x30,x31]
	              "\n\t"
	              "movapd  %%xmm8 ,   0*8(%[b2])\n\t"
	              "movapd  %%xmm9 ,   2*8(%[b2])\n\t"
	              "movapd  %%xmm10,   4*8(%[b2])\n\t"
	              "movapd  %%xmm11,   6*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $4*8 , %[b0]\n\t"
	              "addq  $4*8 , %[b1]\n\t"
	              "addq  $8*8 , %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	            :);

	      //  }
	      }
	      if( K & 2 ){
	            //x00 = *(B+0+0*ldb); x10 = *(B+1+0*ldb);
	            //x01 = *(B+0+1*ldb); x11 = *(B+1+1*ldb);
	            //*(B2+0*2+0) = x00; *(B2+1*2+0) = x10;
	            //*(B2+0*2+1) = x01; *(B2+1*2+1) = x11;
	            //B+=2;
	            //B2+=4;

	            __asm__ __volatile__ (
	              "\n\t"
	              "movapd  0*8(%[b0]), %%xmm0 \n\t" // [x00,x10]
	              "movapd  0*8(%[b1]), %%xmm2 \n\t" // [x01,x11]
	              "\n\t"
	              "vshufpd  $0x00, %%xmm2 , %%xmm0 , %%xmm8 \n\t" // [x00,x01]
	              "vshufpd  $0x03, %%xmm2 , %%xmm0 , %%xmm9 \n\t" // [x10,x11]
	              "\n\t"
	              "movapd  %%xmm8 ,   0*8(%[b2])\n\t"
	              "movapd  %%xmm9 ,   2*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $2*8 , %[b0]\n\t"
	              "addq  $2*8 , %[b1]\n\t"
	              "addq  $4*8 , %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	            :);

	      }
	      if( K & 1 ){
	            //x00 = *(B+0+0*ldb);
	            //x01 = *(B+0+1*ldb);
	            //*(B2+0*2+0) = x00;
	            //*(B2+0*2+1) = x01;
	            //B+=1;
	            //B2+=2;

	            __asm__ __volatile__ (
	              "\n\t"
	              "movsd   0*8(%[b0]), %%xmm0 \n\t" // [x00,   ]
	              "movhpd  0*8(%[b1]), %%xmm0 \n\t" // [x00,x01]
	              "\n\t"
	              "movapd  %%xmm0 ,   0*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $1*8 , %[b0]\n\t"
	              "addq  $1*8 , %[b1]\n\t"
	              "addq  $2*8 , %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	            :);

	      }
	      b0 = b0 - K + 2*ldb ;
	      b1 = b1 - K + 2*ldb ;

	  }
	  if( N & 1 ){

	      if( K >> 3 ){
	        size_t k8 = ( K >> 3 );
	        while( k8-- ){
	            //x00 = *(B+0+0*ldb); x10 = *(B+1+0*ldb); x20 = *(B+2+0*ldb); x30 = *(B+3+0*ldb);
	            //*(B2+0*1+0) = x00; *(B2+1*1+0) = x10; *(B2+2*1+0) = x20; *(B2+3*1+0) = x30;
	            //x00 = *(B+4+0*ldb); x10 = *(B+5+0*ldb); x20 = *(B+6+0*ldb); x30 = *(B+7+0*ldb);
	            //*(B2+4*1+0) = x00; *(B2+5*1+0) = x10; *(B2+6*1+0) = x20; *(B2+7*1+0) = x30;
	            //B+=8;
	            //B2+=8;

	            __asm__ __volatile__ (
	              "\n\t"
	              "vmovapd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	              "vmovapd  4*8(%[b0]        ), %%ymm4 \n\t" // [x00,x10,x20,x30]
	              "\n\t"
	              "vmovapd  %%ymm0 ,   0*8(%[b2])\n\t"
	              "vmovapd  %%ymm4 ,   4*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $8*8 , %[b0]\n\t"
	              "addq  $8*8 , %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b2]"+r"(B2)
	            :);

	        }
	      }
	      if( K & 4  ){
	      //if( K >> 2 ){
	      //  size_t k4 = ( K >> 2 ); // unrolling with 8 elements
	      //  while( k4-- ){
	            //x00 = *(B+0+0*ldb); x10 = *(B+1+0*ldb); x20 = *(B+2+0*ldb); x30 = *(B+3+0*ldb);
	            //*(B2+0*1+0) = x00; *(B2+1*1+0) = x10; *(B2+2*1+0) = x20; *(B2+3*1+0) = x30;
	            //B+=4;
	            //B2+=4;

	            __asm__ __volatile__ (
	              "\n\t"
	              "vmovapd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	              "\n\t"
	              "vmovapd  %%ymm0 ,   0*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $4*8 , %[b0]\n\t"
	              "addq  $4*8 , %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b2]"+r"(B2)
	            :);

	      //  }
	      }
	      if( K & 2 ){
	            //x00 = *(B+0+0*ldb); x10 = *(B+1+0*ldb);
	            //*(B2+0*1+0) = x00; *(B2+1*1+0) = x10;
	            //B+=2;
	            //B2+=2;

	            __asm__ __volatile__ (
	              "\n\t"
	              "movupd  0*8(%[b0]        ), %%xmm0 \n\t" // [x00,x10]
	              "\n\t"
	              "movapd  %%xmm0 ,   0*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $2*8 , %[b0]\n\t"
	              "addq  $2*8 , %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b2]"+r"(B2)
	            :);

	      }
	      if( K & 1 ){
	            //x00 = *(B+0+0*ldb);
	            //*(B2+0*1+0) = x00;
	            //B+=1;
	            //b0+=1;
	            //B2+=1;

	            __asm__ __volatile__ (
	              "\n\t"
	              "movsd  0*8(%[b0]        ), %%xmm0 \n\t" // [x00,x10]
	              "\n\t"
	              "movsd  %%xmm0 ,   0*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $1*8 , %[b0]\n\t"
	              "addq  $1*8 , %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b2]"+r"(B2)
	            :);

	      }
	      b0 = b0 - K + ldb ;

	  }

	}else{ // Not aligned

	  if( N >> 2 ){
	    size_t n4 = ( N >> 2 );
	    while( n4-- ){

	      //__asm__ __volatile__ (
	      //  "\n\t"
	      //  "prefetcht1  0*8(%[b0],%[ldb2],2)\n\t"
	      //  "prefetcht1  0*8(%[b1],%[ldb2],2)\n\t"
	      //  "prefetcht1  0*8(%[b0],%[ldb3],2)\n\t"
	      //  "prefetcht1  0*8(%[b1],%[ldb3],2)\n\t"
	      //  "\n\t"
	      //:[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	      //:[ldb2]"r"(ldb2),[ldb3]"r"(ldb3));

	      if( K >> 3 ){
	        size_t k8 = ( K >> 3 );
	        k8--;

	        __asm__ __volatile__ (
	          "\n\t"
	          "vmovupd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	          "vmovupd  0*8(%[b1]        ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	          "vmovupd  0*8(%[b0],%[ldb2]), %%ymm2 \n\t" // [x02,x12,x22,x32]
	          "vmovupd  0*8(%[b1],%[ldb2]), %%ymm3 \n\t" // [x03,x13,x23,x33]
	          "\n\t"
	          "vshufpd  $0x00, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [x00,x01,x20,x21]
	          "vshufpd  $0x0f, %%ymm1 , %%ymm0 , %%ymm9 \n\t" // [x10,x11,x30,x31]
	          "vshufpd  $0x00, %%ymm3 , %%ymm2 , %%ymm10\n\t" // [x02,x03,x22,x23]
	          "vshufpd  $0x0f, %%ymm3 , %%ymm2 , %%ymm11\n\t" // [x12,x13,x32,x33]
	          "\n\t"
	        :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	        :[ldb2]"r"(ldb2));


	        while( k8-- ){
	            //x00 = *(B+0+0*ldb); x10 = *(B+1+0*ldb); x20 = *(B+2+0*ldb); x30 = *(B+3+0*ldb);
	            //x01 = *(B+0+1*ldb); x11 = *(B+1+1*ldb); x21 = *(B+2+1*ldb); x31 = *(B+3+1*ldb);
	            //x02 = *(B+0+2*ldb); x12 = *(B+1+2*ldb); x22 = *(B+2+2*ldb); x32 = *(B+3+2*ldb);
	            //x03 = *(B+0+3*ldb); x13 = *(B+1+3*ldb); x23 = *(B+2+3*ldb); x33 = *(B+3+3*ldb);
	            //*(B2+0*4+0) = x00; *(B2+1*4+0) = x10; *(B2+2*4+0) = x20; *(B2+3*4+0) = x30;
	            //*(B2+0*4+1) = x01; *(B2+1*4+1) = x11; *(B2+2*4+1) = x21; *(B2+3*4+1) = x31;
	            //*(B2+0*4+2) = x02; *(B2+1*4+2) = x12; *(B2+2*4+2) = x22; *(B2+3*4+2) = x32;
	            //*(B2+0*4+3) = x03; *(B2+1*4+3) = x13; *(B2+2*4+3) = x23; *(B2+3*4+3) = x33;
	            //x00 = *(B+4+0*ldb); x10 = *(B+5+0*ldb); x20 = *(B+6+0*ldb); x30 = *(B+7+0*ldb);
	            //x01 = *(B+4+1*ldb); x11 = *(B+5+1*ldb); x21 = *(B+6+1*ldb); x31 = *(B+7+1*ldb);
	            //x02 = *(B+4+2*ldb); x12 = *(B+5+2*ldb); x22 = *(B+6+2*ldb); x32 = *(B+7+2*ldb);
	            //x03 = *(B+4+3*ldb); x13 = *(B+5+3*ldb); x23 = *(B+6+3*ldb); x33 = *(B+7+3*ldb);
	            //*(B2+4*4+0) = x00; *(B2+5*4+0) = x10; *(B2+6*4+0) = x20; *(B2+7*4+0) = x30;
	            //*(B2+4*4+1) = x01; *(B2+5*4+1) = x11; *(B2+6*4+1) = x21; *(B2+7*4+1) = x31;
	            //*(B2+4*4+2) = x02; *(B2+5*4+2) = x12; *(B2+6*4+2) = x22; *(B2+7*4+2) = x32;
	            //*(B2+4*4+3) = x03; *(B2+5*4+3) = x13; *(B2+6*4+3) = x23; *(B2+7*4+3) = x33;
	            //B+=8;
	            //B2+=32;

	            __asm__ __volatile__ (
	              "\n\t"
	              "prefetcht0  0*8(%[b0],%[ldb2],2)\n\t"
	              "prefetcht0  0*8(%[b1],%[ldb2],2)\n\t"
	              "prefetcht0  0*8(%[b0],%[ldb3],2)\n\t"
	              "prefetcht0  0*8(%[b1],%[ldb3],2)\n\t"
	              "\n\t"
	              "vmovupd  4*8(%[b0]        ), %%ymm4 \n\t" // [x00,x10,x20,x30]
	              "vmovupd  4*8(%[b1]        ), %%ymm5 \n\t" // [x01,x11,x21,x31]
	              "vperm2f128  $0x20, %%ymm10, %%ymm8 , %%ymm12\n\t" // [x00,x01,x02,x03]
	              "vperm2f128  $0x20, %%ymm11, %%ymm9 , %%ymm13\n\t" // [x10,x11,x12,x13]
	              "\n\t"
	              "vmovupd  4*8(%[b0],%[ldb2]), %%ymm6 \n\t" // [x02,x12,x22,x32]
	              "vmovupd  4*8(%[b1],%[ldb2]), %%ymm7 \n\t" // [x03,x13,x23,x33]
	              "vperm2f128  $0x31, %%ymm10, %%ymm8 , %%ymm14\n\t" // [x20,x21,x22,x23]
	              "vperm2f128  $0x31, %%ymm11, %%ymm9 , %%ymm15\n\t" // [x30,x31,x32,x33]
	              "\n\t"
	              "vshufpd  $0x00, %%ymm5 , %%ymm4 , %%ymm8 \n\t" // [x00,x01,x20,x21]
	              "vshufpd  $0x0f, %%ymm5 , %%ymm4 , %%ymm9 \n\t" // [x10,x11,x30,x31]
	              "vmovapd  %%ymm12,   0*8(%[b2])\n\t"
	              "vmovapd  %%ymm13,   4*8(%[b2])\n\t"
	              "\n\t"
	              "vshufpd  $0x00, %%ymm7 , %%ymm6 , %%ymm10\n\t" // [x02,x03,x22,x23]
	              "vshufpd  $0x0f, %%ymm7 , %%ymm6 , %%ymm11\n\t" // [x12,x13,x32,x33]
	              "vmovapd  %%ymm14,   8*8(%[b2])\n\t"
	              "vmovapd  %%ymm15,  12*8(%[b2])\n\t"
	              "\n\t"
	              "vmovupd  8*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	              "vmovupd  8*8(%[b1]        ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	              "vperm2f128  $0x20, %%ymm10, %%ymm8 , %%ymm12\n\t" // [x00,x01,x02,x03]
	              "vperm2f128  $0x20, %%ymm11, %%ymm9 , %%ymm13\n\t" // [x10,x11,x12,x13]
	              "\n\t"
	              "vmovupd  8*8(%[b0],%[ldb2]), %%ymm2 \n\t" // [x02,x12,x22,x32]
	              "vmovupd  8*8(%[b1],%[ldb2]), %%ymm3 \n\t" // [x03,x13,x23,x33]
	              "vperm2f128  $0x31, %%ymm10, %%ymm8 , %%ymm14\n\t" // [x20,x21,x22,x23]
	              "vperm2f128  $0x31, %%ymm11, %%ymm9 , %%ymm15\n\t" // [x30,x31,x32,x33]
	              "\n\t"
	              "vshufpd  $0x00, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [x00,x01,x20,x21]
	              "vshufpd  $0x0f, %%ymm1 , %%ymm0 , %%ymm9 \n\t" // [x10,x11,x30,x31]
	              "vmovapd  %%ymm12,  16*8(%[b2])\n\t"
	              "vmovapd  %%ymm13,  20*8(%[b2])\n\t"
	              "\n\t"
	              "vshufpd  $0x00, %%ymm3 , %%ymm2 , %%ymm10\n\t" // [x02,x03,x22,x23]
	              "vshufpd  $0x0f, %%ymm3 , %%ymm2 , %%ymm11\n\t" // [x12,x13,x32,x33]
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

	        __asm__ __volatile__ (
	          "\n\t"
	          "vmovupd  4*8(%[b0]        ), %%ymm4 \n\t" // [x00,x10,x20,x30]
	          "vmovupd  4*8(%[b1]        ), %%ymm5 \n\t" // [x01,x11,x21,x31]
	          "vperm2f128  $0x20, %%ymm10, %%ymm8 , %%ymm12\n\t" // [x00,x01,x02,x03]
	          "vperm2f128  $0x20, %%ymm11, %%ymm9 , %%ymm13\n\t" // [x10,x11,x12,x13]
	          "\n\t"
	          "vmovupd  4*8(%[b0],%[ldb2]), %%ymm6 \n\t" // [x02,x12,x22,x32]
	          "vmovupd  4*8(%[b1],%[ldb2]), %%ymm7 \n\t" // [x03,x13,x23,x33]
	          "vperm2f128  $0x31, %%ymm10, %%ymm8 , %%ymm14\n\t" // [x20,x21,x22,x23]
	          "vperm2f128  $0x31, %%ymm11, %%ymm9 , %%ymm15\n\t" // [x30,x31,x32,x33]
	          "\n\t"
	          "vshufpd  $0x00, %%ymm5 , %%ymm4 , %%ymm8 \n\t" // [x00,x01,x20,x21]
	          "vshufpd  $0x0f, %%ymm5 , %%ymm4 , %%ymm9 \n\t" // [x10,x11,x30,x31]
	          "vmovapd  %%ymm12,   0*8(%[b2])\n\t"
	          "vmovapd  %%ymm13,   4*8(%[b2])\n\t"
	          "\n\t"
	          "vshufpd  $0x00, %%ymm7 , %%ymm6 , %%ymm10\n\t" // [x02,x03,x22,x23]
	          "vshufpd  $0x0f, %%ymm7 , %%ymm6 , %%ymm11\n\t" // [x12,x13,x32,x33]
	          "vmovapd  %%ymm14,   8*8(%[b2])\n\t"
	          "vmovapd  %%ymm15,  12*8(%[b2])\n\t"
	          "\n\t"
	          "vperm2f128  $0x20, %%ymm10, %%ymm8 , %%ymm12\n\t" // [x00,x01,x02,x03]
	          "vperm2f128  $0x31, %%ymm10, %%ymm8 , %%ymm14\n\t" // [x20,x21,x22,x23]
	          "vperm2f128  $0x20, %%ymm11, %%ymm9 , %%ymm13\n\t" // [x10,x11,x12,x13]
	          "vperm2f128  $0x31, %%ymm11, %%ymm9 , %%ymm15\n\t" // [x30,x31,x32,x33]
	          "\n\t"
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
	      if( K & 4  ){
	      //if( K >> 2 ){
	      //  size_t k4 = ( K >> 2 ); // unrolling with 8 elements
	      //  while( k4-- ){
	            //x00 = *(B+0+0*ldb); x10 = *(B+1+0*ldb); x20 = *(B+2+0*ldb); x30 = *(B+3+0*ldb);
	            //x01 = *(B+0+1*ldb); x11 = *(B+1+1*ldb); x21 = *(B+2+1*ldb); x31 = *(B+3+1*ldb);
	            //x02 = *(B+0+2*ldb); x12 = *(B+1+2*ldb); x22 = *(B+2+2*ldb); x32 = *(B+3+2*ldb);
	            //x03 = *(B+0+3*ldb); x13 = *(B+1+3*ldb); x23 = *(B+2+3*ldb); x33 = *(B+3+3*ldb);
	            //*(B2+0*4+0) = x00; *(B2+1*4+0) = x10; *(B2+2*4+0) = x20; *(B2+3*4+0) = x30;
	            //*(B2+0*4+1) = x01; *(B2+1*4+1) = x11; *(B2+2*4+1) = x21; *(B2+3*4+1) = x31;
	            //*(B2+0*4+2) = x02; *(B2+1*4+2) = x12; *(B2+2*4+2) = x22; *(B2+3*4+2) = x32;
	            //*(B2+0*4+3) = x03; *(B2+1*4+3) = x13; *(B2+2*4+3) = x23; *(B2+3*4+3) = x33;
	            //B+=4;
	            //B2+=16;

	            __asm__ __volatile__ (
	              "\n\t"
	              "vmovupd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	              "vmovupd  0*8(%[b1]        ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	              "vmovupd  0*8(%[b0],%[ldb2]), %%ymm2 \n\t" // [x02,x12,x22,x32]
	              "vmovupd  0*8(%[b1],%[ldb2]), %%ymm3 \n\t" // [x03,x13,x23,x33]
	              "\n\t"
	              "vshufpd  $0x00, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [x00,x01,x20,x21]
	              "vshufpd  $0x0f, %%ymm1 , %%ymm0 , %%ymm9 \n\t" // [x10,x11,x30,x31]
	              "vshufpd  $0x00, %%ymm3 , %%ymm2 , %%ymm10\n\t" // [x02,x03,x22,x23]
	              "vshufpd  $0x0f, %%ymm3 , %%ymm2 , %%ymm11\n\t" // [x12,x13,x32,x33]
	              "\n\t"
	              "vperm2f128  $0x20, %%ymm10, %%ymm8 , %%ymm12\n\t" // [x00,x01,x02,x03]
	              "vperm2f128  $0x31, %%ymm10, %%ymm8 , %%ymm14\n\t" // [x20,x21,x22,x23]
	              "vperm2f128  $0x20, %%ymm11, %%ymm9 , %%ymm13\n\t" // [x10,x11,x12,x13]
	              "vperm2f128  $0x31, %%ymm11, %%ymm9 , %%ymm15\n\t" // [x30,x31,x32,x33]
	              "\n\t"
	              "vmovapd  %%ymm12,   0*8(%[b2])\n\t"
	              "vmovapd  %%ymm13,   4*8(%[b2])\n\t"
	              "vmovapd  %%ymm14,   8*8(%[b2])\n\t"
	              "vmovapd  %%ymm15,  12*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $4*8 , %[b0]\n\t"
	              "addq  $4*8 , %[b1]\n\t"
	              "addq  $16*8, %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	            :[ldb2]"r"(ldb2));

	      //  }
	      }
	      if( K & 2 ){
	            //x00 = *(B+0+0*ldb); x10 = *(B+1+0*ldb);
	            //x01 = *(B+0+1*ldb); x11 = *(B+1+1*ldb);
	            //x02 = *(B+0+2*ldb); x12 = *(B+1+2*ldb);
	            //x03 = *(B+0+3*ldb); x13 = *(B+1+3*ldb);
	            //*(B2+0*4+0) = x00; *(B2+1*4+0) = x10;
	            //*(B2+0*4+1) = x01; *(B2+1*4+1) = x11;
	            //*(B2+0*4+2) = x02; *(B2+1*4+2) = x12;
	            //*(B2+0*4+3) = x03; *(B2+1*4+3) = x13;
	            //B+=2;
	            //B2+=8;

	            __asm__ __volatile__ (
	              "\n\t"
	              "vmovupd  0*8(%[b0]        ), %%xmm0 \n\t" // [x00,x10]
	              "vmovupd  0*8(%[b1]        ), %%xmm1 \n\t" // [x01,x11]
	              "vmovupd  0*8(%[b0],%[ldb2]), %%xmm2 \n\t" // [x02,x12]
	              "vmovupd  0*8(%[b1],%[ldb2]), %%xmm3 \n\t" // [x03,x13]
	              "\n\t"
	              "vshufpd  $0x00, %%xmm1 , %%xmm0 , %%xmm12\n\t" // [x00,x01]
	              "vshufpd  $0x0f, %%xmm1 , %%xmm0 , %%xmm14\n\t" // [x10,x11]
	              "vshufpd  $0x00, %%xmm3 , %%xmm2 , %%xmm13\n\t" // [x02,x03]
	              "vshufpd  $0x0f, %%xmm3 , %%xmm2 , %%xmm15\n\t" // [x12,x13]
	              "\n\t"
	              "vmovapd  %%xmm12,   0*8(%[b2])\n\t"
	              "vmovapd  %%xmm13,   2*8(%[b2])\n\t"
	              "vmovapd  %%xmm14,   4*8(%[b2])\n\t"
	              "vmovapd  %%xmm15,   6*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $2*8 , %[b0]\n\t"
	              "addq  $2*8 , %[b1]\n\t"
	              "addq  $8*8 , %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	            :[ldb2]"r"(ldb2));

	      }
	      if( K & 1 ){
	            //x00 = *(B+0+0*ldb);
	            //x01 = *(B+0+1*ldb);
	            //x02 = *(B+0+2*ldb);
	            //x03 = *(B+0+3*ldb);
	            //*(B2+0*4+0) = x00;
	            //*(B2+0*4+1) = x01;
	            //*(B2+0*4+2) = x02;
	            //*(B2+0*4+3) = x03;
	            //B+=1;
	            //B2+=4;

	            __asm__ __volatile__ (
	              "\n\t"
	              "movsd  0*8(%[b0]        ), %%xmm12\n\t" // [x00]
	              "movsd  0*8(%[b1]        ), %%xmm13\n\t" // [x01]
	              "movsd  0*8(%[b0],%[ldb2]), %%xmm14\n\t" // [x02]
	              "movsd  0*8(%[b1],%[ldb2]), %%xmm15\n\t" // [x03]
	              "\n\t"
	              "movlpd  %%xmm12,   0*8(%[b2])\n\t"
	              "movlpd  %%xmm13,   1*8(%[b2])\n\t"
	              "movlpd  %%xmm14,   2*8(%[b2])\n\t"
	              "movlpd  %%xmm15,   3*8(%[b2])\n\t"
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
	        k8--;

	        __asm__ __volatile__ (
	          "\n\t"
	          "movupd  0*8(%[b0]), %%xmm0 \n\t" // [x00,x10]
	          "movupd  2*8(%[b0]), %%xmm1 \n\t" // [x20,x30]
	          "movupd  0*8(%[b1]), %%xmm2 \n\t" // [x01,x11]
	          "movupd  2*8(%[b1]), %%xmm3 \n\t" // [x21,x31]
	          "\n\t"
	        :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	        :);


	        while( k8-- ){
	            //x00 = *(B+0+0*ldb); x10 = *(B+1+0*ldb); x20 = *(B+2+0*ldb); x30 = *(B+3+0*ldb);
	            //x01 = *(B+0+1*ldb); x11 = *(B+1+1*ldb); x21 = *(B+2+1*ldb); x31 = *(B+3+1*ldb);
	            //*(B2+0*2+0) = x00; *(B2+1*2+0) = x10; *(B2+2*2+0) = x20; *(B2+3*2+0) = x30;
	            //*(B2+0*2+1) = x01; *(B2+1*2+1) = x11; *(B2+2*2+1) = x21; *(B2+3*2+1) = x31;
	            //x00 = *(B+4+0*ldb); x10 = *(B+5+0*ldb); x20 = *(B+6+0*ldb); x30 = *(B+7+0*ldb);
	            //x01 = *(B+4+1*ldb); x11 = *(B+5+1*ldb); x21 = *(B+6+1*ldb); x31 = *(B+7+1*ldb);
	            //*(B2+4*2+0) = x00; *(B2+5*2+0) = x10; *(B2+6*2+0) = x20; *(B2+7*2+0) = x30;
	            //*(B2+4*2+1) = x01; *(B2+5*2+1) = x11; *(B2+6*2+1) = x21; *(B2+7*2+1) = x31;
	            //B+=8;
	            //B2+=16;

	            __asm__ __volatile__ (
	              "\n\t"
	              "vshufpd  $0x00, %%xmm2 , %%xmm0 , %%xmm8 \n\t" // [x00,x01]
	              "vshufpd  $0x03, %%xmm2 , %%xmm0 , %%xmm9 \n\t" // [x10,x11]
	              "movupd  4*8(%[b0]), %%xmm4 \n\t" // [x00,x10]
	              "movupd  4*8(%[b1]), %%xmm6 \n\t" // [x01,x11]
	              "movapd  %%xmm8 ,   0*8(%[b2])\n\t"
	              "movapd  %%xmm9 ,   2*8(%[b2])\n\t"
	              "\n\t"
	              "vshufpd  $0x00, %%xmm3 , %%xmm1 , %%xmm10\n\t" // [x20,x21]
	              "vshufpd  $0x03, %%xmm3 , %%xmm1 , %%xmm11\n\t" // [x30,x31]
	              "movupd  6*8(%[b0]), %%xmm5 \n\t" // [x20,x30]
	              "movupd  6*8(%[b1]), %%xmm7 \n\t" // [x21,x31]
	              "movapd  %%xmm10,   4*8(%[b2])\n\t"
	              "movapd  %%xmm11,   6*8(%[b2])\n\t"
	              "\n\t"
	              "vshufpd  $0x00, %%xmm6 , %%xmm4 , %%xmm12\n\t" // [x00,x01]
	              "vshufpd  $0x03, %%xmm6 , %%xmm4 , %%xmm13\n\t" // [x10,x11]
	              "movupd  8*8(%[b0]), %%xmm0 \n\t" // [x00,x10]
	              "movupd  8*8(%[b1]), %%xmm2 \n\t" // [x01,x11]
	              "movapd  %%xmm12,   8*8(%[b2])\n\t"
	              "movapd  %%xmm13,  10*8(%[b2])\n\t"
	              "\n\t"
	              "vshufpd  $0x00, %%xmm7 , %%xmm5 , %%xmm14\n\t" // [x20,x21]
	              "vshufpd  $0x03, %%xmm7 , %%xmm5 , %%xmm15\n\t" // [x30,x31]
	              "movupd 10*8(%[b0]), %%xmm1 \n\t" // [x20,x30]
	              "movupd 10*8(%[b1]), %%xmm3 \n\t" // [x21,x31]
	              "movapd  %%xmm14,  12*8(%[b2])\n\t"
	              "movapd  %%xmm15,  14*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $8*8 , %[b0]\n\t"
	              "addq  $8*8 , %[b1]\n\t"
	              "addq  $16*8, %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	            :);

	        }

	        __asm__ __volatile__ (
	          "\n\t"
	          "movupd  4*8(%[b0]), %%xmm4 \n\t" // [x00,x10]
	          "movupd  6*8(%[b0]), %%xmm5 \n\t" // [x20,x30]
	          "vshufpd  $0x00, %%xmm2 , %%xmm0 , %%xmm8 \n\t" // [x00,x01]
	          "vshufpd  $0x03, %%xmm2 , %%xmm0 , %%xmm9 \n\t" // [x10,x11]
	          "\n\t"
	          "movupd  4*8(%[b1]), %%xmm6 \n\t" // [x01,x11]
	          "movupd  6*8(%[b1]), %%xmm7 \n\t" // [x21,x31]
	          "vshufpd  $0x00, %%xmm3 , %%xmm1 , %%xmm10\n\t" // [x20,x21]
	          "vshufpd  $0x03, %%xmm3 , %%xmm1 , %%xmm11\n\t" // [x30,x31]
	          "\n\t"
	          "movapd  %%xmm8 ,   0*8(%[b2])\n\t"
	          "movapd  %%xmm9 ,   2*8(%[b2])\n\t"
	          "vshufpd  $0x00, %%xmm6 , %%xmm4 , %%xmm12\n\t" // [x00,x01]
	          "vshufpd  $0x03, %%xmm6 , %%xmm4 , %%xmm13\n\t" // [x10,x11]
	          "\n\t"
	          "movapd  %%xmm10,   4*8(%[b2])\n\t"
	          "movapd  %%xmm11,   6*8(%[b2])\n\t"
	          "vshufpd  $0x00, %%xmm7 , %%xmm5 , %%xmm14\n\t" // [x20,x21]
	          "vshufpd  $0x03, %%xmm7 , %%xmm5 , %%xmm15\n\t" // [x30,x31]
	          "\n\t"
	          "movapd  %%xmm12,   8*8(%[b2])\n\t"
	          "movapd  %%xmm13,  10*8(%[b2])\n\t"
	          "movapd  %%xmm14,  12*8(%[b2])\n\t"
	          "movapd  %%xmm15,  14*8(%[b2])\n\t"
	          "\n\t"
	          "addq  $8*8 , %[b0]\n\t"
	          "addq  $8*8 , %[b1]\n\t"
	          "addq  $16*8, %[b2]\n\t"
	          "\n\t"
	        :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	        :);
	      }
	      if( K & 4  ){
	      //if( K >> 2 ){
	      //  size_t k4 = ( K >> 2 ); // unrolling with 8 elements
	      //  while( k4-- ){
	            //x00 = *(B+0+0*ldb); x10 = *(B+1+0*ldb); x20 = *(B+2+0*ldb); x30 = *(B+3+0*ldb);
	            //x01 = *(B+0+1*ldb); x11 = *(B+1+1*ldb); x21 = *(B+2+1*ldb); x31 = *(B+3+1*ldb);
	            //*(B2+0*2+0) = x00; *(B2+1*2+0) = x10; *(B2+2*2+0) = x20; *(B2+3*2+0) = x30;
	            //*(B2+0*2+1) = x01; *(B2+1*2+1) = x11; *(B2+2*2+1) = x21; *(B2+3*2+1) = x31;
	            //B+=4;
	            //B2+=8;

	            __asm__ __volatile__ (
	              "\n\t"
	              "movupd  0*8(%[b0]), %%xmm0 \n\t" // [x00,x10]
	              "movupd  2*8(%[b0]), %%xmm1 \n\t" // [x20,x30]
	              "movupd  0*8(%[b1]), %%xmm2 \n\t" // [x01,x11]
	              "movupd  2*8(%[b1]), %%xmm3 \n\t" // [x21,x31]
	              "\n\t"
	              "vshufpd  $0x00, %%xmm2 , %%xmm0 , %%xmm8 \n\t" // [x00,x01]
	              "vshufpd  $0x03, %%xmm2 , %%xmm0 , %%xmm9 \n\t" // [x10,x11]
	              "vshufpd  $0x00, %%xmm3 , %%xmm1 , %%xmm10\n\t" // [x20,x21]
	              "vshufpd  $0x03, %%xmm3 , %%xmm1 , %%xmm11\n\t" // [x30,x31]
	              "\n\t"
	              "movapd  %%xmm8 ,   0*8(%[b2])\n\t"
	              "movapd  %%xmm9 ,   2*8(%[b2])\n\t"
	              "movapd  %%xmm10,   4*8(%[b2])\n\t"
	              "movapd  %%xmm11,   6*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $4*8 , %[b0]\n\t"
	              "addq  $4*8 , %[b1]\n\t"
	              "addq  $8*8 , %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	            :);

	      //  }
	      }
	      if( K & 2 ){
	            //x00 = *(B+0+0*ldb); x10 = *(B+1+0*ldb);
	            //x01 = *(B+0+1*ldb); x11 = *(B+1+1*ldb);
	            //*(B2+0*2+0) = x00; *(B2+1*2+0) = x10;
	            //*(B2+0*2+1) = x01; *(B2+1*2+1) = x11;
	            //B+=2;
	            //B2+=4;

	            __asm__ __volatile__ (
	              "\n\t"
	              "movupd  0*8(%[b0]), %%xmm0 \n\t" // [x00,x10]
	              "movupd  0*8(%[b1]), %%xmm2 \n\t" // [x01,x11]
	              "\n\t"
	              "vshufpd  $0x00, %%xmm2 , %%xmm0 , %%xmm8 \n\t" // [x00,x01]
	              "vshufpd  $0x03, %%xmm2 , %%xmm0 , %%xmm9 \n\t" // [x10,x11]
	              "\n\t"
	              "movapd  %%xmm8 ,   0*8(%[b2])\n\t"
	              "movapd  %%xmm9 ,   2*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $2*8 , %[b0]\n\t"
	              "addq  $2*8 , %[b1]\n\t"
	              "addq  $4*8 , %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	            :);

	      }
	      if( K & 1 ){
	            //x00 = *(B+0+0*ldb);
	            //x01 = *(B+0+1*ldb);
	            //*(B2+0*2+0) = x00;
	            //*(B2+0*2+1) = x01;
	            //B+=1;
	            //B2+=2;

	            __asm__ __volatile__ (
	              "\n\t"
	              "movsd   0*8(%[b0]), %%xmm0 \n\t" // [x00,   ]
	              "movhpd  0*8(%[b1]), %%xmm0 \n\t" // [x00,x01]
	              "\n\t"
	              "movapd  %%xmm0 ,   0*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $1*8 , %[b0]\n\t"
	              "addq  $1*8 , %[b1]\n\t"
	              "addq  $2*8 , %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b1]"+r"(b1),[b2]"+r"(B2)
	            :);

	      }
	      b0 = b0 - K + 2*ldb ;
	      b1 = b1 - K + 2*ldb ;

	  }
	  if( N & 1 ){

	      if( K >> 3 ){
	        size_t k8 = ( K >> 3 );
	        while( k8-- ){
	            //x00 = *(B+0+0*ldb); x10 = *(B+1+0*ldb); x20 = *(B+2+0*ldb); x30 = *(B+3+0*ldb);
	            //*(B2+0*1+0) = x00; *(B2+1*1+0) = x10; *(B2+2*1+0) = x20; *(B2+3*1+0) = x30;
	            //x00 = *(B+4+0*ldb); x10 = *(B+5+0*ldb); x20 = *(B+6+0*ldb); x30 = *(B+7+0*ldb);
	            //*(B2+4*1+0) = x00; *(B2+5*1+0) = x10; *(B2+6*1+0) = x20; *(B2+7*1+0) = x30;
	            //B+=8;
	            //B2+=8;

	            __asm__ __volatile__ (
	              "\n\t"
	              "vmovupd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	              "vmovupd  4*8(%[b0]        ), %%ymm4 \n\t" // [x00,x10,x20,x30]
	              "\n\t"
	              "vmovupd  %%ymm0 ,   0*8(%[b2])\n\t"
	              "vmovupd  %%ymm4 ,   4*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $8*8 , %[b0]\n\t"
	              "addq  $8*8 , %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b2]"+r"(B2)
	            :);

	        }
	      }
	      if( K & 4  ){
	      //if( K >> 2 ){
	      //  size_t k4 = ( K >> 2 ); // unrolling with 8 elements
	      //  while( k4-- ){
	            //x00 = *(B+0+0*ldb); x10 = *(B+1+0*ldb); x20 = *(B+2+0*ldb); x30 = *(B+3+0*ldb);
	            //*(B2+0*1+0) = x00; *(B2+1*1+0) = x10; *(B2+2*1+0) = x20; *(B2+3*1+0) = x30;
	            //B+=4;
	            //B2+=4;

	            __asm__ __volatile__ (
	              "\n\t"
	              "vmovupd  0*8(%[b0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	              "\n\t"
	              "vmovupd  %%ymm0 ,   0*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $4*8 , %[b0]\n\t"
	              "addq  $4*8 , %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b2]"+r"(B2)
	            :);

	      //  }
	      }
	      if( K & 2 ){
	            //x00 = *(B+0+0*ldb); x10 = *(B+1+0*ldb);
	            //*(B2+0*1+0) = x00; *(B2+1*1+0) = x10;
	            //B+=2;
	            //B2+=2;

	            __asm__ __volatile__ (
	              "\n\t"
	              "movupd  0*8(%[b0]        ), %%xmm0 \n\t" // [x00,x10]
	              "\n\t"
	              "movupd  %%xmm0 ,   0*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $2*8 , %[b0]\n\t"
	              "addq  $2*8 , %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b2]"+r"(B2)
	            :);

	      }
	      if( K & 1 ){
	            //x00 = *(B+0+0*ldb);
	            //*(B2+0*1+0) = x00;
	            //B+=1;
	            //b0+=1;
	            //B2+=1;

	            __asm__ __volatile__ (
	              "\n\t"
	              "movsd  0*8(%[b0]        ), %%xmm0 \n\t" // [x00,x10]
	              "\n\t"
	              "movsd  %%xmm0 ,   0*8(%[b2])\n\t"
	              "\n\t"
	              "addq  $1*8 , %[b0]\n\t"
	              "addq  $1*8 , %[b2]\n\t"
	              "\n\t"
	            :[b0]"+r"(b0),[b2]"+r"(B2)
	            :);

	      }
	      b0 = b0 - K + ldb ;

	  }

	}

}


