#include "myblas_internal.h"
#include <stdio.h>

void myblas_dgemm_kernel_detail(
         size_t M, size_t N, size_t K,
         double alpha, const double *A, const double *B, 
         double *C, size_t ldc )
{
	double *c0 = C;
	double *c1 = C + ldc;
	size_t ldc2 = ldc * 2 * sizeof(double);
	double alpha4[4] = {alpha,alpha,alpha,alpha};

	size_t NQ = N/6;
	size_t NR = N%6;

	__asm__ __volatile__ (
	    "\n\t"
	    "prefetcht0 0*8(%[a])\n\t"
	    "prefetcht0 0*8(%[b])\n\t"
	    "prefetcht0 4*8(%[a])\n\t"
	    "prefetcht0 4*8(%[b])\n\t"
	    "prefetcht0 8*8(%[b])\n\t"
	    "\n\t"
	    :[a]"+r"(A),[b]"+r"(B)
	:);

	if( NQ ){
	  size_t n4 = NQ; // unrolling N
	  //__asm__ __volatile__ (
	  //    "\n\t"
	  //    "prefetcht0 0*8(%[a])\n\t"
	  //    "prefetcht0 0*8(%[b])\n\t"
	  //    "prefetcht0 4*8(%[a])\n\t"
	  //    "prefetcht0 4*8(%[b])\n\t"
	  //    "prefetcht0 8*8(%[b])\n\t"
	  //    "\n\t"
	  //    :[a]"+r"(A),[b]"+r"(B)
	  //:);
	  while( n4-- ){
	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 ); // unrolling M
	      while( m4-- ){

	        __asm__ __volatile__ (
	            "prefetcht0 0*8(%[c0]          )\n\t"
	            "prefetcht0 0*8(%[c1]          )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0 0*8(%[c1],%[ldc2]  )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc2],2)\n\t"
	            "prefetcht0 0*8(%[c1],%[ldc2],2)\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );

	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm4 , %%ymm4 , %%ymm4 \n\t"
	            "vpxor  %%ymm5 , %%ymm5 , %%ymm5 \n\t"
	            "vpxor  %%ymm6 , %%ymm6 , %%ymm6 \n\t"
	            "vpxor  %%ymm7 , %%ymm7 , %%ymm7 \n\t"
	            "vpxor  %%ymm8 , %%ymm8 , %%ymm8 \n\t"
	            "vpxor  %%ymm9 , %%ymm9 , %%ymm9 \n\t"
	            "vpxor  %%ymm10, %%ymm10, %%ymm10\n\t"
	            "vpxor  %%ymm11, %%ymm11, %%ymm11\n\t"
	            "vpxor  %%ymm12, %%ymm12, %%ymm12\n\t"
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);

	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            // 4x6x4x2  LD+PM/FMA = 10/24 = 0.42
	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0  96*8(%[a])\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovupd   0*8(%[b]), %%ymm2 \n\t" // [b00,b10,b01,b11]
	                "prefetcht0 144*8(%[b])\n\t"
	                "prefetcht0 152*8(%[b])\n\t"
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b01,b11,b00,b10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm5 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm6 \n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c21,c21,c30,c30]
	                "\n\t"
	                "vmovupd   4*8(%[b]), %%ymm2 \n\t" // [b02,b12,b03,b13]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm10\n\t" // [c22,c22,c33,c33]
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b03,b13,b02,b12]
	                "vmovupd   8*8(%[b]), %%ymm2 \n\t" // [b04,b14,b05,b15]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c23,c23,c32,c32]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b05,b15,b04,b14]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c04,c15,c15]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm14\n\t" // [c24,c24,c35,c35]
	                "vmovupd  12*8(%[b]), %%ymm2 \n\t" // [b20,b30,b21,b31]
	                "prefetcht0 104*8(%[a])\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm13\n\t" // [c05,c05,c14,c14]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c25,c25,c34,c34]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm0 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm1 \n\t" // [a22,a32,a23,a33]
	                "prefetcht0 160*8(%[b])\n\t"
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b21,b31,b20,b30]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm6 \n\t" // [c20,c20,c31,c31]
	                "vmovupd  16*8(%[b]), %%ymm2 \n\t" // [b22,b32,b23,b33]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm5 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c21,c21,c30,c30]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b23,b33,b22,b32]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm10\n\t" // [c22,c22,c33,c33]
	                "vmovupd  20*8(%[b]), %%ymm2 \n\t" // [b04,b14,b05,b15]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c23,c23,c32,c32]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b05,b15,b04,b14]
	                "prefetcht0 112*8(%[a])\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c04,c15,c15]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm14\n\t" // [c24,c24,c35,c35]
	                "vmovupd  24*8(%[b]), %%ymm2 \n\t" // [b40,b50,b41,b51]
	                "prefetcht0 168*8(%[b])\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm13\n\t" // [c05,c05,c14,c14]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c25,c25,c34,c34]
	                "\n\t"
	                "\n\t"
	                "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a40,a50,a41,a51]
	                "vmovapd  20*8(%[a]), %%ymm1 \n\t" // [a42,a52,a43,a53]
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b41,b51,b40,b50]
	                "prefetcht0 176*8(%[b])\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm6 \n\t" // [c20,c20,c31,c31]
	                "vmovupd  28*8(%[b]), %%ymm2 \n\t" // [b42,b52,b43,b53]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm5 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c21,c21,c30,c30]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b43,b53,b42,b52]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm10\n\t" // [c22,c22,c33,c33]
	                "vmovupd  32*8(%[b]), %%ymm2 \n\t" // [b44,b54,b45,b55]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c23,c23,c32,c32]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b45,b55,b44,b54]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c04,c15,c15]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm14\n\t" // [c24,c24,c35,c35]
	                "vmovupd  36*8(%[b]), %%ymm2 \n\t" // [b60,b70,b61,b71]
	                "prefetcht0 120*8(%[a])\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm13\n\t" // [c05,c05,c14,c14]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c25,c25,c34,c34]
	                "\n\t"
	                "\n\t"
	                "vmovapd  24*8(%[a]), %%ymm0 \n\t" // [a60,a70,a61,a71]
	                "vmovapd  28*8(%[a]), %%ymm1 \n\t" // [a62,a72,a63,a73]
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b61,b71,b60,b70]
	                "prefetcht0 184*8(%[b])\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm6 \n\t" // [c20,c20,c31,c31]
	                "vmovupd  40*8(%[b]), %%ymm2 \n\t" // [b62,b72,b63,b73]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm5 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c21,c21,c30,c30]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b63,b73,b62,b72]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm10\n\t" // [c22,c22,c33,c33]
	                "vmovupd  44*8(%[b]), %%ymm2 \n\t" // [b64,b74,b65,b75]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c23,c23,c32,c32]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b65,b75,b64,b74]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c04,c15,c15]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm14\n\t" // [c24,c24,c35,c35]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm13\n\t" // [c05,c05,c14,c14]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c25,c25,c34,c34]
	                "\n\t"
	                "addq  $32*8 , %[a]\n\t"
	                "addq  $48*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          }

	        }
	        if( K & 4 ){
	        //if( K >> 2 ){
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){

	            // 4x6x4x2  LD+PM/FMA = 10/24 = 0.42
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovupd   0*8(%[b]), %%ymm2 \n\t" // [b00,b10,b01,b11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b01,b11,b00,b10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm5 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm6 \n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c21,c21,c30,c30]
	                "\n\t"
	                "vmovupd   4*8(%[b]), %%ymm2 \n\t" // [b02,b12,b03,b13]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm10\n\t" // [c22,c22,c33,c33]
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b03,b13,b02,b12]
	                "vmovupd   8*8(%[b]), %%ymm2 \n\t" // [b04,b14,b05,b15]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c23,c23,c32,c32]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b05,b15,b04,b14]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c04,c15,c15]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm14\n\t" // [c24,c24,c35,c35]
	                "vmovupd  12*8(%[b]), %%ymm2 \n\t" // [b20,b30,b21,b31]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm13\n\t" // [c05,c05,c14,c14]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c25,c25,c34,c34]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm0 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm1 \n\t" // [a22,a32,a23,a33]
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b21,b31,b20,b30]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm6 \n\t" // [c20,c20,c31,c31]
	                "vmovupd  16*8(%[b]), %%ymm2 \n\t" // [b22,b32,b23,b33]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm5 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c21,c21,c30,c30]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b23,b33,b22,b32]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm10\n\t" // [c22,c22,c33,c33]
	                "vmovupd  20*8(%[b]), %%ymm2 \n\t" // [b04,b14,b05,b15]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c23,c23,c32,c32]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b05,b15,b04,b14]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c04,c15,c15]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm14\n\t" // [c24,c24,c35,c35]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm13\n\t" // [c05,c05,c14,c14]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c25,c25,c34,c34]
	                "\n\t"

	                ////"prefetcht0  16*8(%[a])\n\t"
	                ////"prefetcht0  24*8(%[b])\n\t"
	                //"\n\t"
	                //"vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                //"vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                //"\n\t"
	                //"vmovupd   0*8(%[b]), %%ymm2 \n\t" // [b00,b10,b01,b11]
	                //"vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b01,b11,b00,b10]
	                //"vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c00,c11,c11]
	                //"vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm5 \n\t" // [c01,c01,c10,c10]
	                //"vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm6 \n\t" // [c20,c20,c31,c31]
	                //"vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c21,c21,c30,c30]
	                //"\n\t"
	                //"vmovupd   4*8(%[b]), %%ymm2 \n\t" // [b02,b12,b03,b13]
	                //"vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b03,b13,b02,b12]
	                //"vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c02,c13,c13]
	                //"vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c03,c03,c12,c12]
	                //"vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm10\n\t" // [c22,c22,c33,c33]
	                //"vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c23,c23,c32,c32]
	                //"\n\t"
	                //"vmovupd   8*8(%[b]), %%ymm2 \n\t" // [b04,b14,b05,b15]
	                //"vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b05,b15,b04,b14]
	                //"vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c04,c15,c15]
	                //"vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm13\n\t" // [c05,c05,c14,c14]
	                //"vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm14\n\t" // [c24,c24,c35,c35]
	                //"vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c25,c25,c34,c34]
	                //"\n\t"
	                ////"prefetcht0 112*8(%[a])\n\t"
	                ////"prefetcht0 160*8(%[b])\n\t"
	                //"\n\t"
	                //"vmovapd   8*8(%[a]), %%ymm0 \n\t" // [a20,a30,a21,a31]
	                //"vmovapd  12*8(%[a]), %%ymm1 \n\t" // [a22,a32,a23,a33]
	                //"\n\t"
	                //"vmovupd  12*8(%[b]), %%ymm2 \n\t" // [b20,b30,b21,b31]
	                //"vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b21,b31,b20,b30]
	                //"vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c00,c11,c11]
	                //"vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm5 \n\t" // [c01,c01,c10,c10]
	                //"vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm6 \n\t" // [c20,c20,c31,c31]
	                //"vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c21,c21,c30,c30]
	                //"\n\t"
	                //"vmovupd  16*8(%[b]), %%ymm2 \n\t" // [b22,b32,b23,b33]
	                //"vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b23,b33,b22,b32]
	                //"vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c02,c13,c13]
	                //"vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c03,c03,c12,c12]
	                //"vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm10\n\t" // [c22,c22,c33,c33]
	                //"vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c23,c23,c32,c32]
	                //"\n\t"
	                //"vmovupd  20*8(%[b]), %%ymm2 \n\t" // [b04,b14,b05,b15]
	                //"vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b05,b15,b04,b14]
	                //"vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c04,c15,c15]
	                //"vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm13\n\t" // [c05,c05,c14,c14]
	                //"vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm14\n\t" // [c24,c24,c35,c35]
	                //"vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c25,c25,c34,c34]
	                "\n\t"
	                "addq  $16*8 , %[a]\n\t"
	                "addq  $24*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          //}
	        }
	        if( K & 2 ){

	            // 4x6x4x2  LD+PM/FMA = 10/24 = 0.42
	            __asm__ __volatile__ (
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovupd   0*8(%[b]), %%ymm2 \n\t" // [b00,b10,b01,b11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b01,b11,b00,b10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm5 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm6 \n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c21,c21,c30,c30]
	                "\n\t"
	                "vmovupd   4*8(%[b]), %%ymm2 \n\t" // [b02,b12,b03,b13]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm10\n\t" // [c22,c22,c33,c33]
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b03,b13,b02,b12]
	                "vmovupd   8*8(%[b]), %%ymm2 \n\t" // [b04,b14,b05,b15]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c23,c23,c32,c32]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b05,b15,b04,b14]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c04,c15,c15]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm14\n\t" // [c24,c24,c35,c35]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm13\n\t" // [c05,c05,c14,c14]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c25,c25,c34,c34]

	                //"\n\t"
	                //"vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                //"vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                //"\n\t"
	                //"vmovupd   0*8(%[b]), %%ymm2 \n\t" // [b00,b10,b01,b11]
	                //"vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b01,b11,b00,b10]
	                //"vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c00,c11,c11]
	                //"vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm5 \n\t" // [c01,c01,c10,c10]
	                //"vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm6 \n\t" // [c20,c20,c31,c31]
	                //"vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c21,c21,c30,c30]
	                //"\n\t"
	                //"vmovupd   4*8(%[b]), %%ymm2 \n\t" // [b02,b12,b03,b13]
	                //"vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b03,b13,b02,b12]
	                //"vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c02,c13,c13]
	                //"vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c03,c03,c12,c12]
	                //"vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm10\n\t" // [c22,c22,c33,c33]
	                //"vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c23,c23,c32,c32]
	                //"\n\t"
	                //"vmovupd   8*8(%[b]), %%ymm2 \n\t" // [b04,b14,b05,b15]
	                //"vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b05,b15,b04,b14]
	                //"vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c04,c15,c15]
	                //"vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm13\n\t" // [c05,c05,c14,c14]
	                //"vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm14\n\t" // [c24,c24,c35,c35]
	                //"vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c25,c25,c34,c34]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $12*8, %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){

	            // 4x6x4x2  LD+PM/FMA = 10/24 = 0.42
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd    0*8(%[a]), %%xmm0 \n\t" // [a00,  0]
	                "vmovsd    1*8(%[a]), %%xmm1 \n\t" // [a01,  0]
	                "vmovsd    2*8(%[a]), %%xmm2 \n\t" // [a02,  0]
	                "vmovsd    3*8(%[a]), %%xmm3 \n\t" // [a03,  0]
	                "vperm2f128 $0x20 , %%ymm1 , %%ymm0 , %%ymm0 \n\t" // [a00,  0,a01,  0]
	                "vperm2f128 $0x20 , %%ymm3 , %%ymm2 , %%ymm1 \n\t" // [a02,  0,a03,  0]
	                "\n\t"
	                "vmovsd    0*8(%[b]), %%xmm2 \n\t" // [b00,  0]
	                "vmovsd    1*8(%[b]), %%xmm3 \n\t" // [b01,  0]
	                "vperm2f128 $0x20 , %%ymm3 , %%ymm2 , %%ymm2 \n\t" // [b00,  0,b01,  0]
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b01,  0,b00,  0]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm5 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm6 \n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c21,c21,c30,c30]
	                "\n\t"
	                "vmovsd    2*8(%[b]), %%xmm2 \n\t" // [b02,  0]
	                "vmovsd    3*8(%[b]), %%xmm3 \n\t" // [b03,  0]
	                "vperm2f128 $0x20 , %%ymm3 , %%ymm2 , %%ymm2 \n\t" // [b02,  0,b03,  0]
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b03,  0,b02,  0]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm10\n\t" // [c22,c22,c33,c33]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c23,c23,c32,c32]
	                "\n\t"
	                "vmovsd    4*8(%[b]), %%xmm2 \n\t" // [b04,  0]
	                "vmovsd    5*8(%[b]), %%xmm3 \n\t" // [b05,  0]
	                "vperm2f128 $0x20 , %%ymm3 , %%ymm2 , %%ymm2 \n\t" // [b04,  0,b05,  0]
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [b05,  0,b04,  0]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c04,c15,c15]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm13\n\t" // [c05,c05,c14,c14]
	                "vfmadd231pd  %%ymm1 , %%ymm2 , %%ymm14\n\t" // [c24,c24,c35,c35]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c25,c25,c34,c34]
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $6*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        __asm__ __volatile__ (
	            "\n\t"
	            // %%ymm4  [c00,c00,c11,c11]
	            // %%ymm5  [c01,c01,c10,c10]
	            // %%ymm6  [c20,c20,c31,c31]
	            // %%ymm7  [c21,c21,c30,c30]
	            // %%ymm8  [c02,c02,c13,c13]
	            // %%ymm9  [c03,c03,c12,c12]
	            // %%ymm10 [c22,c22,c33,c33]
	            // %%ymm11 [c23,c23,c32,c32]
	            // %%ymm12 [c04,c04,c15,c15]
	            // %%ymm13 [c05,c05,c14,c14]
	            // %%ymm14 [c24,c24,c35,c35]
	            // %%ymm15 [c25,c25,c34,c34]
	            "\n\t"
	            "vperm2f128 $0x20 , %%ymm6 , %%ymm4 , %%ymm0 \n\t" // [c00,c00,c20,c20]
	            "vperm2f128 $0x31 , %%ymm7 , %%ymm5 , %%ymm1 \n\t" // [c10,c10,c30,c30]
	            "vperm2f128 $0x20 , %%ymm7 , %%ymm5 , %%ymm2 \n\t" // [c01,c01,c21,c21]
	            "vperm2f128 $0x31 , %%ymm6 , %%ymm4 , %%ymm3 \n\t" // [c11,c11,c31,c31]
	            "vmovapd %[alpha], %%ymm6\n\t"
	            "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	            "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm0 \n\t" // [c00,c10,c20,c30]
	            "vshufpd    $0x00 , %%ymm3 , %%ymm2 , %%ymm5 \n\t" // [c01,c11,c21,c31]
	            "vshufpd    $0x0f , %%ymm3 , %%ymm2 , %%ymm2 \n\t" // [c01,c11,c21,c31]
	            "vaddpd             %%ymm0 , %%ymm4 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	            "vaddpd             %%ymm2 , %%ymm5 , %%ymm5 \n\t" // [c01,c11,c21,c31]
	            "vfmadd213pd 0*8(%[c0]        ), %%ymm6, %%ymm4 \n\t"
	            "vfmadd213pd 0*8(%[c1]        ), %%ymm6, %%ymm5 \n\t"
	            "vmovupd  %%ymm4 , 0*8(%[c0]        )\n\t"
	            "vmovupd  %%ymm5 , 0*8(%[c1]        )\n\t"
	            "\n\t"
	            "vperm2f128 $0x20 , %%ymm10, %%ymm8 , %%ymm0 \n\t" // [c02,c02,c22,c22]
	            "vperm2f128 $0x31 , %%ymm11, %%ymm9 , %%ymm1 \n\t" // [c12,c12,c32,c32]
	            "vperm2f128 $0x20 , %%ymm11, %%ymm9 , %%ymm2 \n\t" // [c03,c03,c23,c23]
	            "vperm2f128 $0x31 , %%ymm10, %%ymm8 , %%ymm3 \n\t" // [c13,c13,c33,c33]
	            "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	            "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm10\n\t" // [c02,c12,c22,c32]
	            "vshufpd    $0x00 , %%ymm3 , %%ymm2 , %%ymm9 \n\t" // [c03,c13,c23,c33]
	            "vshufpd    $0x0f , %%ymm3 , %%ymm2 , %%ymm11\n\t" // [c03,c13,c23,c33]
	            "vaddpd             %%ymm10, %%ymm8 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	            "vaddpd             %%ymm11, %%ymm9 , %%ymm9 \n\t" // [c03,c13,c23,c33]
	            "vfmadd213pd 0*8(%[c0],%[ldc2]), %%ymm6, %%ymm8 \n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2]), %%ymm6, %%ymm9 \n\t"
	            "vmovupd  %%ymm8 , 0*8(%[c0],%[ldc2])\n\t"
	            "vmovupd  %%ymm9 , 0*8(%[c1],%[ldc2])\n\t"
	            "\n\t"
	            "vperm2f128 $0x20 , %%ymm14, %%ymm12, %%ymm0 \n\t" // [c04,c04,c24,c24]
	            "vperm2f128 $0x31 , %%ymm15, %%ymm13, %%ymm1 \n\t" // [c14,c14,c34,c34]
	            "vperm2f128 $0x20 , %%ymm15, %%ymm13, %%ymm2 \n\t" // [c05,c05,c25,c25]
	            "vperm2f128 $0x31 , %%ymm14, %%ymm12, %%ymm3 \n\t" // [c15,c15,c35,c35]
	            "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm12\n\t" // [c04,c14,c24,c34]
	            "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm14\n\t" // [c04,c14,c24,c34]
	            "vshufpd    $0x00 , %%ymm3 , %%ymm2 , %%ymm13\n\t" // [c05,c15,c25,c35]
	            "vshufpd    $0x0f , %%ymm3 , %%ymm2 , %%ymm15\n\t" // [c05,c15,c25,c35]
	            "vaddpd             %%ymm14, %%ymm12, %%ymm12\n\t" // [c04,c14,c24,c34]
	            "vaddpd             %%ymm15, %%ymm13, %%ymm13\n\t" // [c05,c15,c25,c35]
	            "vfmadd213pd 0*8(%[c0],%[ldc2],2), %%ymm6, %%ymm12\n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2],2), %%ymm6, %%ymm13\n\t"
	            "vmovupd  %%ymm12, 0*8(%[c0],%[ldc2],2)\n\t"
	            "vmovupd  %%ymm13, 0*8(%[c1],%[ldc2],2)\n\t"
	            "\n\t"
	            "\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "addq  $4*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 6*K;

	      }
	    }
	    if( M & 2 ){

	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm10, %%ymm10, %%ymm10\n\t"
	            "vpxor  %%ymm11, %%ymm11, %%ymm11\n\t"
	            "vpxor  %%ymm12, %%ymm12, %%ymm12\n\t"
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);

	        __asm__ __volatile__ (
	            "prefetcht0 0*8(%[c0]          )\n\t"
	            "prefetcht0 0*8(%[c1]          )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0 0*8(%[c1],%[ldc2]  )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc2],2)\n\t"
	            "prefetcht0 0*8(%[c1],%[ldc2],2)\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );

	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            // 2x6x4x2  LD+PM/FMA = 8/12 = 0.66
	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0  48*8(%[a])\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "prefetcht0 144*8(%[b])\n\t"
	                "prefetcht0 152*8(%[b])\n\t"
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b02,b12,b03,b13]
	                "vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm11\n\t" // [c10,c10,c01,c01]
	                "vmovupd   8*8(%[b]), %%ymm6 \n\t" // [b04,b14,b05,b15]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm12\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm13\n\t" // [c12,c12,c03,c03]
	                "vmovupd   4*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c04,c04,c15,c15]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c14,c14,c05,c05]
	                "\n\t"
	                "vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b20,b30,b21,b31]
	                "vmovupd  16*8(%[b]), %%ymm8 \n\t" // [b22,b32,b23,b33]
	                "prefetcht0 160*8(%[b])\n\t"
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [a21,a31,a20,a30]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm10\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm11\n\t" // [c10,c10,c01,c01]
	                "vmovupd  20*8(%[b]), %%ymm9 \n\t" // [b24,b34,b25,b35]
	                "vfmadd231pd  %%ymm2 , %%ymm8 , %%ymm12\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm3 , %%ymm8 , %%ymm13\n\t" // [c12,c12,c03,c03]
	                "vmovapd   8*8(%[a]), %%ymm0 \n\t" // [a40,a50,a41,a51]
	                "prefetcht0  56*8(%[a])\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm9 , %%ymm14\n\t" // [c04,c04,c15,c15]
	                "vfmadd231pd  %%ymm3 , %%ymm9 , %%ymm15\n\t" // [c14,c14,c05,c05]
	                "\n\t"
	                "vmovupd  24*8(%[b]), %%ymm4 \n\t" // [b40,b50,b41,b51]
	                "vmovupd  28*8(%[b]), %%ymm5 \n\t" // [b42,b52,b43,b53]
	                "prefetcht0 168*8(%[b])\n\t"
	                "vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a41,a51,a40,a50]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm11\n\t" // [c10,c10,c01,c01]
	                "vmovupd  32*8(%[b]), %%ymm6 \n\t" // [b44,b54,b45,b55]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm12\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm13\n\t" // [c12,c12,c03,c03]
	                "vmovupd  12*8(%[a]), %%ymm2 \n\t" // [a60,a70,a61,a71]
	                "prefetcht0 176*8(%[b])\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c04,c04,c15,c15]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c14,c14,c05,c05]
	                "\n\t"
	                "vmovupd  36*8(%[b]), %%ymm7 \n\t" // [b60,b70,b61,b71]
	                "vmovupd  40*8(%[b]), %%ymm8 \n\t" // [b62,b72,b63,b73]
	                "prefetcht0 184*8(%[b])\n\t"
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [a61,a71,a60,a70]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm10\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm11\n\t" // [c10,c10,c01,c01]
	                "vmovupd  44*8(%[b]), %%ymm9 \n\t" // [b64,b74,b65,b75]
	                "vfmadd231pd  %%ymm2 , %%ymm8 , %%ymm12\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm3 , %%ymm8 , %%ymm13\n\t" // [c12,c12,c03,c03]
	                "vfmadd231pd  %%ymm2 , %%ymm9 , %%ymm14\n\t" // [c04,c04,c15,c15]
	                "vfmadd231pd  %%ymm3 , %%ymm9 , %%ymm15\n\t" // [c14,c14,c05,c05]
	                "\n\t"
	                //"prefetcht0  48*8(%[a])\n\t"
	                //"prefetcht0  56*8(%[a])\n\t"
	                //"prefetcht0 144*8(%[b])\n\t"
	                //"prefetcht0 152*8(%[b])\n\t"
	                //"prefetcht0 160*8(%[b])\n\t"
	                //"prefetcht0 168*8(%[b])\n\t"
	                //"prefetcht0 176*8(%[b])\n\t"
	                //"prefetcht0 184*8(%[b])\n\t"
	                //"\n\t"
	                //"vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                //"vmovupd   4*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                //"vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                //"vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b02,b12,b03,b13]
	                //"vmovupd   8*8(%[b]), %%ymm6 \n\t" // [b04,b14,b05,b15]
	                //"vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b20,b30,b21,b31]
	                //"vmovupd  16*8(%[b]), %%ymm8 \n\t" // [b22,b32,b23,b33]
	                //"vmovupd  20*8(%[b]), %%ymm9 \n\t" // [b24,b34,b25,b35]
	                //"vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                //"vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [a21,a31,a20,a30]
	                //"vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c00,c11,c11]
	                //"vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm11\n\t" // [c10,c10,c01,c01]
	                //"vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm12\n\t" // [c02,c02,c13,c13]
	                //"vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm13\n\t" // [c12,c12,c03,c03]
	                //"vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c04,c04,c15,c15]
	                //"vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c14,c14,c05,c05]
	                //"vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm10\n\t" // [c00,c00,c11,c11]
	                //"vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm11\n\t" // [c10,c10,c01,c01]
	                //"vfmadd231pd  %%ymm2 , %%ymm8 , %%ymm12\n\t" // [c02,c02,c13,c13]
	                //"vfmadd231pd  %%ymm3 , %%ymm8 , %%ymm13\n\t" // [c12,c12,c03,c03]
	                //"vfmadd231pd  %%ymm2 , %%ymm9 , %%ymm14\n\t" // [c04,c04,c15,c15]
	                //"vfmadd231pd  %%ymm3 , %%ymm9 , %%ymm15\n\t" // [c14,c14,c05,c05]
	                //"\n\t"
	                //"vmovapd   8*8(%[a]), %%ymm0 \n\t" // [a40,a50,a41,a51]
	                //"vmovupd  12*8(%[a]), %%ymm2 \n\t" // [a60,a70,a61,a71]
	                //"vmovupd  24*8(%[b]), %%ymm4 \n\t" // [b40,b50,b41,b51]
	                //"vmovupd  28*8(%[b]), %%ymm5 \n\t" // [b42,b52,b43,b53]
	                //"vmovupd  32*8(%[b]), %%ymm6 \n\t" // [b44,b54,b45,b55]
	                //"vmovupd  36*8(%[b]), %%ymm7 \n\t" // [b60,b70,b61,b71]
	                //"vmovupd  40*8(%[b]), %%ymm8 \n\t" // [b62,b72,b63,b73]
	                //"vmovupd  44*8(%[b]), %%ymm9 \n\t" // [b64,b74,b65,b75]
	                //"vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a41,a51,a40,a50]
	                //"vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [a61,a71,a60,a70]
	                //"vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c00,c11,c11]
	                //"vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm11\n\t" // [c10,c10,c01,c01]
	                //"vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm12\n\t" // [c02,c02,c13,c13]
	                //"vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm13\n\t" // [c12,c12,c03,c03]
	                //"vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c04,c04,c15,c15]
	                //"vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c14,c14,c05,c05]
	                //"vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm10\n\t" // [c00,c00,c11,c11]
	                //"vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm11\n\t" // [c10,c10,c01,c01]
	                //"vfmadd231pd  %%ymm2 , %%ymm8 , %%ymm12\n\t" // [c02,c02,c13,c13]
	                //"vfmadd231pd  %%ymm3 , %%ymm8 , %%ymm13\n\t" // [c12,c12,c03,c03]
	                //"vfmadd231pd  %%ymm2 , %%ymm9 , %%ymm14\n\t" // [c04,c04,c15,c15]
	                //"vfmadd231pd  %%ymm3 , %%ymm9 , %%ymm15\n\t" // [c14,c14,c05,c05]
	                "\n\t"
	                "addq  $16*8 , %[a]\n\t"
	                "addq  $48*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          }

	        }
	        if( K & 4 ){
	        //if( K >> 2 ){
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){

	            // 2x6x4x2  LD+PM/FMA = 8/12 = 0.66
	            __asm__ __volatile__ (
	                "\n\t"
	                //"vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                //"vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                //"vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b02,b12,b03,b13]
	                //"vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                //"vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c00,c11,c11]
	                //"vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm11\n\t" // [c10,c10,c01,c01]
	                //"vmovupd   8*8(%[b]), %%ymm6 \n\t" // [b04,b14,b05,b15]
	                //"vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm12\n\t" // [c02,c02,c13,c13]
	                //"vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm13\n\t" // [c12,c12,c03,c03]
	                //"vmovupd   4*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                //"vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c04,c04,c15,c15]
	                //"vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c14,c14,c05,c05]
	                //"\n\t"
	                //"vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b20,b30,b21,b31]
	                //"vmovupd  16*8(%[b]), %%ymm8 \n\t" // [b22,b32,b23,b33]
	                //"vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [a21,a31,a20,a30]
	                //"vmovupd  20*8(%[b]), %%ymm9 \n\t" // [b24,b34,b25,b35]
	                //"vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm10\n\t" // [c00,c00,c11,c11]
	                //"vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm11\n\t" // [c10,c10,c01,c01]
	                //"vfmadd231pd  %%ymm2 , %%ymm8 , %%ymm12\n\t" // [c02,c02,c13,c13]
	                //"vfmadd231pd  %%ymm3 , %%ymm8 , %%ymm13\n\t" // [c12,c12,c03,c03]
	                //"vfmadd231pd  %%ymm2 , %%ymm9 , %%ymm14\n\t" // [c04,c04,c15,c15]
	                //"vfmadd231pd  %%ymm3 , %%ymm9 , %%ymm15\n\t" // [c14,c14,c05,c05]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovupd   4*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b02,b12,b03,b13]
	                "vmovupd   8*8(%[b]), %%ymm6 \n\t" // [b04,b14,b05,b15]
	                "vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b20,b30,b21,b31]
	                "vmovupd  16*8(%[b]), %%ymm8 \n\t" // [b22,b32,b23,b33]
	                "vmovupd  20*8(%[b]), %%ymm9 \n\t" // [b24,b34,b25,b35]
	                "vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [a21,a31,a20,a30]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm11\n\t" // [c10,c10,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm12\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm13\n\t" // [c12,c12,c03,c03]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c04,c04,c15,c15]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c14,c14,c05,c05]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm10\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm11\n\t" // [c10,c10,c01,c01]
	                "vfmadd231pd  %%ymm2 , %%ymm8 , %%ymm12\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm3 , %%ymm8 , %%ymm13\n\t" // [c12,c12,c03,c03]
	                "vfmadd231pd  %%ymm2 , %%ymm9 , %%ymm14\n\t" // [c04,c04,c15,c15]
	                "vfmadd231pd  %%ymm3 , %%ymm9 , %%ymm15\n\t" // [c14,c14,c05,c05]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $24*8, %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          //}
	        }
	        if( K & 2 ){
	            
	            __asm__ __volatile__ (
	                "\n\t"
	                //"vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                //"vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                //"vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b02,b12,b03,b13]
	                //"vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                //"vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c00,c11,c11]
	                //"vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm11\n\t" // [c10,c10,c01,c01]
	                //"vmovupd   8*8(%[b]), %%ymm6 \n\t" // [b04,b14,b05,b15]
	                //"vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm12\n\t" // [c02,c02,c13,c13]
	                //"vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm13\n\t" // [c12,c12,c03,c03]
	                //"vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c04,c04,c15,c15]
	                //"vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c14,c14,c05,c05]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b02,b12,b03,b13]
	                "vmovupd   8*8(%[b]), %%ymm6 \n\t" // [b04,b14,b05,b15]
	                "vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm11\n\t" // [c10,c10,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm12\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm13\n\t" // [c12,c12,c03,c03]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c04,c04,c15,c15]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c14,c14,c05,c05]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $12*8, %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){
	            
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd    0*8(%[a]), %%xmm0 \n\t" // [a00,  0]
	                "vmovsd    1*8(%[a]), %%xmm1 \n\t" // [a01,  0]
	                "vmovsd    0*8(%[b]), %%xmm4 \n\t" // [b00,  0]
	                "vmovsd    1*8(%[b]), %%xmm5 \n\t" // [b01,  0]
	                "vmovsd    2*8(%[b]), %%xmm6 \n\t" // [b02,  0]
	                "vmovsd    3*8(%[b]), %%xmm7 \n\t" // [b03,  0]
	                "vmovsd    4*8(%[b]), %%xmm8 \n\t" // [b04,  0]
	                "vmovsd    5*8(%[b]), %%xmm9 \n\t" // [b05,  0]
	                "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm0 \n\t" // [a00,  0,a01,  0]
	                "vperm2f128  $0x20, %%ymm5 , %%ymm4 , %%ymm4 \n\t" // [b00,  0,b01,  0]
	                "vperm2f128  $0x20, %%ymm7 , %%ymm6 , %%ymm6 \n\t" // [b02,  0,b03,  0]
	                "vperm2f128  $0x20, %%ymm9 , %%ymm8 , %%ymm8 \n\t" // [b04,  0,b05,  0]
	                "vperm2f128  $0x01, %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,  0,a00,  0]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm11\n\t" // [c10,c10,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm12\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm13\n\t" // [c12,c12,c03,c03]
	                "vfmadd231pd  %%ymm0 , %%ymm8 , %%ymm14\n\t" // [c04,c04,c15,c15]
	                "vfmadd231pd  %%ymm1 , %%ymm8 , %%ymm15\n\t" // [c14,c14,c05,c05]
	                "\n\t"
	                "addq  $2*8, %[a]\n\t"
	                "addq  $6*8, %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	            
	        __asm__ __volatile__ (
	            "\n\t"
	            // %%ymm10 [c00,c00,c11,c11]
	            // %%ymm11 [c10,c10,c01,c01]
	            // %%ymm12 [c02,c02,c13,c13]
	            // %%ymm13 [c12,c12,c03,c03]
	            // %%ymm14 [c04,c04,c15,c15]
	            // %%ymm15 [c14,c14,c05,c05]
	            "\n\t"
	            "movupd %[alpha], %%xmm0\n\t"
	            "\n\t"
	            "vshufpd    $0x00 , %%ymm11, %%ymm10, %%ymm4 \n\t" // [c00,c10,c11,c01]
	            "vshufpd    $0x0f , %%ymm11, %%ymm10, %%ymm5 \n\t" // [c00,c10,c11,c01]
	            "vshufpd    $0x00 , %%ymm13, %%ymm12, %%ymm6 \n\t" // [c02,c12,c13,c03]
	            "vshufpd    $0x0f , %%ymm13, %%ymm12, %%ymm7 \n\t" // [c02,c12,c13,c03]
	            "vshufpd    $0x00 , %%ymm15, %%ymm14, %%ymm8 \n\t" // [c04,c14,c15,c05]
	            "vshufpd    $0x0f , %%ymm15, %%ymm14, %%ymm9 \n\t" // [c04,c14,c15,c05]
	            "\n\t"
	            "vaddpd             %%ymm5 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c11,c01]
	            "vaddpd             %%ymm7 , %%ymm6 , %%ymm12\n\t" // [c02,c12,c13,c03]
	            "vaddpd             %%ymm9 , %%ymm8 , %%ymm14\n\t" // [c04,c14,c15,c05]
	            "\n\t"
	            "vperm2f128 $0x01 , %%ymm10, %%ymm10, %%ymm11\n\t" // [c11,c01,c00,c10]
	            "vperm2f128 $0x01 , %%ymm12, %%ymm12, %%ymm13\n\t" // [c13,c03,c02,c12]
	            "vperm2f128 $0x01 , %%ymm14, %%ymm14, %%ymm15\n\t" // [c15,c05,c04,c14]
	            "vshufpd    $0x09 , %%ymm11, %%ymm11, %%ymm11\n\t" // [c01,c11,c00,c10]
	            "vshufpd    $0x09 , %%ymm13, %%ymm13, %%ymm13\n\t" // [c03,c13,c02,c12]
	            "vshufpd    $0x09 , %%ymm15, %%ymm15, %%ymm15\n\t" // [c05,c15,c04,c14]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]        ), %%xmm0, %%xmm10\n\t" // [c00,c10,---,---]
	            "vfmadd213pd 0*8(%[c1]        ), %%xmm0, %%xmm11\n\t" // [c01,c11,---,---]
	            "vfmadd213pd 0*8(%[c0],%[ldc2]), %%xmm0, %%xmm12\n\t" // [c02,c12,---,---]
	            "vfmadd213pd 0*8(%[c1],%[ldc2]), %%xmm0, %%xmm13\n\t" // [c03,c13,---,---]
	            "vfmadd213pd 0*8(%[c0],%[ldc2],2), %%xmm0, %%xmm14\n\t" // [c04,c14,---,---]
	            "vfmadd213pd 0*8(%[c1],%[ldc2],2), %%xmm0, %%xmm15\n\t" // [c05,c15,---,---]
	            "\n\t"
	            "movupd  %%xmm10 , 0*8(%[c0]          )\n\t"
	            "movupd  %%xmm11 , 0*8(%[c1]          )\n\t"
	            "movupd  %%xmm12 , 0*8(%[c0],%[ldc2]  )\n\t"
	            "movupd  %%xmm13 , 0*8(%[c1],%[ldc2]  )\n\t"
	            "movupd  %%xmm14 , 0*8(%[c0],%[ldc2],2)\n\t"
	            "movupd  %%xmm15 , 0*8(%[c1],%[ldc2],2)\n\t"
	            "\n\t"
	            "addq  $2*8, %[c0]\n\t"
	            "addq  $2*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 6*K;

	    }
	    if( M & 1 ){
	            
	        __asm__ __volatile__ (
	            "\n\t"
	            //"vpxor  %%ymm8 , %%ymm8 , %%ymm8 \n\t"
	            //"vpxor  %%ymm9 , %%ymm9 , %%ymm9 \n\t"
	            "vpxor  %%ymm10, %%ymm10, %%ymm10\n\t"
	            "vpxor  %%ymm11, %%ymm11, %%ymm11\n\t"
	            "vpxor  %%ymm12, %%ymm12, %%ymm12\n\t"
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);

	        __asm__ __volatile__ (
	            "prefetcht0 0*8(%[c0]          )\n\t"
	            "prefetcht0 0*8(%[c1]          )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0 0*8(%[c1],%[ldc2]  )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc2],2)\n\t"
	            "prefetcht0 0*8(%[c1],%[ldc2],2)\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            // 1x6x4x2  LD+PM/FMA = 7/6 = 1.17
	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  24*8(%[a])\n\t"
	                "vmovupd   0*8(%[a]), %%ymm10\n\t" // [a00,a10,a20,a30]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                //"prefetcht0 144*8(%[b])\n\t"
	                //"prefetcht0 152*8(%[b])\n\t"
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b02,b12,b03,b13]
	                "vperm2f128 $0x00 , %%ymm10, %%ymm10, %%ymm0 \n\t" // [a00,a10,a00,a10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm13\n\t" // [c00,c00,c01,c01]
	                "vmovupd   8*8(%[b]), %%ymm6 \n\t" // [b04,b14,b05,b15]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm14\n\t" // [c02,c02,c03,c03]
	                "vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b20,b30,b21,b31]
	                //"prefetcht0 160*8(%[b])\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm15\n\t" // [c04,c04,c05,c05]
	                "vmovupd  16*8(%[b]), %%ymm8 \n\t" // [b22,b32,b23,b33]
	                "vperm2f128 $0x11 , %%ymm10, %%ymm10, %%ymm1 \n\t" // [a20,a30,a20,a30]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm13\n\t" // [c00,c00,c01,c01]
	                "vmovupd  20*8(%[b]), %%ymm9 \n\t" // [b24,b34,b25,b35]
	                "vfmadd231pd  %%ymm1 , %%ymm8 , %%ymm14\n\t" // [c02,c02,c03,c03]
	                "vmovupd  24*8(%[b]), %%ymm4 \n\t" // [b40,b50,b41,b51]
	                //"prefetcht0 168*8(%[b])\n\t"
	                "vfmadd231pd  %%ymm1 , %%ymm9 , %%ymm15\n\t" // [c04,c04,c05,c05]
	                "\n\t"
	                "vmovupd   4*8(%[a]), %%ymm11\n\t" // [a40,a50,a60,a70]
	                "vperm2f128 $0x00 , %%ymm11, %%ymm11, %%ymm2 \n\t" // [a40,a50,a40,a50]
	                "vmovupd  28*8(%[b]), %%ymm5 \n\t" // [b42,b52,b43,b53]
	                //"prefetcht0 176*8(%[b])\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm13\n\t" // [c00,c00,c01,c01]
	                "vmovupd  32*8(%[b]), %%ymm6 \n\t" // [b44,b54,b45,b55]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm14\n\t" // [c02,c02,c03,c03]
	                "vmovupd  36*8(%[b]), %%ymm7 \n\t" // [b60,b70,b61,b71]
	                //"prefetcht0 184*8(%[b])\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm15\n\t" // [c04,c04,c05,c05]
	                "vperm2f128 $0x11 , %%ymm11, %%ymm11, %%ymm3 \n\t" // [a60,a70,a60,a70]
	                "vmovupd  40*8(%[b]), %%ymm8 \n\t" // [b62,b72,b63,b73]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm13\n\t" // [c00,c00,c01,c01]
	                "vmovupd  44*8(%[b]), %%ymm9 \n\t" // [b64,b74,b65,b75]
	                "vfmadd231pd  %%ymm3 , %%ymm8 , %%ymm14\n\t" // [c02,c02,c03,c03]
	                "vfmadd231pd  %%ymm3 , %%ymm9 , %%ymm15\n\t" // [c04,c04,c05,c05]
	                "\n\t"
	                "\n\t"
	                //"prefetcht0  24*8(%[a])\n\t"
	                //"prefetcht0 144*8(%[b])\n\t"
	                //"prefetcht0 152*8(%[b])\n\t"
	                //"prefetcht0 160*8(%[b])\n\t"
	                //"prefetcht0 168*8(%[b])\n\t"
	                //"prefetcht0 176*8(%[b])\n\t"
	                //"prefetcht0 184*8(%[b])\n\t"
	                //"\n\t"
	                //"vmovupd   0*8(%[a]), %%ymm10\n\t" // [a00,a10,a20,a30]
	                //"vperm2f128 $0x11 , %%ymm10, %%ymm10, %%ymm1 \n\t" // [a20,a30,a20,a30]
	                //"vperm2f128 $0x00 , %%ymm10, %%ymm10, %%ymm0 \n\t" // [a00,a10,a00,a10]
	                //"vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                //"vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b02,b12,b03,b13]
	                //"vmovupd   8*8(%[b]), %%ymm6 \n\t" // [b04,b14,b05,b15]
	                //"vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b20,b30,b21,b31]
	                //"vmovupd  16*8(%[b]), %%ymm8 \n\t" // [b22,b32,b23,b33]
	                //"vmovupd  20*8(%[b]), %%ymm9 \n\t" // [b24,b34,b25,b35]
	                //"vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm13\n\t" // [c00,c00,c01,c01]
	                //"vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm14\n\t" // [c02,c02,c03,c03]
	                //"vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm15\n\t" // [c04,c04,c05,c05]
	                //"vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm13\n\t" // [c00,c00,c01,c01]
	                //"vfmadd231pd  %%ymm1 , %%ymm8 , %%ymm14\n\t" // [c02,c02,c03,c03]
	                //"vfmadd231pd  %%ymm1 , %%ymm9 , %%ymm15\n\t" // [c04,c04,c05,c05]
	                //"\n\t"
	                //"\n\t"
	                //"vmovupd   4*8(%[a]), %%ymm11\n\t" // [a40,a50,a60,a70]
	                //"vperm2f128 $0x11 , %%ymm11, %%ymm11, %%ymm3 \n\t" // [a60,a70,a60,a70]
	                //"vperm2f128 $0x00 , %%ymm11, %%ymm11, %%ymm2 \n\t" // [a40,a50,a40,a50]
	                //"vmovupd  24*8(%[b]), %%ymm4 \n\t" // [b40,b50,b41,b51]
	                //"vmovupd  28*8(%[b]), %%ymm5 \n\t" // [b42,b52,b43,b53]
	                //"vmovupd  32*8(%[b]), %%ymm6 \n\t" // [b44,b54,b45,b55]
	                //"vmovupd  36*8(%[b]), %%ymm7 \n\t" // [b60,b70,b61,b71]
	                //"vmovupd  40*8(%[b]), %%ymm8 \n\t" // [b62,b72,b63,b73]
	                //"vmovupd  44*8(%[b]), %%ymm9 \n\t" // [b64,b74,b65,b75]
	                //"vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm13\n\t" // [c00,c00,c01,c01]
	                //"vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm14\n\t" // [c02,c02,c03,c03]
	                //"vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm15\n\t" // [c04,c04,c05,c05]
	                //"vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm13\n\t" // [c00,c00,c01,c01]
	                //"vfmadd231pd  %%ymm3 , %%ymm8 , %%ymm14\n\t" // [c02,c02,c03,c03]
	                //"vfmadd231pd  %%ymm3 , %%ymm9 , %%ymm15\n\t" // [c04,c04,c05,c05]
	                "\n\t"
	                "\n\t"
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $48*8, %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          }
	        }
	        if( K & 4 ){
	        //if( K >> 2 ){
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){

	            // 1x6x4x2  LD+PM/FMA = 7/6 = 1.17
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovupd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vperm2f128 $0x11 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a20,a30,a20,a30]
	                "vperm2f128 $0x00 , %%ymm0 , %%ymm0 , %%ymm0 \n\t" // [a00,a10,a00,a10]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b02,b12,b03,b13]
	                "vmovupd   8*8(%[b]), %%ymm6 \n\t" // [b04,b14,b05,b15]
	                "vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b20,b30,b21,b31]
	                "vmovupd  16*8(%[b]), %%ymm8 \n\t" // [b22,b32,b23,b33]
	                "vmovupd  20*8(%[b]), %%ymm9 \n\t" // [b24,b34,b25,b35]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm13\n\t" // [c00,c00,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm14\n\t" // [c02,c02,c03,c03]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm15\n\t" // [c04,c04,c05,c05]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm13\n\t" // [c00,c00,c01,c01]
	                "vfmadd231pd  %%ymm1 , %%ymm8 , %%ymm14\n\t" // [c02,c02,c03,c03]
	                "vfmadd231pd  %%ymm1 , %%ymm9 , %%ymm15\n\t" // [c04,c04,c05,c05]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $24*8, %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          //}
	        }
	        if( K & 2 ){
	            
	            // 1x6x4x2  LD+PM/FMA = 7/6 = 1.17
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovupd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,---,---]
	                "vperm2f128 $0x00 , %%ymm0 , %%ymm0 , %%ymm0 \n\t" // [a00,a10,a00,a10]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b02,b12,b03,b13]
	                "vmovupd   8*8(%[b]), %%ymm6 \n\t" // [b04,b14,b05,b15]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm13\n\t" // [c00,c00,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm14\n\t" // [c02,c02,c03,c03]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm15\n\t" // [c04,c04,c05,c05]
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $12*8, %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){
	            
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd    0*8(%[a]), %%xmm0 \n\t" // [a00,  0,---,---]
	                "vperm2f128 $0x00 , %%ymm0 , %%ymm0 , %%ymm0 \n\t" // [a00,  0,a00,  0]
	                "vmovsd   0*8(%[b]), %%xmm4 \n\t" // [b00,  0,  0,  0]
	                "vmovsd   1*8(%[b]), %%xmm7 \n\t" // [b01,  0,  0,  0]
	                "vmovsd   2*8(%[b]), %%xmm5 \n\t" // [b02,  0,  0,  0]
	                "vmovsd   3*8(%[b]), %%xmm8 \n\t" // [b03,  0,  0,  0]
	                "vmovsd   4*8(%[b]), %%xmm6 \n\t" // [b04,  0,  0,  0]
	                "vmovsd   5*8(%[b]), %%xmm9 \n\t" // [b05,  0,  0,  0]
	                "vperm2f128 $0x00 , %%ymm7 , %%ymm4 , %%ymm4 \n\t" // [b00,  0,b01,  0]
	                "vperm2f128 $0x00 , %%ymm8 , %%ymm5 , %%ymm5 \n\t" // [b02,  0,b03,  0]
	                "vperm2f128 $0x00 , %%ymm9 , %%ymm6 , %%ymm6 \n\t" // [b04,  0,b05,  0]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm13\n\t" // [c00,c00,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm14\n\t" // [c02,c02,c03,c03]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm15\n\t" // [c04,c04,c05,c05]
	                "\n\t"
	                "addq  $1*8  , %[a]\n\t"
	                "addq  $6*8  , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }

	        __asm__ __volatile__ (
	            "\n\t"
	            // %%ymm13 [c00,c00,c01,c01]
	            // %%ymm14 [c02,c02,c03,c03]
	            // %%ymm15 [c04,c04,c05,c05]
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "movlpd   0*8(%[c0]          ), %%xmm1\n\t" // [c00,---,---,---]
	            "movhpd   0*8(%[c1]          ), %%xmm1\n\t" // [c00,c01,---,---]
	            "movlpd   0*8(%[c0],%[ldc2]  ), %%xmm2\n\t" // [c02,---,---,---]
	            "movhpd   0*8(%[c1],%[ldc2]  ), %%xmm2\n\t" // [c02,c03,---,---]
	            "movlpd   0*8(%[c0],%[ldc2],2), %%xmm3\n\t" // [c04,---,---,---]
	            "movhpd   0*8(%[c1],%[ldc2],2), %%xmm3\n\t" // [c04,c05,---,---]
	            "\n\t"
	            "vperm2f128 $0x01 , %%ymm13, %%ymm13, %%ymm10\n\t" // [c01,c01,c00,c00]
	            "vperm2f128 $0x01 , %%ymm14, %%ymm14, %%ymm11\n\t" // [c03,c03,c02,c02]
	            "vperm2f128 $0x01 , %%ymm15, %%ymm15, %%ymm12\n\t" // [c05,c05,c04,c04]
	            "\n\t"
	            "vshufpd    $0x00 , %%ymm10, %%ymm13, %%ymm4 \n\t" // [c00,c01,---,---]
	            "vshufpd    $0x0f , %%ymm10, %%ymm13, %%ymm7 \n\t" // [c00,c01,---,---]
	            "vshufpd    $0x00 , %%ymm11, %%ymm14, %%ymm5 \n\t" // [c02,c03,---,---]
	            "vshufpd    $0x0f , %%ymm11, %%ymm14, %%ymm8 \n\t" // [c02,c03,---,---]
	            "vshufpd    $0x00 , %%ymm12, %%ymm15, %%ymm6 \n\t" // [c04,c05,---,---]
	            "vshufpd    $0x0f , %%ymm12, %%ymm15, %%ymm9 \n\t" // [c04,c05,---,---]
	            "\n\t"
	            "vaddpd             %%ymm7 , %%ymm4 , %%ymm13\n\t" // [c00,c01,---,---]
	            "vaddpd             %%ymm8 , %%ymm5 , %%ymm14\n\t" // [c02,c03,---,---]
	            "vaddpd             %%ymm9 , %%ymm6 , %%ymm15\n\t" // [c04,c05,---,---]
	            "\n\t"
	            "vfmadd213pd  %%ymm1, %%ymm0, %%ymm13\n\t"
	            "vfmadd213pd  %%ymm2, %%ymm0, %%ymm14\n\t"
	            "vfmadd213pd  %%ymm3, %%ymm0, %%ymm15\n\t"
	            "\n\t"
	            "movlpd   %%xmm13, 0*8(%[c0]          )\n\t"
	            "movhpd   %%xmm13, 0*8(%[c1]          )\n\t"
	            "movlpd   %%xmm14, 0*8(%[c0],%[ldc2]  )\n\t"
	            "movhpd   %%xmm14, 0*8(%[c1],%[ldc2]  )\n\t"
	            "movlpd   %%xmm15, 0*8(%[c0],%[ldc2],2)\n\t"
	            "movhpd   %%xmm15, 0*8(%[c1],%[ldc2],2)\n\t"
	            "\n\t"
	            "addq  $1*8, %[c0]\n\t"
	            "addq  $1*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 6*K;

	    }

	    A = A - M*K;
	    B = B + 6*K;
	    c0 = c0- M + 6*ldc;
	    c1 = c1- M + 6*ldc;
	  }
	}

	if( NR & 4 ){
	  //size_t n4 = ( N >> 2 ); // unrolling N
	  //while( n4-- ){
	  //__asm__ __volatile__ (
	  //    "\n\t"
	  //    "prefetcht0 0*8(%[a])\n\t"
	  //    "prefetcht0 0*8(%[b])\n\t"
	  //    "prefetcht0 4*8(%[a])\n\t"
	  //    "prefetcht0 4*8(%[b])\n\t"
	  //    "\n\t"
	  //    :[a]"+r"(A),[b]"+r"(B)
	  //:);

	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 ); // unrolling M
	      while( m4-- ){

	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm8 , %%ymm8 , %%ymm8 \n\t"
	            "vpxor  %%ymm9 , %%ymm9 , %%ymm9 \n\t"
	            "vpxor  %%ymm10, %%ymm10, %%ymm10\n\t"
	            "vpxor  %%ymm11, %%ymm11, %%ymm11\n\t"
	            "vpxor  %%ymm12, %%ymm12, %%ymm12\n\t"
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);

	        __asm__ __volatile__ (
	            "prefetcht0 0*8(%[c0]        )\n\t"
	            "prefetcht0 0*8(%[c1]        )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc2])\n\t"
	            "prefetcht0 0*8(%[c1],%[ldc2])\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          k8--;
	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  96*8(%[a])\n\t"
	                //"prefetcht0  96*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovupd   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,b11,b00,b10]
	                "vperm2f128 $0x01 , %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b03,b13,b02,b12]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a22,a32,a23,a33]
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          while( k8-- ){

	            // 4x4x4x2  LD+PM/FMA = 6/8 = 0.75
	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0  96*8(%[a])\n\t"
	                "prefetcht0  96*8(%[b])\n\t"
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm12\n\t" // [c20,c20,c31,c31]
	                "vmovupd   8*8(%[b]), %%ymm4 \n\t" // [b20,b30,b21,b31]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm9 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm13\n\t" // [c21,c21,c30,c30]
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b21,b31,b20,b30]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm10\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm14\n\t" // [c22,c22,c33,c33]
	                "vmovupd  12*8(%[b]), %%ymm6 \n\t" // [b22,b32,b23,b33]
	                //"prefetcht0 104*8(%[a])\n\t"
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm7 , %%ymm11\n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm15\n\t" // [c23,c23,c32,c32]
	                "vperm2f128 $0x01 , %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b23,b33,b22,b32]
	                "\n\t"
	                "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a40,a50,a41,a51]
	                "vmovapd  20*8(%[a]), %%ymm1 \n\t" // [a42,a52,a43,a53]
	                "\n\t"
	                //"prefetcht0 104*8(%[b])\n\t"
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm4 , %%ymm12\n\t" // [c20,c20,c31,c31]
	                "vmovupd  16*8(%[b]), %%ymm4 \n\t" // [b40,b50,b41,b51]
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm9 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm13\n\t" // [c21,c21,c30,c30]
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,b11,b00,b10]
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm10\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm3 , %%ymm6 , %%ymm14\n\t" // [c22,c22,c33,c33]
	                "vmovupd  20*8(%[b]), %%ymm6 \n\t" // [b42,b52,b43,b53]
	                "prefetcht0 112*8(%[a])\n\t"
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm11\n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c23,c23,c32,c32]
	                "vperm2f128 $0x01 , %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b03,b13,b02,b12]
	                "\n\t"
	                "vmovapd  24*8(%[a]), %%ymm2 \n\t" // [a60,a70,a61,a71]
	                "vmovapd  28*8(%[a]), %%ymm3 \n\t" // [a62,a72,a63,a73]
	                "\n\t"
	                "prefetcht0 112*8(%[b])\n\t"
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm12\n\t" // [c20,c20,c31,c31]
	                "vmovupd  24*8(%[b]), %%ymm4 \n\t" // [b60,b70,b61,b71]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm9 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm13\n\t" // [c21,c21,c30,c30]
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b21,b31,b20,b30]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm10\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm14\n\t" // [c22,c22,c33,c33]
	                "vmovupd  28*8(%[b]), %%ymm6 \n\t" // [b62,b72,b63,b73]
	                //"prefetcht0 120*8(%[a])\n\t"
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm7 , %%ymm11\n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm15\n\t" // [c23,c23,c32,c32]
	                "vperm2f128 $0x01 , %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b23,b33,b22,b32]
	                "\n\t"
	                "vmovapd  32*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd  36*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "\n\t"
	                //"prefetcht0 120*8(%[b])\n\t"
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm4 , %%ymm12\n\t" // [c20,c20,c31,c31]
	                "vmovupd  32*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm9 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm13\n\t" // [c21,c21,c30,c30]
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,b11,b00,b10]
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm10\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm3 , %%ymm6 , %%ymm14\n\t" // [c22,c22,c33,c33]
	                "vmovupd  36*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm11\n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c23,c23,c32,c32]
	                "vperm2f128 $0x01 , %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b03,b13,b02,b12]
	                "\n\t"
	                "vmovapd  40*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  44*8(%[a]), %%ymm3 \n\t" // [a22,a32,a23,a33]
	                "\n\t"
	                "addq  $32*8 , %[a]\n\t"
	                "addq  $32*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          }
	            __asm__ __volatile__ (
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm12\n\t" // [c20,c20,c31,c31]
	                "vmovupd   8*8(%[b]), %%ymm4 \n\t" // [b20,b30,b21,b31]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm9 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm13\n\t" // [c21,c21,c30,c30]
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b21,b31,b20,b30]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm10\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm14\n\t" // [c22,c22,c33,c33]
	                "vmovupd  12*8(%[b]), %%ymm6 \n\t" // [b22,b32,b23,b33]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm7 , %%ymm11\n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm15\n\t" // [c23,c23,c32,c32]
	                "vperm2f128 $0x01 , %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b23,b33,b22,b32]
	                "\n\t"
	                "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a40,a50,a41,a51]
	                "vmovapd  20*8(%[a]), %%ymm1 \n\t" // [a42,a52,a43,a53]
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm4 , %%ymm12\n\t" // [c20,c20,c31,c31]
	                "vmovupd  16*8(%[b]), %%ymm4 \n\t" // [b40,b50,b41,b51]
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm9 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm13\n\t" // [c21,c21,c30,c30]
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,b11,b00,b10]
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm10\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm3 , %%ymm6 , %%ymm14\n\t" // [c22,c22,c33,c33]
	                "vmovupd  20*8(%[b]), %%ymm6 \n\t" // [b42,b52,b43,b53]
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm11\n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c23,c23,c32,c32]
	                "vperm2f128 $0x01 , %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b03,b13,b02,b12]
	                "\n\t"
	                "vmovapd  24*8(%[a]), %%ymm2 \n\t" // [a60,a70,a61,a71]
	                "vmovapd  28*8(%[a]), %%ymm3 \n\t" // [a62,a72,a63,a73]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm12\n\t" // [c20,c20,c31,c31]
	                "vmovupd  24*8(%[b]), %%ymm4 \n\t" // [b60,b70,b61,b71]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm9 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm13\n\t" // [c21,c21,c30,c30]
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b21,b31,b20,b30]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm10\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm14\n\t" // [c22,c22,c33,c33]
	                "vmovupd  28*8(%[b]), %%ymm6 \n\t" // [b62,b72,b63,b73]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm7 , %%ymm11\n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm15\n\t" // [c23,c23,c32,c32]
	                "vperm2f128 $0x01 , %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b23,b33,b22,b32]
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm9 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm10\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm11\n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm3 , %%ymm4 , %%ymm12\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm13\n\t" // [c21,c21,c30,c30]
	                "vfmadd231pd  %%ymm3 , %%ymm6 , %%ymm14\n\t" // [c22,c22,c33,c33]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c23,c23,c32,c32]
	                "\n\t"
	                "addq  $32*8 , %[a]\n\t"
	                "addq  $32*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 4 ){
	        //if( K >> 2 ){
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){
	            // 4x4x4x2  LD+PM/FMA = 6/8 = 0.75
	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  16*8(%[a])\n\t"
	                //"prefetcht0  16*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovupd   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,b11,b00,b10]
	                "vperm2f128 $0x01 , %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b03,b13,b02,b12]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm9 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm10\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm0 , %%ymm7 , %%ymm11\n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm12\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm13\n\t" // [c21,c21,c30,c30]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm14\n\t" // [c22,c22,c33,c33]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm15\n\t" // [c23,c23,c32,c32]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a22,a32,a23,a33]
	                "vmovupd   8*8(%[b]), %%ymm4 \n\t" // [b20,b30,b21,b31]
	                "vmovupd  12*8(%[b]), %%ymm6 \n\t" // [b22,b32,b23,b33]
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b21,b31,b20,b30]
	                "vperm2f128 $0x01 , %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b23,b33,b22,b32]
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm9 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm10\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm11\n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm3 , %%ymm4 , %%ymm12\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm13\n\t" // [c21,c21,c30,c30]
	                "vfmadd231pd  %%ymm3 , %%ymm6 , %%ymm14\n\t" // [c22,c22,c33,c33]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c23,c23,c32,c32]
	                "\n\t"
	                "addq  $16*8 , %[a]\n\t"
	                "addq  $16*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	            
	          //}
	        }
	        if( K & 2 ){

	            // 4x4x4x2  LD+PM/FMA = 6/8 = 0.75
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovupd   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,b11,b00,b10]
	                "vperm2f128 $0x01 , %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b03,b13,b02,b12]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm9 \n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm10\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm0 , %%ymm7 , %%ymm11\n\t" // [c03,c03,c12,c12]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm12\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm13\n\t" // [c21,c21,c30,c30]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm14\n\t" // [c22,c22,c33,c33]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm15\n\t" // [c23,c23,c32,c32]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[a]), %%xmm0 \n\t" // [a00,  0]
	                "vmovsd   1*8(%[a]), %%xmm1 \n\t" // [a01,  0]
	                "vmovsd   2*8(%[a]), %%xmm2 \n\t" // [a02,  0]
	                "vmovsd   3*8(%[a]), %%xmm3 \n\t" // [a03,  0]
	                "vmovsd   0*8(%[b]), %%xmm4 \n\t" // [b00,  0]
	                "vmovsd   1*8(%[b]), %%xmm5 \n\t" // [b01,  0]
	                "vmovsd   2*8(%[b]), %%xmm6 \n\t" // [b02,  0]
	                "vmovsd   3*8(%[b]), %%xmm7 \n\t" // [b03,  0]
	                "\n\t"
	                "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm0 \n\t" // [a00,  0,a01,  0]
	                "vperm2f128  $0x20, %%ymm3 , %%ymm2 , %%ymm2 \n\t" // [a02,  0,a03,  0]
	                "vperm2f128  $0x20, %%ymm5 , %%ymm4 , %%ymm4 \n\t" // [b00,  0,b01,  0]
	                "vperm2f128  $0x20, %%ymm7 , %%ymm6 , %%ymm6 \n\t" // [b02,  0,b03,  0]
	                "vperm2f128  $0x01, %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,  0,b00,  0]
	                "vperm2f128  $0x01, %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b03,  0,b02,  0]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm8 \n\t" // [c00,  0,c11,  0]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm9 \n\t" // [c01,  0,c10,  0]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm10\n\t" // [c02,  0,c13,  0]
	                "vfmadd231pd  %%ymm0 , %%ymm7 , %%ymm11\n\t" // [c03,  0,c12,  0]
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm12\n\t" // [c20,  0,c31,  0]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm13\n\t" // [c21,  0,c30,  0]
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm14\n\t" // [c22,  0,c33,  0]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm15\n\t" // [c23,  0,c32,  0]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }

	        __asm__ __volatile__ (
	            "\n\t"
	            // %%ymm8  [c00,c00,c11,c11]
	            // %%ymm9  [c01,c01,c10,c10]
	            // %%ymm10 [c02,c02,c13,c13]
	            // %%ymm11 [c03,c03,c12,c12]
	            // %%ymm12 [c20,c20,c31,c31]
	            // %%ymm13 [c21,c21,c30,c30]
	            // %%ymm14 [c22,c22,c33,c33]
	            // %%ymm15 [c23,c23,c32,c32]
	            "\n\t"
	            "vperm2f128 $0x20 , %%ymm12, %%ymm8 , %%ymm0 \n\t" // [c00,c00,c20,c20] // 4% slow down
	            "vperm2f128 $0x31 , %%ymm13, %%ymm9 , %%ymm1 \n\t" // [c10,c10,c30,c30] // 4% slow down
	            "vperm2f128 $0x20 , %%ymm13, %%ymm9 , %%ymm2 \n\t" // [c01,c01,c21,c21] // 4% slow down
	            "vperm2f128 $0x31 , %%ymm12, %%ymm8 , %%ymm3 \n\t" // [c11,c11,c31,c31] // 4% slow down
	            "vperm2f128 $0x20 , %%ymm14, %%ymm10, %%ymm4 \n\t" // [c02,c02,c22,c22] // 4% slow down
	            "vperm2f128 $0x31 , %%ymm15, %%ymm11, %%ymm5 \n\t" // [c12,c12,c32,c32] // 4% slow down
	            "vperm2f128 $0x20 , %%ymm15, %%ymm11, %%ymm6 \n\t" // [c03,c03,c23,c23] // 4% slow down
	            "vperm2f128 $0x31 , %%ymm14, %%ymm10, %%ymm7 \n\t" // [c13,c13,c33,c33] // 4% slow down
	            "\n\t"
	            "vshufpd    $0x00 , %%ymm1 , %%ymm0 , %%ymm12\n\t" // [c00,c10,c20,c30] // 2% slow down
	            "vshufpd    $0x0f , %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [c00,c10,c20,c30] // 2% slow down
	            "vshufpd    $0x00 , %%ymm3 , %%ymm2 , %%ymm13\n\t" // [c01,c11,c21,c31] // 2% slow down
	            "vshufpd    $0x0f , %%ymm3 , %%ymm2 , %%ymm9 \n\t" // [c01,c11,c21,c31] // 2% slow down
	            "vshufpd    $0x00 , %%ymm5 , %%ymm4 , %%ymm14\n\t" // [c02,c12,c22,c32] // 2% slow down
	            "vshufpd    $0x0f , %%ymm5 , %%ymm4 , %%ymm10\n\t" // [c02,c12,c22,c32] // 2% slow down
	            "vshufpd    $0x00 , %%ymm7 , %%ymm6 , %%ymm15\n\t" // [c03,c13,c23,c33] // 2% slow down
	            "vshufpd    $0x0f , %%ymm7 , %%ymm6 , %%ymm11\n\t" // [c03,c13,c23,c33] // 2% slow down
	            "\n\t"
	            "vmovapd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vaddpd             %%ymm8 , %%ymm12, %%ymm12\n\t" // [c00,c10,c20,c30]
	            "vaddpd             %%ymm9 , %%ymm13, %%ymm13\n\t" // [c01,c11,c21,c31]
	            "vaddpd             %%ymm10, %%ymm14, %%ymm14\n\t" // [c02,c12,c22,c32]
	            "vaddpd             %%ymm11, %%ymm15, %%ymm15\n\t" // [c03,c13,c23,c33]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]        ), %%ymm0, %%ymm12\n\t"
	            "vfmadd213pd 0*8(%[c1]        ), %%ymm0, %%ymm13\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc2]), %%ymm0, %%ymm14\n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2]), %%ymm0, %%ymm15\n\t"
	            "\n\t"
	            "vmovupd  %%ymm12, 0*8(%[c0]        )\n\t"
	            "vmovupd  %%ymm13, 0*8(%[c1]        )\n\t"
	            "vmovupd  %%ymm14, 0*8(%[c0],%[ldc2])\n\t"
	            "vmovupd  %%ymm15, 0*8(%[c1],%[ldc2])\n\t"
	            "\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "addq  $4*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 4*K;

	      }
	    }
	    if( M & 2 ){

	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm8 , %%ymm8 , %%ymm8 \n\t"
	            "vpxor  %%ymm9 , %%ymm9 , %%ymm9 \n\t"
	            "vpxor  %%ymm10, %%ymm10, %%ymm10\n\t"
	            "vpxor  %%ymm11, %%ymm11, %%ymm11\n\t"
	            "vpxor  %%ymm12, %%ymm12, %%ymm12\n\t"
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);

	        __asm__ __volatile__ (
	            "prefetcht0 0*8(%[c0]        )\n\t"
	            "prefetcht0 0*8(%[c1]        )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc2])\n\t"
	            "prefetcht0 0*8(%[c1],%[ldc2])\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          k8--;
	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  48*8(%[a])\n\t"
	                //"prefetcht0  96*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovupd   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovupd   8*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
	                "vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b22,b32,b23,b33]
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  48*8(%[a])\n\t"
	                //"prefetcht0  96*8(%[b])\n\t"
	                "\n\t"
	                //"vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                //"vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                //"vmovupd   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
	                //"vmovapd   4*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                //"vmovupd   8*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
	                //"vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b22,b32,b23,b33]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm13\n\t" // [c10,c10,c01,c01]
	                "vmovupd  16*8(%[b]), %%ymm4 \n\t" // [b40,b50,b41,b51]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c12,c12,c03,c03]
	                "vmovupd  20*8(%[b]), %%ymm6 \n\t" // [b42,b52,b43,b53]
	                "vmovapd   8*8(%[a]), %%ymm0 \n\t" // [a40,a50,a41,a51]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [a21,a31,a20,a30]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm13\n\t" // [c10,c10,c01,c01]
	                "vmovupd  24*8(%[b]), %%ymm5 \n\t" // [b60,b70,b61,b71]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm14\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c12,c12,c03,c03]
	                "vmovupd  28*8(%[b]), %%ymm7 \n\t" // [b62,b72,b63,b73]
	                "vmovapd  12*8(%[a]), %%ymm2 \n\t" // [a60,a70,a61,a71]
	                "\n\t"
	                //"prefetcht0 112*8(%[b])\n\t"
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm13\n\t" // [c10,c10,c01,c01]
	                "vmovupd  32*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c12,c12,c03,c03]
	                "vmovupd  36*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
	                "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [a21,a31,a20,a30]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm13\n\t" // [c10,c10,c01,c01]
	                "vmovupd  40*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm14\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c12,c12,c03,c03]
	                "vmovupd  44*8(%[b]), %%ymm7 \n\t" // [b22,b32,b23,b33]
	                "vmovapd  20*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "\n\t"
	                "addq  $16*8 , %[a]\n\t"
	                "addq  $32*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          }

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  48*8(%[a])\n\t"
	                //"prefetcht0  96*8(%[b])\n\t"
	                "\n\t"
	                //k"vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                //k"vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                //k"vmovupd   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
	                //k"vmovapd   4*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                //k"vmovupd   8*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
	                //k"vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b22,b32,b23,b33]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm13\n\t" // [c10,c10,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c12,c12,c03,c03]
	                "vmovapd   8*8(%[a]), %%ymm0 \n\t" // [a40,a50,a41,a51]
	                "vmovupd  16*8(%[b]), %%ymm4 \n\t" // [b40,b50,b41,b51]
	                "vmovupd  20*8(%[b]), %%ymm6 \n\t" // [b42,b52,b43,b53]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [a21,a31,a20,a30]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm13\n\t" // [c10,c10,c01,c01]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm14\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c12,c12,c03,c03]
	                "vmovapd  12*8(%[a]), %%ymm2 \n\t" // [a60,a70,a61,a71]
	                "vmovupd  24*8(%[b]), %%ymm5 \n\t" // [b60,b70,b61,b71]
	                "vmovupd  28*8(%[b]), %%ymm7 \n\t" // [b62,b72,b63,b73]
	                "\n\t"
	                //"prefetcht0 112*8(%[b])\n\t"
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm13\n\t" // [c10,c10,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c12,c12,c03,c03]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [a21,a31,a20,a30]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm13\n\t" // [c10,c10,c01,c01]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm14\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c12,c12,c03,c03]
	                "\n\t"
	                "addq  $16*8 , %[a]\n\t"
	                "addq  $32*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 4 ){
	        //if( K >> 2 ){
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0   8*8(%[a])\n\t"
	                //"prefetcht0  16*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovupd   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
	                "vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm13\n\t" // [c10,c10,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c12,c12,c03,c03]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovupd   8*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
	                "vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b22,b32,b23,b33]
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [a21,a31,a20,a30]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm13\n\t" // [c10,c10,c01,c01]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm14\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c12,c12,c03,c03]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $16*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          //}
	        }
	        if( K & 2 ){
	            
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovupd   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
	                "vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm13\n\t" // [c10,c10,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c12,c12,c03,c03]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){
	            
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[a]), %%xmm0 \n\t" // [a00,  0]
	                "vmovsd   1*8(%[a]), %%xmm1 \n\t" // [a01,  0]
	                "vmovsd   0*8(%[b]), %%xmm4 \n\t" // [b00,  0]
	                "vmovsd   1*8(%[b]), %%xmm5 \n\t" // [b01,  0]
	                "vmovsd   2*8(%[b]), %%xmm6 \n\t" // [b02,  0]
	                "vmovsd   3*8(%[b]), %%xmm7 \n\t" // [b03,  0]
	                "\n\t"
	                "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm0 \n\t" // [a00,  0,a01,  0]
	                "vperm2f128  $0x20, %%ymm0 , %%ymm1 , %%ymm1 \n\t" // [a01,  0,a00,  0]
	                "vperm2f128  $0x20, %%ymm5 , %%ymm4 , %%ymm4 \n\t" // [b00,  0,b01,  0]
	                "vperm2f128  $0x20, %%ymm7 , %%ymm6 , %%ymm6 \n\t" // [b02,  0,b03,  0]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,  0,c11,  0]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm13\n\t" // [c10,  0,c01,  0]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c02,  0,c13,  0]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c12,  0,c03,  0]
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	            
	        __asm__ __volatile__ (
	            "\n\t"
	            // %%ymm12 [c00,c00,c11,c11]
	            // %%ymm13 [c10,c10,c01,c01]
	            // %%ymm14 [c02,c02,c13,c13]
	            // %%ymm15 [c12,c12,c03,c03]
	            "\n\t"
	            "movupd %[alpha], %%xmm0\n\t"
	            "\n\t"
	            "vshufpd    $0x00 , %%ymm13, %%ymm12, %%ymm8 \n\t" // [c00,c10,c11,c01]
	            "vshufpd    $0x0f , %%ymm13, %%ymm12, %%ymm9 \n\t" // [c00,c10,c11,c01]
	            "vshufpd    $0x00 , %%ymm15, %%ymm14, %%ymm10\n\t" // [c02,c12,c13,c03]
	            "vshufpd    $0x0f , %%ymm15, %%ymm14, %%ymm11\n\t" // [c02,c12,c13,c03]
	            "\n\t"
	            "vaddpd             %%ymm9 , %%ymm8 , %%ymm12\n\t" // [c00,c10,c11,c01]
	            "vaddpd             %%ymm11, %%ymm10, %%ymm14\n\t" // [c02,c12,c13,c03]
	            "\n\t"
	            "vperm2f128 $0x01 , %%ymm12, %%ymm12, %%ymm13\n\t" // [c11,c01,c00,c10]
	            "vperm2f128 $0x01 , %%ymm14, %%ymm14, %%ymm15\n\t" // [c13,c03,c02,c12]
	            "vshufpd    $0x09 , %%ymm13, %%ymm13, %%ymm13\n\t" // [c01,c11,c00,c10]
	            "vshufpd    $0x09 , %%ymm15, %%ymm15, %%ymm15\n\t" // [c03,c13,c02,c12]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]        ), %%xmm0, %%xmm12\n\t" // [c00,c10,---,---]
	            "vfmadd213pd 0*8(%[c1]        ), %%xmm0, %%xmm13\n\t" // [c01,c11,---,---]
	            "vfmadd213pd 0*8(%[c0],%[ldc2]), %%xmm0, %%xmm14\n\t" // [c02,c12,---,---]
	            "vfmadd213pd 0*8(%[c1],%[ldc2]), %%xmm0, %%xmm15\n\t" // [c03,c13,---,---]
	            "\n\t"
	            "movupd  %%xmm12 , 0*8(%[c0]        )\n\t"
	            "movupd  %%xmm13 , 0*8(%[c1]        )\n\t"
	            "movupd  %%xmm14 , 0*8(%[c0],%[ldc2])\n\t"
	            "movupd  %%xmm15 , 0*8(%[c1],%[ldc2])\n\t"
	            "\n\t"
	            "addq  $2*8, %[c0]\n\t"
	            "addq  $2*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 4*K;

	    }
	    if( M & 1 ){
	            
	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm8 , %%ymm8 , %%ymm8 \n\t"
	            "vpxor  %%ymm9 , %%ymm9 , %%ymm9 \n\t"
	            "vpxor  %%ymm10, %%ymm10, %%ymm10\n\t"
	            "vpxor  %%ymm11, %%ymm11, %%ymm11\n\t"
	            "vpxor  %%ymm12, %%ymm12, %%ymm12\n\t"
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);

	        __asm__ __volatile__ (
	            "prefetcht0 0*8(%[c0]        )\n\t"
	            "prefetcht0 0*8(%[c1]        )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc2])\n\t"
	            "prefetcht0 0*8(%[c1],%[ldc2])\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  24*8(%[a])\n\t"
	                //"prefetcht0  96*8(%[b])\n\t"
	                "\n\t"
	                "vmovupd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovupd   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
	                "vmovupd   8*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
	                "vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b22,b32,b23,b33]
	                "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm8 \n\t" // [b00,b10,b20,b30]
	                "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm9 \n\t" // [b01,b11,b21,b31]
	                "vperm2f128 $0x20 , %%ymm7 , %%ymm6 , %%ymm10\n\t" // [b02,b12,b22,b32]
	                "vperm2f128 $0x31 , %%ymm7 , %%ymm6 , %%ymm11\n\t" // [b03,b13,b23,b33]
	                "vfmadd231pd  %%ymm0 , %%ymm8 , %%ymm12\n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm0 , %%ymm9 , %%ymm13\n\t" // [c01,c01,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm10, %%ymm14\n\t" // [c02,c02,c02,c02]
	                "vfmadd231pd  %%ymm0 , %%ymm11, %%ymm15\n\t" // [c03,c03,c03,c03]
	                "\n\t"
	                //"prefetcht0 112*8(%[b])\n\t"
	                "\n\t"
	                "vmovupd   4*8(%[a]), %%ymm1 \n\t" // [a40,a50,a60,a70]
	                "vmovupd  16*8(%[b]), %%ymm4 \n\t" // [b40,b50,b41,b51]
	                "vmovupd  20*8(%[b]), %%ymm6 \n\t" // [b42,b52,b43,b53]
	                "vmovupd  24*8(%[b]), %%ymm5 \n\t" // [b60,b70,b61,b71]
	                "vmovupd  28*8(%[b]), %%ymm7 \n\t" // [b62,b72,b63,b73]
	                "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm8 \n\t" // [b40,b50,b60,b70]
	                "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm9 \n\t" // [b41,b51,b61,b71]
	                "vperm2f128 $0x20 , %%ymm7 , %%ymm6 , %%ymm10\n\t" // [b42,b52,b62,b72]
	                "vperm2f128 $0x31 , %%ymm7 , %%ymm6 , %%ymm11\n\t" // [b43,b53,b63,b73]
	                "vfmadd231pd  %%ymm1 , %%ymm8 , %%ymm12\n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm1 , %%ymm9 , %%ymm13\n\t" // [c01,c01,c01,c01]
	                "vfmadd231pd  %%ymm1 , %%ymm10, %%ymm14\n\t" // [c02,c02,c02,c02]
	                "vfmadd231pd  %%ymm1 , %%ymm11, %%ymm15\n\t" // [c03,c03,c03,c03]
	                "\n\t"
	                "addq  $8*8  , %[a]\n\t"
	                "addq  $32*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          }
	        }
	        if( K & 4 ){
	        //if( K >> 2 ){
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0   4*8(%[a])\n\t"
	                //"prefetcht0  16*8(%[b])\n\t"
	                "\n\t"
	                "vmovupd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovupd   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
	                "vmovupd   8*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
	                "vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b22,b32,b23,b33]
	                "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm8 \n\t" // [b00,b10,b20,b30]
	                "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm9 \n\t" // [b01,b11,b21,b31]
	                "vperm2f128 $0x20 , %%ymm7 , %%ymm6 , %%ymm10\n\t" // [b02,b12,b22,b32]
	                "vperm2f128 $0x31 , %%ymm7 , %%ymm6 , %%ymm11\n\t" // [b03,b13,b23,b33]
	                "vfmadd231pd  %%ymm0 , %%ymm8 , %%ymm12\n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm0 , %%ymm9 , %%ymm13\n\t" // [c01,c01,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm10, %%ymm14\n\t" // [c02,c02,c02,c02]
	                "vfmadd231pd  %%ymm0 , %%ymm11, %%ymm15\n\t" // [c03,c03,c03,c03]
	                "\n\t"
	                "addq  $4*8  , %[a]\n\t"
	                "addq  $16*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          //}
	        }
	        if( K & 2 ){
	            
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,  0,  0]
	                "vmovapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10,  0,  0]
	                "vmovapd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11,  0,  0]
	                "vmovapd   4*8(%[b]), %%xmm6 \n\t" // [b02,b12,  0,  0]
	                "vmovapd   6*8(%[b]), %%xmm7 \n\t" // [b03,b13,  0,  0]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm13\n\t" // [c01,c01,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c02,c02,c02,c02]
	                "vfmadd231pd  %%ymm0 , %%ymm7 , %%ymm15\n\t" // [c03,c03,c03,c03]
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){
	            
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[a]), %%xmm0 \n\t" // [a00,  0,  0,  0]
	                "vmovsd   0*8(%[b]), %%xmm4 \n\t" // [b00,  0,  0,  0]
	                "vmovsd   1*8(%[b]), %%xmm5 \n\t" // [b01,  0,  0,  0]
	                "vmovsd   2*8(%[b]), %%xmm6 \n\t" // [b02,  0,  0,  0]
	                "vmovsd   3*8(%[b]), %%xmm7 \n\t" // [b03,  0,  0,  0]
	                "\n\t"
	                "mulsd   %%xmm0 , %%xmm4 \n\t" // [c00,--,--,--]
	                "mulsd   %%xmm0 , %%xmm5 \n\t" // [c01,--,--,--]
	                "mulsd   %%xmm0 , %%xmm6 \n\t" // [c02,--,--,--]
	                "mulsd   %%xmm0 , %%xmm7 \n\t" // [c03,--,--,--]
	                "addsd   %%xmm4 , %%xmm12\n\t" // [c00,--,--,--]
	                "addsd   %%xmm5 , %%xmm13\n\t" // [c01,--,--,--]
	                "addsd   %%xmm6 , %%xmm14\n\t" // [c02,--,--,--]
	                "addsd   %%xmm7 , %%xmm15\n\t" // [c03,--,--,--]
	                "\n\t"
	                "addq  $1*8  , %[a]\n\t"
	                "addq  $4*8  , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }

	        __asm__ __volatile__ (
	            "\n\t"
	            // %%ymm12 [c00,c00,c00,c00]
	            // %%ymm13 [c01,c01,c01,c01]
	            // %%ymm14 [c02,c02,c02,c02]
	            // %%ymm15 [c03,c03,c03,c03]
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "movlpd   0*8(%[c0]        ), %%xmm4\n\t" // [c00,---,---,---]
	            "movhpd   0*8(%[c1]        ), %%xmm4\n\t" // [c00,c01,---,---]
	            "movlpd   0*8(%[c0],%[ldc2]), %%xmm5\n\t" // [c02,---,---,---]
	            "movhpd   0*8(%[c1],%[ldc2]), %%xmm5\n\t" // [c02,c03,---,---]
	            "\n\t"
	            "vshufpd    $0x00 , %%ymm13, %%ymm12, %%ymm8 \n\t" // [c00,c01,c00,c01]
	            "vshufpd    $0x0f , %%ymm13, %%ymm12, %%ymm9 \n\t" // [c00,c01,c00,c01]
	            "vshufpd    $0x00 , %%ymm15, %%ymm14, %%ymm10\n\t" // [c02,c03,c02,c03]
	            "vshufpd    $0x0f , %%ymm15, %%ymm14, %%ymm11\n\t" // [c02,c03,c02,c03]
	            "vaddpd             %%ymm9 , %%ymm8 , %%ymm12\n\t" // [c00,c01,c00,c01]
	            "vaddpd             %%ymm11, %%ymm10, %%ymm13\n\t" // [c02,c03,c02,c03]
	            "vperm2f128 $0x31 , %%ymm12, %%ymm12, %%ymm14\n\t" // [c00,c01,c00,c01]
	            "vperm2f128 $0x31 , %%ymm13, %%ymm13, %%ymm15\n\t" // [c02,c03,c02,c03]
	            "\n\t"
	            "addpd    %%xmm12, %%xmm14\n\t" // [c00,c01,---,---]
	            "addpd    %%xmm13, %%xmm15\n\t" // [c02,c03,---,---]
	            "mulpd    %%xmm0 , %%xmm14\n\t" // [c00,c01,---,---]
	            "mulpd    %%xmm0 , %%xmm15\n\t" // [c02,c03,---,---]
	            "addpd    %%xmm4 , %%xmm14\n\t" // [c00,c01,---,---]
	            "addpd    %%xmm5 , %%xmm15\n\t" // [c02,c03,---,---]
	            "\n\t"
	            "movlpd   %%xmm14, 0*8(%[c0]        )\n\t"
	            "movhpd   %%xmm14, 0*8(%[c1]        )\n\t"
	            "movlpd   %%xmm15, 0*8(%[c0],%[ldc2])\n\t"
	            "movhpd   %%xmm15, 0*8(%[c1],%[ldc2])\n\t"
	            "\n\t"
	            "addq  $1*8, %[c0]\n\t"
	            "addq  $1*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 4*K;

	    }

	    A = A - M*K;
	    B = B + 4*K;
	    c0 = c0- M + 4*ldc;
	    c1 = c1- M + 4*ldc;

	}
	if( NR & 2 ){
	            
	  //__asm__ __volatile__ (
	  //    "\n\t"
	  //    "prefetcht0 0*8(%[a])\n\t"
	  //    "prefetcht0 0*8(%[b])\n\t"
	  //    "prefetcht0 4*8(%[a])\n\t"
	  //    "\n\t"
	  //    :[a]"+r"(A),[b]"+r"(B)
	  //:);


	    if( M >> 2 ){
	            
	      size_t m4 = ( M >> 2 ); // unrolling M
	      while( m4-- ){

	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm8 , %%ymm8 , %%ymm8 \n\t"
	            "vpxor  %%ymm9 , %%ymm9 , %%ymm9 \n\t"
	            "vpxor  %%ymm10, %%ymm10, %%ymm10\n\t"
	            "vpxor  %%ymm11, %%ymm11, %%ymm11\n\t"
	            "vpxor  %%ymm12, %%ymm12, %%ymm12\n\t"
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);

	        __asm__ __volatile__ (
	            "prefetcht0 0*8(%[c0]        )\n\t"
	            "prefetcht0 0*8(%[c1]        )\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	        :);


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          k8--;
	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  96*8(%[a])\n\t"
	                //"prefetcht0  48*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a22,a32,a23,a33]
	                "vmovupd   4*8(%[b]), %%ymm6 \n\t" // [b20,b30,b21,b31]
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0  96*8(%[a])\n\t"
	                "prefetcht0  96*8(%[b])\n\t"
	                //"prefetcht0  48*8(%[b])\n\t"
	                //"vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                //"vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                //"vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                //"vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                //"vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a22,a32,a23,a33]
	                //"vmovupd   4*8(%[b]), %%ymm6 \n\t" // [b20,b30,b21,b31]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,b11,b00,b10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm13\n\t" // [c01,c01,c10,c10]
	                "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a40,a50,a41,a51]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm14\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm15\n\t" // [c21,c21,c30,c30]
	                "vmovapd  20*8(%[a]), %%ymm1 \n\t" // [a42,a52,a43,a53]
	                "vmovupd   8*8(%[b]), %%ymm4 \n\t" // [b40,b50,b41,b51]
	                "\n\t"
	                "prefetcht0 104*8(%[a])\n\t"
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b21,b31,b20,b30]
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm13\n\t" // [c01,c01,c10,c10]
	                "vmovapd  24*8(%[a]), %%ymm2 \n\t" // [a60,a70,a61,a71]
	                "vfmadd231pd  %%ymm3 , %%ymm6 , %%ymm14\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c21,c21,c30,c30]
	                "vmovapd  28*8(%[a]), %%ymm3 \n\t" // [a62,a72,a63,a73]
	                "vmovupd  12*8(%[b]), %%ymm6 \n\t" // [b60,b70,b61,b71]
	                "\n\t"
	                "prefetcht0 112*8(%[a])\n\t"
	                "prefetcht0 104*8(%[b])\n\t"
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,b11,b00,b10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm13\n\t" // [c01,c01,c10,c10]
	                "vmovapd  32*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm14\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm15\n\t" // [c21,c21,c30,c30]
	                "vmovapd  36*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovupd  16*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "\n\t"
	                "prefetcht0 120*8(%[a])\n\t"
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b21,b31,b20,b30]
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm13\n\t" // [c01,c01,c10,c10]
	                "vmovapd  40*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vfmadd231pd  %%ymm3 , %%ymm6 , %%ymm14\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c21,c21,c30,c30]
	                "vmovapd  44*8(%[a]), %%ymm3 \n\t" // [a22,a32,a23,a33]
	                "vmovupd  20*8(%[b]), %%ymm6 \n\t" // [b20,b30,b21,b31]
	                "\n\t"
	                "addq  $32*8 , %[a]\n\t"
	                "addq  $16*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  96*8(%[a])\n\t"
	                //"prefetcht0  48*8(%[b])\n\t"
	                "\n\t"
	                //"vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                //"vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                //"vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                //"vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                //"vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a22,a32,a23,a33]
	                //"vmovupd   4*8(%[b]), %%ymm6 \n\t" // [b20,b30,b21,b31]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,b11,b00,b10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm13\n\t" // [c01,c01,c10,c10]
	                "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a40,a50,a41,a51]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm14\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm15\n\t" // [c21,c21,c30,c30]
	                "vmovapd  20*8(%[a]), %%ymm1 \n\t" // [a42,a52,a43,a53]
	                "vmovupd   8*8(%[b]), %%ymm4 \n\t" // [b40,b50,b41,b51]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b21,b31,b20,b30]
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm13\n\t" // [c01,c01,c10,c10]
	                "vmovapd  24*8(%[a]), %%ymm2 \n\t" // [a60,a70,a61,a71]
	                "vfmadd231pd  %%ymm3 , %%ymm6 , %%ymm14\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c21,c21,c30,c30]
	                "vmovapd  28*8(%[a]), %%ymm3 \n\t" // [a62,a72,a63,a73]
	                "vmovupd  12*8(%[b]), %%ymm6 \n\t" // [b60,b70,b61,b71]
	                "\n\t"
	                //"prefetcht0 112*8(%[a])\n\t"
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,b11,b00,b10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm13\n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm14\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm15\n\t" // [c21,c21,c30,c30]
	                "\n\t"
	                "vperm2f128 $0x01 , %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b21,b31,b20,b30]
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm13\n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm3 , %%ymm6 , %%ymm14\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c21,c21,c30,c30]
	                "\n\t"
	                "addq  $32*8 , %[a]\n\t"
	                "addq  $16*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 4 ){
	        //if( K >> 2 ){
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  16*8(%[a])\n\t"
	                //"prefetcht0   8*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,b11,b00,b10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm13\n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm14\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm15\n\t" // [c21,c21,c30,c30]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a22,a32,a23,a33]
	                "vmovupd   4*8(%[b]), %%ymm6 \n\t" // [b20,b30,b21,b31]
	                "vperm2f128 $0x01 , %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b21,b31,b20,b30]
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm13\n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm3 , %%ymm6 , %%ymm14\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c21,c21,c30,c30]
	                "\n\t"
	                "addq  $16*8 , %[a]\n\t"
	                "addq  $8*8  , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          //}
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,b11,b00,b10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm13\n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm14\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm15\n\t" // [c21,c21,c30,c30]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[a]), %%xmm0 \n\t" // [a00,  0]
	                "vmovsd   1*8(%[a]), %%xmm1 \n\t" // [a01,  0]
	                "vmovsd   2*8(%[a]), %%xmm2 \n\t" // [a02,  0]
	                "vmovsd   3*8(%[a]), %%xmm3 \n\t" // [a03,  0]
	                "vmovsd   0*8(%[b]), %%xmm4 \n\t" // [b00,  0]
	                "vmovsd   1*8(%[b]), %%xmm5 \n\t" // [b01,  0]
	                "\n\t"
	                "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm0 \n\t" // [a00,  0,a01,  0]
	                "vperm2f128  $0x20, %%ymm3 , %%ymm2 , %%ymm2 \n\t" // [a02,  0,a03,  0]
	                "vperm2f128  $0x20, %%ymm5 , %%ymm4 , %%ymm4 \n\t" // [b00,  0,b01,  0]
	                "vperm2f128  $0x01, %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,  0,b00,  0]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,  0,c11,  0]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm13\n\t" // [c01,  0,c10,  0]
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm14\n\t" // [c20,  0,c31,  0]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm15\n\t" // [c21,  0,c30,  0]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $2*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	            
	        __asm__ __volatile__ (
	            "\n\t"
	            // %%ymm12 [c00,c00,c11,c11]
	            // %%ymm13 [c01,c01,c10,c10]
	            // %%ymm14 [c20,c20,c31,c31]
	            // %%ymm15 [c21,c21,c30,c30]
	            "\n\t"
	            "vmovapd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vperm2f128 $0x20 , %%ymm14, %%ymm12, %%ymm8 \n\t" // [c00,c00,c20,c20] // 4% slow down
	            "vperm2f128 $0x31 , %%ymm15, %%ymm13, %%ymm9 \n\t" // [c10,c10,c30,c30] // 4% slow down
	            "vperm2f128 $0x20 , %%ymm15, %%ymm13, %%ymm10\n\t" // [c01,c01,c21,c21] // 4% slow down
	            "vperm2f128 $0x31 , %%ymm14, %%ymm12, %%ymm11\n\t" // [c11,c11,c31,c31] // 4% slow down
	            "\n\t"
	            "vshufpd    $0x00 , %%ymm9 , %%ymm8 , %%ymm14\n\t" // [c00,c10,c20,c30] // 2% slow down
	            "vshufpd    $0x0f , %%ymm9 , %%ymm8 , %%ymm12\n\t" // [c00,c10,c20,c30] // 2% slow down
	            "vshufpd    $0x00 , %%ymm11, %%ymm10, %%ymm15\n\t" // [c01,c11,c21,c31] // 2% slow down
	            "vshufpd    $0x0f , %%ymm11, %%ymm10, %%ymm13\n\t" // [c01,c11,c21,c31] // 2% slow down
	            "\n\t"
	            "vaddpd             %%ymm12, %%ymm14, %%ymm14\n\t" // [c00,c10,c20,c30]
	            "vaddpd             %%ymm13, %%ymm15, %%ymm15\n\t" // [c01,c11,c21,c31]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]        ), %%ymm0, %%ymm14\n\t"
	            "vfmadd213pd 0*8(%[c1]        ), %%ymm0, %%ymm15\n\t"
	            "\n\t"
	            "vmovupd  %%ymm14, 0*8(%[c0]        )\n\t"
	            "vmovupd  %%ymm15, 0*8(%[c1]        )\n\t"
	            "\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "addq  $4*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );


	        B = B - 2*K;

	      }
	    }
	    if( M & 2 ){

	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm12, %%ymm12, %%ymm12\n\t"
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        __asm__ __volatile__ (
	            "prefetcht0 0*8(%[c0]        )\n\t"
	            "prefetcht0 0*8(%[c1]        )\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	        :);


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  48*8(%[a])\n\t"
	                //"prefetcht0  48*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm15\n\t" // [c10,c10,c01,c01]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [a21,a31,a20,a30]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm14\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm15\n\t" // [c10,c10,c01,c01]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm8 \n\t" // [a40,a50,a41,a51]
	                "vmovupd   8*8(%[b]), %%ymm6 \n\t" // [b40,b50,b41,b51]
	                "vperm2f128 $0x01 , %%ymm8 , %%ymm8 , %%ymm9 \n\t" // [a01,a11,a00,a10]
	                "vfmadd231pd  %%ymm8 , %%ymm6 , %%ymm14\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm9 , %%ymm6 , %%ymm15\n\t" // [c10,c10,c01,c01]
	                "\n\t"
	                "vmovapd  12*8(%[a]), %%ymm10\n\t" // [a60,a70,a61,a71]
	                "vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b60,b70,b61,b71]
	                "vperm2f128 $0x01 , %%ymm10, %%ymm10, %%ymm11\n\t" // [a21,a31,a20,a30]
	                "vfmadd231pd  %%ymm10, %%ymm7 , %%ymm14\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm11, %%ymm7 , %%ymm15\n\t" // [c10,c10,c01,c01]
	                "\n\t"
	                "addq  $16*8 , %[a]\n\t"
	                "addq  $16*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	        }
	        if( K & 4 ){
	        //if( K >> 2 ){
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0   8*8(%[a])\n\t"
	                //"prefetcht0   8*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm15\n\t" // [c10,c10,c01,c01]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [a21,a31,a20,a30]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm14\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm15\n\t" // [c10,c10,c01,c01]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          //}
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm15\n\t" // [c10,c10,c01,c01]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[a]), %%xmm0 \n\t" // [a00,  0]
	                "vmovsd   1*8(%[a]), %%xmm1 \n\t" // [a01,  0]
	                "vmovsd   0*8(%[b]), %%xmm4 \n\t" // [b00,  0]
	                "vmovsd   1*8(%[b]), %%xmm5 \n\t" // [b01,  0]
	                "\n\t"
	                "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm0 \n\t" // [a00,  0,a01,  0]
	                "vperm2f128  $0x20, %%ymm0 , %%ymm1 , %%ymm1 \n\t" // [a01,  0,a00,  0]
	                "vperm2f128  $0x20, %%ymm5 , %%ymm4 , %%ymm4 \n\t" // [b00,  0,b01,  0]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t" // [c00,  0,c11,  0]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm15\n\t" // [c10,  0,c01,  0]
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $2*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	            
	        __asm__ __volatile__ (
	            "\n\t"
	            // %%ymm14 [c00,c00,c11,c11]
	            // %%ymm15 [c10,c10,c01,c01]
	            "\n\t"
	            "movupd %[alpha], %%xmm0\n\t"
	            "\n\t"
	            "vshufpd    $0x00 , %%ymm15, %%ymm14, %%ymm8 \n\t" // [c00,c10,c11,c01]
	            "vshufpd    $0x0f , %%ymm15, %%ymm14, %%ymm9 \n\t" // [c00,c10,c11,c01]
	            "\n\t"
	            "vaddpd             %%ymm9 , %%ymm8 , %%ymm12\n\t" // [c00,c10,c11,c01]
	            "\n\t"
	            "vperm2f128 $0x01 , %%ymm12, %%ymm12, %%ymm13\n\t" // [c11,c01,c00,c10]
	            "vshufpd    $0x09 , %%ymm13, %%ymm13, %%ymm13\n\t" // [c01,c11,c00,c10]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]        ), %%xmm0, %%xmm12\n\t" // [c00,c10,---,---]
	            "vfmadd213pd 0*8(%[c1]        ), %%xmm0, %%xmm13\n\t" // [c01,c11,---,---]
	            "\n\t"
	            "movupd  %%xmm12 , 0*8(%[c0]        )\n\t"
	            "movupd  %%xmm13 , 0*8(%[c1]        )\n\t"
	            "\n\t"
	            "addq  $2*8, %[c0]\n\t"
	            "addq  $2*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 2*K;

	    }
	    if( M & 1 ){

	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        __asm__ __volatile__ (
	            "prefetcht0 0*8(%[c0]        )\n\t"
	            "prefetcht0 0*8(%[c1]        )\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	        :);


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  24*8(%[a])\n\t"
	                //"prefetcht0  48*8(%[b])\n\t"
	                "\n\t"
	                "vmovupd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
	                "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm8 \n\t" // [b00,b10,b20,b30]
	                "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm9 \n\t" // [b01,b11,b21,b31]
	                "vfmadd231pd  %%ymm0 , %%ymm8 , %%ymm14\n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm0 , %%ymm9 , %%ymm15\n\t" // [c01,c01,c01,c01]
	                "\n\t"
	                "vmovupd   4*8(%[a]), %%ymm1 \n\t" // [a40,a50,a60,a70]
	                "vmovupd   8*8(%[b]), %%ymm6 \n\t" // [b40,b50,b41,b51]
	                "vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b60,b70,b61,b71]
	                "vperm2f128 $0x20 , %%ymm7 , %%ymm6 , %%ymm10\n\t" // [b40,b50,b60,b70]
	                "vperm2f128 $0x31 , %%ymm7 , %%ymm6 , %%ymm11\n\t" // [b41,b51,b61,b71]
	                "vfmadd231pd  %%ymm1 , %%ymm10, %%ymm14\n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm1 , %%ymm11, %%ymm15\n\t" // [c01,c01,c01,c01]
	                "\n\t"
	                "addq  $8*8  , %[a]\n\t"
	                "addq  $16*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          }
	        }
	        if( K & 4 ){
	        //if( K >> 2 ){
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){
	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  8*8(%[b])\n\t"
	                "\n\t"
	                "vmovupd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
	                "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm8 \n\t" // [b00,b10,b20,b30]
	                "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm9 \n\t" // [b01,b11,b21,b31]
	                "vfmadd231pd  %%ymm0 , %%ymm8 , %%ymm14\n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm0 , %%ymm9 , %%ymm15\n\t" // [c01,c01,c01,c01]
	                "\n\t"
	                "addq  $4*8  , %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          //}
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,  0,  0]
	                "vmovapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10,  0,  0]
	                "vmovapd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11,  0,  0]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm15\n\t" // [c01,c01,c01,c01]
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[a]), %%xmm0 \n\t" // [a00,  0,  0,  0]
	                "vmovsd   0*8(%[b]), %%xmm4 \n\t" // [b00,  0,  0,  0]
	                "vmovsd   1*8(%[b]), %%xmm5 \n\t" // [b01,  0,  0,  0]
	                "\n\t"
	                "mulsd   %%xmm0 , %%xmm4 \n\t" // [c00,--,--,--]
	                "mulsd   %%xmm0 , %%xmm5 \n\t" // [c01,--,--,--]
	                "addsd   %%xmm4 , %%xmm14\n\t" // [c00,--,--,--]
	                "addsd   %%xmm5 , %%xmm15\n\t" // [c01,--,--,--]
	                "\n\t"
	                "addq  $1*8  , %[a]\n\t"
	                "addq  $2*8  , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	            
	        __asm__ __volatile__ (
	            "\n\t"
	            // %%ymm14 [c00,c00,c00,c00]
	            // %%ymm15 [c01,c01,c01,c01]
	            "\n\t"
	            "vmovapd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "movlpd   0*8(%[c0]        ), %%xmm4\n\t" // [c00,---,---,---]
	            "movhpd   0*8(%[c1]        ), %%xmm4\n\t" // [c00,c01,---,---]
	            "\n\t"
	            "vshufpd    $0x00 , %%ymm15, %%ymm14, %%ymm8 \n\t" // [c00,c01,c00,c01]
	            "vshufpd    $0x0f , %%ymm15, %%ymm14, %%ymm9 \n\t" // [c00,c01,c00,c01]
	            "vaddpd             %%ymm9 , %%ymm8 , %%ymm14\n\t" // [c00,c01,c00,c01]
	            "vperm2f128 $0x31 , %%ymm14, %%ymm14, %%ymm15\n\t" // [c00,c01,c00,c01]
	            "\n\t"
	            "addpd    %%xmm14, %%xmm15\n\t" // [c00,c01,---,---]
	            "mulpd    %%xmm0 , %%xmm15\n\t" // [c00,c01,---,---]
	            "addpd    %%xmm4 , %%xmm15\n\t" // [c00,c01,---,---]
	            "\n\t"
	            "movlpd   %%xmm15, 0*8(%[c0]        )\n\t"
	            "movhpd   %%xmm15, 0*8(%[c1]        )\n\t"
	            "\n\t"
	            "addq  $1*8, %[c0]\n\t"
	            "addq  $1*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 2*K;
	    }
	    A = A - M*K;
	    B = B + 2*K;
	    c0 = c0- M + 2*ldc;
	    c1 = c1- M + 2*ldc;

	}
	if( NR & 1 ){
	            

	    if( M >> 2 ){
	            
	      size_t m4 = ( M >> 2 ); // unrolling M
	      while( m4-- ){

	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm12, %%ymm12, %%ymm12\n\t"
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        __asm__ __volatile__ (
	            "prefetcht0 0*8(%[c0]        )\n\t"
	            "prefetcht0 0*8(%[c1]        )\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	        :);

	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  96*8(%[a])\n\t"
	                //"prefetcht0  24*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a22,a32,a23,a33]
	                "vmovupd   0*8(%[b]), %%ymm8 \n\t" // [b00,b10,b20,b30]
	                "vperm2f128 $0x00 , %%ymm8 , %%ymm8 , %%ymm10\n\t" // [b00,b10,b00,b10]
	                "vperm2f128 $0x11 , %%ymm8 , %%ymm8 , %%ymm11\n\t" // [b20,b30,b20,b30]
	                "vfmadd231pd  %%ymm0 , %%ymm10, %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm10, %%ymm15\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm2 , %%ymm11, %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm3 , %%ymm11, %%ymm15\n\t" // [c20,c20,c30,c30]
	                "\n\t"
	                //"prefetcht0 112*8(%[a])\n\t"
	                "\n\t"
	                "vmovapd  16*8(%[a]), %%ymm4 \n\t" // [a40,a50,a41,a51]
	                "vmovapd  20*8(%[a]), %%ymm5 \n\t" // [a42,a52,a43,a53]
	                "vmovapd  24*8(%[a]), %%ymm6 \n\t" // [a60,a70,a61,a71]
	                "vmovapd  28*8(%[a]), %%ymm7 \n\t" // [a62,a72,a63,a73]
	                "vmovupd   4*8(%[b]), %%ymm9 \n\t" // [b40,b50,b60,b70]
	                "vperm2f128 $0x00 , %%ymm9 , %%ymm9 , %%ymm12\n\t" // [b40,b50,b40,b50]
	                "vperm2f128 $0x11 , %%ymm9 , %%ymm9 , %%ymm13\n\t" // [b60,b70,b60,b70]
	                "vfmadd231pd  %%ymm4 , %%ymm12, %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm5 , %%ymm12, %%ymm15\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm6 , %%ymm13, %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm7 , %%ymm13, %%ymm15\n\t" // [c20,c20,c30,c30]
	                "\n\t"
	                "addq  $32*8 , %[a]\n\t"
	                "addq  $8*8  , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          }
	        }
	        if( K & 4 ){
	        //if( K >> 2 ){
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  16*8(%[a])\n\t"
	                //"prefetcht0   4*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a22,a32,a23,a33]
	                "vmovupd   0*8(%[b]), %%ymm8 \n\t" // [b00,b10,b20,b30]
	                "vperm2f128 $0x00 , %%ymm8 , %%ymm8 , %%ymm10\n\t" // [b00,b10,b00,b10]
	                "vperm2f128 $0x11 , %%ymm8 , %%ymm8 , %%ymm11\n\t" // [b20,b30,b20,b30]
	                "vfmadd231pd  %%ymm0 , %%ymm10, %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm10, %%ymm15\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm2 , %%ymm11, %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm3 , %%ymm11, %%ymm15\n\t" // [c20,c20,c30,c30]
	                "\n\t"
	                "addq  $16*8 , %[a]\n\t"
	                "addq  $4*8  , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          //}
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   0*8(%[b]), %%xmm8 \n\t" // [b00,b10,  0,  0]
	                "vperm2f128 $0x00 , %%ymm8 , %%ymm8 , %%ymm10\n\t" // [b00,b10,b00,b10]
	                "vfmadd231pd  %%ymm0 , %%ymm10, %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm10, %%ymm15\n\t" // [c20,c20,c30,c30]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $2*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[a]), %%xmm0 \n\t" // [a00,  0]
	                "vmovsd   1*8(%[a]), %%xmm1 \n\t" // [a01,  0]
	                "vmovsd   2*8(%[a]), %%xmm2 \n\t" // [a02,  0]
	                "vmovsd   3*8(%[a]), %%xmm3 \n\t" // [a03,  0]
	                "vmovsd   0*8(%[b]), %%xmm4 \n\t" // [b00,  0]
	                "\n\t"
	                "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm0 \n\t" // [a00,  0,a01,  0]
	                "vperm2f128  $0x20, %%ymm3 , %%ymm2 , %%ymm2 \n\t" // [a02,  0,a03,  0]
	                "vperm2f128  $0x00, %%ymm4 , %%ymm4 , %%ymm4 \n\t" // [b00,  0,b00,  0]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t" // [c00,  0,c10,  0]
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm15\n\t" // [c20,  0,c30,  0]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $1*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	            
	        __asm__ __volatile__ (
	            "\n\t"
	            // %%ymm14 [c00,c00,c10,c10]
	            // %%ymm15 [c20,c20,c30,c30]
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vperm2f128 $0x20 , %%ymm15, %%ymm14, %%ymm8 \n\t" // [c00,c00,c20,c20] // 4% slow down
	            "vperm2f128 $0x31 , %%ymm15, %%ymm14, %%ymm9 \n\t" // [c10,c10,c30,c30] // 4% slow down
	            "vshufpd    $0x00 , %%ymm9 , %%ymm8 , %%ymm10\n\t" // [c00,c10,c20,c30] // 2% slow down
	            "vshufpd    $0x0f , %%ymm9 , %%ymm8 , %%ymm11\n\t" // [c00,c10,c20,c30] // 2% slow down
	            "vaddpd             %%ymm10, %%ymm11, %%ymm15\n\t" // [c00,c10,c20,c30]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]        ), %%ymm0, %%ymm15\n\t"
	            "vmovupd  %%ymm15, 0*8(%[c0]        )\n\t"
	            "\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "addq  $4*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 1*K;

	      }
	    }
	    if( M & 2 ){

	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);

	        __asm__ __volatile__ (
	            "prefetcht0 0*8(%[c0]        )\n\t"
	            "prefetcht0 0*8(%[c1]        )\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	        :);


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){
	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  48*8(%[a])\n\t"
	                //"prefetcht0  24*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovupd   0*8(%[b]), %%ymm8 \n\t" // [b00,b10,b20,b30]
	                "vperm2f128 $0x00 , %%ymm8 , %%ymm8 , %%ymm10\n\t" // [b00,b10,b00,b10]
	                "vperm2f128 $0x11 , %%ymm8 , %%ymm8 , %%ymm11\n\t" // [b20,b30,b20,b30]
	                "vfmadd231pd  %%ymm0 , %%ymm10, %%ymm15\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm2 , %%ymm11, %%ymm15\n\t" // [c00,c00,c10,c10]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm4 \n\t" // [a40,a50,a41,a51]
	                "vmovapd  12*8(%[a]), %%ymm6 \n\t" // [a60,a70,a61,a71]
	                "vmovupd   4*8(%[b]), %%ymm9 \n\t" // [b40,b50,b60,b70]
	                "vperm2f128 $0x00 , %%ymm9 , %%ymm9 , %%ymm12\n\t" // [b40,b50,b40,b50]
	                "vperm2f128 $0x11 , %%ymm9 , %%ymm9 , %%ymm13\n\t" // [b60,b70,b60,b70]
	                "vfmadd231pd  %%ymm4 , %%ymm12, %%ymm15\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm6 , %%ymm13, %%ymm15\n\t" // [c00,c00,c10,c10]
	                "\n\t"
	                "addq  $16*8 , %[a]\n\t"
	                "addq  $8*8  , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	        }
	        if( K & 4 ){
	        //if( K >> 2 ){
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  8*8(%[a])\n\t"
	                //"prefetcht0  4*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovupd   0*8(%[b]), %%ymm8 \n\t" // [b00,b10,b20,b30]
	                "vperm2f128 $0x00 , %%ymm8 , %%ymm8 , %%ymm10\n\t" // [b00,b10,b00,b10]
	                "vperm2f128 $0x11 , %%ymm8 , %%ymm8 , %%ymm11\n\t" // [b20,b30,b20,b30]
	                "vfmadd231pd  %%ymm0 , %%ymm10, %%ymm15\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm2 , %%ymm11, %%ymm15\n\t" // [c00,c00,c10,c10]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          //}
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   0*8(%[b]), %%xmm8 \n\t" // [b00,b10,---,---]
	                "vperm2f128 $0x00 , %%ymm8 , %%ymm8 , %%ymm10\n\t" // [b00,b10,b00,b10]
	                "vfmadd231pd  %%ymm0 , %%ymm10, %%ymm15\n\t" // [c00,c00,c10,c10]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $2*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[a]), %%xmm0 \n\t" // [a00,  0]
	                "vmovsd   1*8(%[a]), %%xmm1 \n\t" // [a01,  0]
	                "vmovsd   0*8(%[b]), %%xmm4 \n\t" // [b00,  0]
	                "\n\t"
	                "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm0 \n\t" // [a00,  0,a01,  0]
	                "vperm2f128  $0x20, %%ymm4 , %%ymm4 , %%ymm4 \n\t" // [b00,  0,b00,  0]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm15\n\t" // [c00,  0,c10,  0]
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $1*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        __asm__ __volatile__ (
	            "\n\t"
	            // %%ymm15 [c00,c00,c10,c10]
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vperm2f128 $0x01 , %%ymm15, %%ymm15, %%ymm14\n\t" // [c10,c10,c00,c00] // 4% slow down
	            "vshufpd    $0x00 , %%ymm14, %%ymm15, %%ymm12\n\t" // [c00,c10,---,---] // 2% slow down
	            "vshufpd    $0x0f , %%ymm14, %%ymm15, %%ymm13\n\t" // [c00,c10,---,---] // 2% slow down
	            "vaddpd             %%ymm13, %%ymm12, %%ymm10\n\t" // [c00,c10,---,---]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]        ), %%xmm0, %%xmm10\n\t"
	            "\n\t"
	            "movupd  %%xmm10, 0*8(%[c0]        )\n\t"
	            "\n\t"
	            "addq  $2*8, %[c0]\n\t"
	            "addq  $2*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 1*K;

	    }
	    if( M & 1 ){

	        __asm__ __volatile__ (
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);

	        __asm__ __volatile__ (
	            "prefetcht0 0*8(%[c0]        )\n\t"
	            "prefetcht0 0*8(%[c1]        )\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	        :);


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovupd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovupd   0*8(%[b]), %%ymm2 \n\t" // [b00,b10,b20,b30]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm15\n\t" // [c00,c00,c00,c00]
	                "\n\t"
	                "vmovupd   4*8(%[a]), %%ymm1 \n\t" // [a40,a50,a60,a70]
	                "vmovupd   4*8(%[b]), %%ymm3 \n\t" // [b40,b50,b60,b70]
	                "vfmadd231pd  %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c00,c00,c00,c00]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          }
	        }
	        if( K & 4 ){
	        //if( K >> 2 ){
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovupd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovupd   0*8(%[b]), %%ymm2 \n\t" // [b00,b10,b20,b30]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm15\n\t" // [c00,c00,c00,c00]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          //}
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,---,---]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10,---,---]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm4 , %%xmm6 \n\t" // [c00,c00,---,---]
	                "addpd    %%xmm6 , %%xmm15\n\t" // [c00,c00,---,---]
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $2*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "movsd    0*8(%[a]), %%xmm0 \n\t" // [a00,---,---,---]
	                "\n\t"
	                "movsd    0*8(%[b]), %%xmm4 \n\t" // [b00,---,---,---]
	                "\n\t"
	                "vmulsd   %%xmm0 , %%xmm4 , %%xmm6 \n\t" // [c00,---,---,---]
	                "addsd    %%xmm6 , %%xmm15\n\t" // [c00,---,---,---]
	                "\n\t"
	                "addq  $1*8 , %[a]\n\t"
	                "addq  $1*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "movsd   0*8(%[c0]), %%xmm1 \n\t"
	            "\n\t"
	            "vhaddpd  %%ymm15, %%ymm15, %%ymm15 \n\t" // [c00,---,c00,---]
	            "vperm2f128 $0x31, %%ymm15, %%ymm15, %%ymm14\n\t" // [c00,---,---,---]
	            "addsd    %%xmm14, %%xmm15\n\t" // [c00,---,---,---]
	            "mulsd    %%xmm0 , %%xmm15\n\t"
	            "addsd    %%xmm1 , %%xmm15\n\t"
	            "\n\t"
	            "movsd  %%xmm15, 0*8(%[c0]        )\n\t"
	            "\n\t"
	            "addq  $1*8, %[c0]\n\t"
	            "addq  $1*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 1*K;

	    }
	    A = A - M*K;
	    B = B + 1*K;
	    c0 = c0- M + 1*ldc;
	    c1 = c1- M + 1*ldc;

	}

	A = A + M*K;
	B = B - K*N;
	c0 = c0- ldc*N + M;
	c1 = c1- ldc*N + M;
	// ---- Kernel


}

