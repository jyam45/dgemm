#include "myblas_internal.h"
#include <stdio.h>

void myblas_dgemm_kernel_detail(
         size_t M, size_t N, size_t K,
         double alpha, const double *A, const double *B, 
         double *C, size_t ldc )
{
	size_t ldc1 = ldc * sizeof(double);
	double alpha4[4] = {alpha,alpha,alpha,alpha};

	//printf("A alignment = 0x%x\n",((size_t)A)&0x1f);
	//printf("B alignment = 0x%x\n",((size_t)B)&0x1f);

	// Kernel ----
	if( N >> 1 ){
	  size_t n2 = ( N >> 1 ); // unrolling N
	  while( n2-- ){
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

	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          k8--;

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0  64*8(%[a])\n\t"
	                "prefetcht0  32*8(%[b])\n\t"
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b40,b50,b60,b70]
	                "vmovupd   8*8(%[b]), %%ymm6 \n\t" // [b01,b11,b21,b31]
	                "vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b41,b51,b61,b71]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a40,a50,a60,a70]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a01,a11,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a41,a51,a61,a71]
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          while( k8-- ){

	            //__asm__ __volatile__ (
	            //    "\n\t"
	            //    "prefetcht0  64*8(%[a])\n\t"
	            //    "prefetcht0  32*8(%[b])\n\t"
	            //    "\n\t"
	            //    "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	            //    "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b40,b50,b60,b70]
	            //    "vmovupd   8*8(%[b]), %%ymm6 \n\t" // [b01,b11,b21,b31]
	            //    "vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b41,b51,b61,b71]
	            //    "\n\t"
	            //    "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	            //    "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a40,a50,a60,a70]
	            //    "\n\t"
	            //    "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c00,c00]
	            //    "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm8 \n\t" // [c00,c00,c00,c00]
	            //    "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm12\n\t" // [c01,c01,c01,c01]
	            //    "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm12\n\t" // [c01,c01,c01,c01]
	            //    "\n\t"
	            //    "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a01,a11,a21,a31]
	            //    "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a41,a51,a61,a71]
	            //    "\n\t"
	            //    "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm9 \n\t" // [c10,c10,c10,c10]
	            //    "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm9 \n\t" // [c10,c10,c10,c10]
	            //    "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm13\n\t" // [c11,c11,c11,c11]
	            //    "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm13\n\t" // [c11,c11,c11,c11]
	            //    "\n\t"
	            //    "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a02,a12,a22,a32]
	            //    "vmovapd  20*8(%[a]), %%ymm1 \n\t" // [a42,a52,a62,a72]
	            //    "\n\t"
	            //    "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c20,c20,c20,c20]
	            //    "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm10\n\t" // [c20,c20,c20,c20]
	            //    "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c21,c21,c21,c21]
	            //    "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm14\n\t" // [c21,c21,c21,c21]
	            //    "\n\t"
	            //    "vmovapd  24*8(%[a]), %%ymm2 \n\t" // [a03,a13,a23,a33]
	            //    "vmovapd  28*8(%[a]), %%ymm3 \n\t" // [a43,a53,a63,a73]
	            //    "\n\t"
	            //    "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm11\n\t" // [c30,c30,c30,c30]
	            //    "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm11\n\t" // [c30,c30,c30,c30]
	            //    "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm15\n\t" // [c31,c31,c31,c31]
	            //    "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c31,c31,c31,c31]
	            //    "\n\t"
	            //    "addq  $32*8, %[a]\n\t"
	            //    "addq  $16*8, %[b]\n\t"
	            //    "\n\t"
	            //    "\n\t"
	            //    :[a]"+r"(A),[b]"+r"(B)
	            //:);

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0  96*8(%[a])\n\t"
	                "prefetcht0  48*8(%[b])\n\t"
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm12\n\t" // [c01,c01,c01,c01]
	                "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a02,a12,a22,a32]
	                "\n\t"
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm8 \n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm12\n\t" // [c01,c01,c01,c01]
	                "vmovapd  20*8(%[a]), %%ymm1 \n\t" // [a42,a52,a62,a72]
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm9 \n\t" // [c10,c10,c10,c10]
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm13\n\t" // [c11,c11,c11,c11]
	                "vmovapd  24*8(%[a]), %%ymm2 \n\t" // [a03,a13,a23,a33]
	                "\n\t"
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm9 \n\t" // [c10,c10,c10,c10]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm13\n\t" // [c11,c11,c11,c11]
	                "vmovapd  28*8(%[a]), %%ymm3 \n\t" // [a43,a53,a63,a73]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c20,c20,c20,c20]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c21,c21,c21,c21]
	                "vmovapd  32*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "\n\t"
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm10\n\t" // [c20,c20,c20,c20]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm14\n\t" // [c21,c21,c21,c21]
	                "vmovapd  36*8(%[a]), %%ymm1 \n\t" // [a40,a50,a60,a70]
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm11\n\t" // [c30,c30,c30,c30]
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm15\n\t" // [c31,c31,c31,c31]
	                "vmovapd  40*8(%[a]), %%ymm2 \n\t" // [a01,a11,a21,a31]
	                "\n\t"
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm11\n\t" // [c30,c30,c30,c30]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c31,c31,c31,c31]
	                "vmovapd  44*8(%[a]), %%ymm3 \n\t" // [a41,a51,a61,a71]
	                "\n\t"
	                "vmovupd  16*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "vmovupd  20*8(%[b]), %%ymm5 \n\t" // [b40,b50,b60,b70]
	                "addq  $32*8, %[a]\n\t"
	                "\n\t"
	                "vmovupd  24*8(%[b]), %%ymm6 \n\t" // [b01,b11,b21,b31]
	                "vmovupd  28*8(%[b]), %%ymm7 \n\t" // [b41,b51,b61,b71]
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          }

	            __asm__ __volatile__ (
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm12\n\t" // [c01,c01,c01,c01]
	                "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a02,a12,a22,a32]
	                "\n\t"
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm8 \n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm12\n\t" // [c01,c01,c01,c01]
	                "vmovapd  20*8(%[a]), %%ymm1 \n\t" // [a42,a52,a62,a72]
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm9 \n\t" // [c10,c10,c10,c10]
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm13\n\t" // [c11,c11,c11,c11]
	                "vmovapd  24*8(%[a]), %%ymm2 \n\t" // [a03,a13,a23,a33]
	                "\n\t"
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm9 \n\t" // [c10,c10,c10,c10]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm13\n\t" // [c11,c11,c11,c11]
	                "vmovapd  28*8(%[a]), %%ymm3 \n\t" // [a43,a53,a63,a73]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c20,c20,c20,c20]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm10\n\t" // [c20,c20,c20,c20]
	                "addq  $32*8, %[a]\n\t"
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c21,c21,c21,c21]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm14\n\t" // [c21,c21,c21,c21]
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm11\n\t" // [c30,c30,c30,c30]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm11\n\t" // [c30,c30,c30,c30]
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm15\n\t" // [c31,c31,c31,c31]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c31,c31,c31,c31]
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 4 ){
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a02,a12,a22,a32]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a03,a13,a23,a30]
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b01,b11,b21,b31]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm8 \n\t"
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm9 \n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm10\n\t"
	                "vfmadd231pd  %%ymm3 , %%ymm4 , %%ymm11\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm12\n\t"
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm13\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm14\n\t"
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,---,---]
	                "movapd   2*8(%[a]), %%xmm1 \n\t" // [a01,a11,---,---]
	                "movapd   4*8(%[a]), %%xmm2 \n\t" // [a02,a12,---,---]
	                "movapd   6*8(%[a]), %%xmm3 \n\t" // [a03,a13,---,---]
	                "\n\t"
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10,---,---]
	                "movapd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11,---,---]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm4 , %%xmm6 \n\t" // [c00,c00,---,---]
	                "vmulpd   %%xmm1 , %%xmm4 , %%xmm7 \n\t" // [c10,c10,---,---]
	                "addpd    %%xmm6 , %%xmm8 \n\t" // [c00,c00,---,---]
	                "addpd    %%xmm7 , %%xmm9 \n\t" // [c10,c10,---,---]
	                "\n\t"
	                "vmulpd   %%xmm2 , %%xmm4 , %%xmm6 \n\t" // [c20,c20,---,---]
	                "vmulpd   %%xmm3 , %%xmm4 , %%xmm7 \n\t" // [c30,c30,---,---]
	                "addpd    %%xmm6 , %%xmm10\n\t" // [c20,c20,---,---]
	                "addpd    %%xmm7 , %%xmm11\n\t" // [c30,c30,---,---]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm5 , %%xmm6 \n\t" // [c01,c01,---,---]
	                "vmulpd   %%xmm1 , %%xmm5 , %%xmm7 \n\t" // [c11,c11,---,---]
	                "addpd    %%xmm6 , %%xmm12\n\t" // [c01,c01,---,---]
	                "addpd    %%xmm7 , %%xmm13\n\t" // [c11,c11,---,---]
	                "\n\t"
	                "vmulpd   %%xmm2 , %%xmm5 , %%xmm6 \n\t" // [c21,c21,---,---]
	                "vmulpd   %%xmm3 , %%xmm5 , %%xmm7 \n\t" // [c31,c31,---,---]
	                "addpd    %%xmm6 , %%xmm14\n\t" // [c21,c21,---,---]
	                "addpd    %%xmm7 , %%xmm15\n\t" // [c31,c31,---,---]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "movsd    0*8(%[a]), %%xmm0 \n\t" // [a00,---,---,---]
	                "movsd    1*8(%[a]), %%xmm1 \n\t" // [a01,---,---,---]
	                "movsd    2*8(%[a]), %%xmm2 \n\t" // [a02,---,---,---]
	                "movsd    3*8(%[a]), %%xmm3 \n\t" // [a03,---,---,---]
	                "\n\t"
	                "movsd    0*8(%[b]), %%xmm4 \n\t" // [b00,---,---,---]
	                "movsd    1*8(%[b]), %%xmm5 \n\t" // [b01,---,---,---]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm4 , %%xmm6 \n\t" // [c00,---,---,---]
	                "vmulpd   %%xmm1 , %%xmm4 , %%xmm7 \n\t" // [c10,---,---,---]
	                "addsd    %%xmm6 , %%xmm8 \n\t" // [c00,---,---,---]
	                "addsd    %%xmm7 , %%xmm9 \n\t" // [c10,---,---,---]
	                "\n\t"
	                "vmulpd   %%xmm2 , %%xmm4 , %%xmm6 \n\t" // [c20,---,---,---]
	                "vmulpd   %%xmm3 , %%xmm4 , %%xmm7 \n\t" // [c30,---,---,---]
	                "addsd    %%xmm6 , %%xmm10\n\t" // [c20,---,---,---]
	                "addsd    %%xmm7 , %%xmm11\n\t" // [c30,---,---,---]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm5 , %%xmm6 \n\t" // [c01,---,---,---]
	                "vmulpd   %%xmm1 , %%xmm5 , %%xmm7 \n\t" // [c11,---,---,---]
	                "addsd    %%xmm6 , %%xmm12\n\t" // [c01,---,---,---]
	                "addsd    %%xmm7 , %%xmm13\n\t" // [c11,---,---,---]
	                "\n\t"
	                "vmulpd   %%xmm2 , %%xmm5 , %%xmm6 \n\t" // [c21,---,---,---]
	                "vmulpd   %%xmm3 , %%xmm5 , %%xmm7 \n\t" // [c31,---,---,---]
	                "addsd    %%xmm6 , %%xmm14\n\t" // [c21,---,---,---]
	                "addsd    %%xmm7 , %%xmm15\n\t" // [c31,---,---,---]
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
	            "vmovapd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vhaddpd  %%ymm9 , %%ymm8 , %%ymm8  \n\t" // [c00,c10,c00,c10]
	            "vhaddpd  %%ymm11, %%ymm10, %%ymm10 \n\t" // [c20,c30,c20,c30]
	            "vhaddpd  %%ymm13, %%ymm12, %%ymm12 \n\t" // [c01,c11,c01,c11]
	            "vhaddpd  %%ymm15, %%ymm14, %%ymm14 \n\t" // [c21,c31,c21,c31]
	            "\n\t"
	            "vperm2f128 $0x20, %%ymm10, %%ymm8 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	            "vperm2f128 $0x31, %%ymm10, %%ymm8 , %%ymm5 \n\t" // [c00,c10,c20,c30]
	            "vperm2f128 $0x20, %%ymm14, %%ymm12, %%ymm6 \n\t" // [c01,c11,c21,c31]
	            "vperm2f128 $0x31, %%ymm14, %%ymm12, %%ymm7 \n\t" // [c01,c11,c21,c31]
	            "\n\t"
	            "vaddpd   %%ymm5 , %%ymm4 , %%ymm14\n\t" // [c00,c10,c20,c30]
	            "vaddpd   %%ymm7 , %%ymm6 , %%ymm15\n\t" // [c01,c11,c21,c31]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]        ), %%ymm0, %%ymm14\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc1]), %%ymm0, %%ymm15\n\t"
	            "\n\t"
	            "vmovupd  %%ymm14, 0*8(%[c0]        )\n\t"
	            "vmovupd  %%ymm15, 0*8(%[c0],%[ldc1])\n\t"
	            "\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"+r"(C)
	            :[alpha]"m"(alpha4),[ldc1]"r"(ldc1)
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
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0  32*8(%[a])\n\t"
	                "prefetcht0  32*8(%[b])\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a40,a50,a60,a70]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a01,a11,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a41,a51,a61,a71]
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b40,b50,b60,b70]
	                "vmovupd   8*8(%[b]), %%ymm6 \n\t" // [b01,b11,b21,b31]
	                "vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b41,b51,b61,b71]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm12\n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm13\n\t" // [c10,c10,c10,c10]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm13\n\t" // [c10,c10,c10,c10]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c01,c01,c01,c01]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm14\n\t" // [c01,c01,c01,c01]
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm15\n\t" // [c11,c11,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c11,c11,c11,c11]
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

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b01,b11,b21,b31]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t"
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm13\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm14\n\t"
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,---,---]
	                "movapd   2*8(%[a]), %%xmm1 \n\t" // [a01,a11,---,---]
	                "\n\t"
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10,---,---]
	                "movapd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11,---,---]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm4 , %%xmm6 \n\t" // [c00,c00,---,---]
	                "vmulpd   %%xmm1 , %%xmm4 , %%xmm7 \n\t" // [c10,c10,---,---]
	                "addpd    %%xmm6 , %%xmm12\n\t" // [c00,c00,---,---]
	                "addpd    %%xmm7 , %%xmm13\n\t" // [c10,c10,---,---]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm5 , %%xmm8 \n\t" // [c01,c01,---,---]
	                "vmulpd   %%xmm1 , %%xmm5 , %%xmm9 \n\t" // [c11,c11,---,---]
	                "addpd    %%xmm8 , %%xmm14\n\t" // [c01,c01,---,---]
	                "addpd    %%xmm9 , %%xmm15\n\t" // [c11,c11,---,---]
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
	                "movsd    0*8(%[a]), %%xmm0 \n\t" // [a00,---,---,---]
	                "movsd    1*8(%[a]), %%xmm1 \n\t" // [a01,---,---,---]
	                "\n\t"
	                "movsd    0*8(%[b]), %%xmm4 \n\t" // [b00,---,---,---]
	                "movsd    1*8(%[b]), %%xmm5 \n\t" // [b01,---,---,---]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm4 , %%xmm6 \n\t" // [c00,---,---,---]
	                "vmulpd   %%xmm1 , %%xmm4 , %%xmm7 \n\t" // [c10,---,---,---]
	                "addsd    %%xmm6 , %%xmm12\n\t" // [c00,---,---,---]
	                "addsd    %%xmm7 , %%xmm13\n\t" // [c10,---,---,---]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm5 , %%xmm8 \n\t" // [c01,---,---,---]
	                "vmulpd   %%xmm1 , %%xmm5 , %%xmm9 \n\t" // [c11,---,---,---]
	                "addsd    %%xmm8 , %%xmm14\n\t" // [c01,---,---,---]
	                "addsd    %%xmm9 , %%xmm15\n\t" // [c11,---,---,---]
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
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vhaddpd  %%ymm13, %%ymm12, %%ymm12 \n\t" // [c00,c10,c00,c10]
	            "vhaddpd  %%ymm15, %%ymm14, %%ymm14 \n\t" // [c01,c11,c01,c11]
	            "\n\t"
	            "vperm2f128 $0x31, %%ymm12, %%ymm12, %%ymm13\n\t" // [c00,c10,---,---]
	            "vperm2f128 $0x31, %%ymm14, %%ymm14, %%ymm15\n\t" // [c01,c11,---,---]
	            "\n\t"
	            "addpd    %%xmm13, %%xmm12\n\t" // [c00,c10,---,---]
	            "addpd    %%xmm15, %%xmm14\n\t" // [c01,c11,---,---]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]        ), %%xmm0, %%xmm12\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc1]), %%xmm0, %%xmm14\n\t"
	            "\n\t"
	            "movupd  %%xmm12, 0*8(%[c0]        )\n\t"
	            "movupd  %%xmm14, 0*8(%[c0],%[ldc1])\n\t"
	            "\n\t"
	            "addq  $2*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"+r"(C)
	            :[alpha]"m"(alpha4),[ldc1]"r"(ldc1)
	        );

	        B = B - 2*K;
	    }
	    if( M & 1 ){

	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0  16*8(%[a])\n\t"
	                "prefetcht0  32*8(%[b])\n\t"
	                "vmovupd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovupd   4*8(%[a]), %%ymm1 \n\t" // [a40,a50,a60,a70]
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b40,b50,b60,b70]
	                "vmovupd   8*8(%[b]), %%ymm6 \n\t" // [b01,b11,b21,b31]
	                "vmovupd  12*8(%[b]), %%ymm7 \n\t" // [b41,b51,b61,b71]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm14\n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm15\n\t" // [c01,c01,c01,c01]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm15\n\t" // [c01,c01,c01,c01]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	        }
	        if( K & 4 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovupd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b01,b11,b21,b31]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm15\n\t" // [c01,c01,c01,c01]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,---,---]
	                "\n\t"
	                "movupd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10,---,---]
	                "movupd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11,---,---]
	                "\n\t"
	                "mulpd    %%xmm0 , %%xmm4 \n\t"
	                "addpd    %%xmm4 , %%xmm14\n\t" // [c00,c00,---,---]
	                "\n\t"
	                "mulpd    %%xmm0 , %%xmm5 \n\t"
	                "addpd    %%xmm5 , %%xmm15\n\t" // [c01,c01,---,---]
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
	                "movsd    0*8(%[a]), %%xmm0 \n\t" // [a00,---,---,---]
	                "\n\t"
	                "movsd    0*8(%[b]), %%xmm4 \n\t" // [b00,---,---,---]
	                "movsd    1*8(%[b]), %%xmm5 \n\t" // [b01,---,---,---]
	                "\n\t"
	                "mulsd    %%xmm0 , %%xmm4 \n\t" // [c00,---,---,---]
	                "addsd    %%xmm4 , %%xmm14\n\t" // [c00,---,---,---]
	                "\n\t"
	                "mulsd    %%xmm0 , %%xmm5 \n\t" // [c01,---,---,---]
	                "addsd    %%xmm5 , %%xmm15\n\t" // [c01,---,---,---]
	                "\n\t"
	                "addq  $1*8 , %[a]\n\t"
	                "addq  $2*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "movsd  0*8(%[c0]        ), %%xmm2 \n\t"
	            "movsd  0*8(%[c0],%[ldc1]), %%xmm3 \n\t"
	            "\n\t"
	            "vhaddpd  %%ymm14, %%ymm14, %%ymm14 \n\t" // [c00,---,c00,---]
	            "vhaddpd  %%ymm15, %%ymm15, %%ymm15 \n\t" // [c01,---,c01,---]
	            "\n\t"
	            "vperm2f128 $0x31, %%ymm14, %%ymm14, %%ymm12\n\t" // [c00,---,---,---]
	            "vperm2f128 $0x31, %%ymm15, %%ymm15, %%ymm13\n\t" // [c01,---,---,---]
	            "\n\t"
	            "addsd    %%xmm12, %%xmm14\n\t" // [c00,---,---,---]
	            "addsd    %%xmm13, %%xmm15\n\t" // [c01,---,---,---]
	            "mulsd    %%xmm0 , %%xmm14\n\t" // [c00,---,---,---]
	            "mulsd    %%xmm0 , %%xmm15\n\t" // [c01,---,---,---]
	            "addsd    %%xmm2 , %%xmm14\n\t" // [c00,---,---,---]
	            "addsd    %%xmm3 , %%xmm15\n\t" // [c01,---,---,---]
	            "\n\t"
	            "movsd   %%xmm14, 0*8(%[c0]        )\n\t"
	            "movsd   %%xmm15, 0*8(%[c0],%[ldc1])\n\t"
	            "\n\t"
	            "addq  $1*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"+r"(C)
	            :[alpha]"m"(alpha4),[ldc1]"r"(ldc1)
	        );

	        B = B - 2*K;

	    }
	    A = A - M*K;
	    B = B + 2*K;
	    C  = C - M + 2*ldc;

	  }
	}
	if( N & 1 ){

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

	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0  64*8(%[a])\n\t"
	                "prefetcht0  16*8(%[b])\n\t"
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b40,b50,b60,b70]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a40,a50,a60,a70]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm12\n\t" // [c00,c00,c00,c00]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a01,a11,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a41,a51,a61,a71]
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm13\n\t" // [c10,c10,c10,c10]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm13\n\t" // [c10,c10,c10,c10]
	                "\n\t"
	                "vmovapd  16*8(%[a]), %%ymm8 \n\t" // [a02,a12,a22,a32]
	                "vmovapd  20*8(%[a]), %%ymm9 \n\t" // [a42,a52,a62,a72]
	                "\n\t"
	                "vfmadd231pd  %%ymm8 , %%ymm4 , %%ymm14\n\t" // [c20,c20,c20,c20]
	                "vfmadd231pd  %%ymm9 , %%ymm5 , %%ymm14\n\t" // [c20,c20,c20,c20]
	                "\n\t"
	                "vmovapd  24*8(%[a]), %%ymm10\n\t" // [a03,a13,a23,a33]
	                "vmovapd  28*8(%[a]), %%ymm11\n\t" // [a43,a53,a63,a73]
	                "\n\t"
	                "vfmadd231pd  %%ymm10, %%ymm4 , %%ymm15\n\t" // [c30,c30,c30,c30]
	                "vfmadd231pd  %%ymm11, %%ymm5 , %%ymm15\n\t" // [c30,c30,c30,c30]
	                "\n\t"
	                "addq  $32*8, %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	        }
	        if( K & 4 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a02,a12,a22,a32]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a03,a13,a23,a30]
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t"
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm13\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm14\n\t"
	                "vfmadd231pd  %%ymm3 , %%ymm4 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,---,---]
	                "movapd   2*8(%[a]), %%xmm1 \n\t" // [a01,a11,---,---]
	                "movapd   4*8(%[a]), %%xmm2 \n\t" // [a02,a12,---,---]
	                "movapd   6*8(%[a]), %%xmm3 \n\t" // [a03,a13,---,---]
	                "\n\t"
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10,---,---]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm4 , %%xmm6 \n\t" // [c00,c00,---,---]
	                "vmulpd   %%xmm1 , %%xmm4 , %%xmm7 \n\t" // [c10,c10,---,---]
	                "addpd    %%xmm6 , %%xmm12\n\t" // [c00,c00,---,---]
	                "addpd    %%xmm7 , %%xmm13\n\t" // [c10,c10,---,---]
	                "\n\t"
	                "vmulpd   %%xmm2 , %%xmm4 , %%xmm6 \n\t" // [c20,c20,---,---]
	                "vmulpd   %%xmm3 , %%xmm4 , %%xmm7 \n\t" // [c30,c30,---,---]
	                "addpd    %%xmm6 , %%xmm14\n\t" // [c20,c20,---,---]
	                "addpd    %%xmm7 , %%xmm15\n\t" // [c30,c30,---,---]
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
	                "movsd    0*8(%[a]), %%xmm0 \n\t" // [a00,---,---,---]
	                "movsd    1*8(%[a]), %%xmm1 \n\t" // [a01,---,---,---]
	                "movsd    2*8(%[a]), %%xmm2 \n\t" // [a02,---,---,---]
	                "movsd    3*8(%[a]), %%xmm3 \n\t" // [a03,---,---,---]
	                "\n\t"
	                "movsd    0*8(%[b]), %%xmm4 \n\t" // [b00,---,---,---]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm4 , %%xmm6 \n\t" // [c00,---,---,---]
	                "vmulpd   %%xmm1 , %%xmm4 , %%xmm7 \n\t" // [c10,---,---,---]
	                "addsd    %%xmm6 , %%xmm12\n\t" // [c00,---,---,---]
	                "addsd    %%xmm7 , %%xmm13\n\t" // [c10,---,---,---]
	                "\n\t"
	                "vmulpd   %%xmm2 , %%xmm4 , %%xmm6 \n\t" // [c20,---,---,---]
	                "vmulpd   %%xmm3 , %%xmm4 , %%xmm7 \n\t" // [c30,---,---,---]
	                "addsd    %%xmm6 , %%xmm14\n\t" // [c20,---,---,---]
	                "addsd    %%xmm7 , %%xmm15\n\t" // [c30,---,---,---]
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
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vhaddpd  %%ymm13, %%ymm12, %%ymm12 \n\t" // [c00,c10,c00,c10]
	            "vhaddpd  %%ymm15, %%ymm14, %%ymm14 \n\t" // [c20,c30,c20,c30]
	            "\n\t"
	            "vperm2f128 $0x20, %%ymm14, %%ymm12, %%ymm6 \n\t" // [c00,c10,c20,c30] 
	            "vperm2f128 $0x31, %%ymm14, %%ymm12, %%ymm7 \n\t" // [c00,c10,c20,c30] 
	            "\n\t"
	            "vaddpd   %%ymm7 , %%ymm6 , %%ymm15\n\t" // [c00,c10,c20,c30]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]        ), %%ymm0, %%ymm15\n\t"
	            "\n\t"
	            "vmovupd  %%ymm15, 0*8(%[c0]        )\n\t"
	            "\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"+r"(C)
	            :[alpha]"m"(alpha4),[ldc1]"r"(ldc1)
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
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0  32*8(%[a])\n\t"
	                "prefetcht0  16*8(%[b])\n\t"
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b40,b50,b60,b70]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a40,a50,a60,a70]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm14\n\t" // [c00,c00,c00,c00]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a01,a11,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a41,a51,a61,a71]
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm15\n\t" // [c10,c10,c10,c10]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm15\n\t" // [c10,c10,c10,c10]
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	        }
	        if( K & 4 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t"
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,---,---]
	                "movapd   2*8(%[a]), %%xmm1 \n\t" // [a01,a11,---,---]
	                "\n\t"
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10,---,---]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm4 , %%xmm6 \n\t" // [c00,c00,---,---]
	                "vmulpd   %%xmm1 , %%xmm4 , %%xmm7 \n\t" // [c10,c10,---,---]
	                "addpd    %%xmm6 , %%xmm14\n\t" // [c00,c00,---,---]
	                "addpd    %%xmm7 , %%xmm15\n\t" // [c10,c10,---,---]
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
	                "movsd    0*8(%[a]), %%xmm0 \n\t" // [a00,---,---,---]
	                "movsd    1*8(%[a]), %%xmm1 \n\t" // [a01,---,---,---]
	                "\n\t"
	                "movsd    0*8(%[b]), %%xmm4 \n\t" // [b00,---,---,---]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm4 , %%xmm6 \n\t" // [c00,---,---,---]
	                "vmulpd   %%xmm1 , %%xmm4 , %%xmm7 \n\t" // [c10,---,---,---]
	                "addsd    %%xmm6 , %%xmm14\n\t" // [c00,---,---,---]
	                "addsd    %%xmm7 , %%xmm15\n\t" // [c10,---,---,---]
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $1*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vhaddpd  %%ymm15, %%ymm14, %%ymm14 \n\t" // [c00,c10,c00,c10]
	            "\n\t"
	            "vperm2f128 $0x31, %%ymm14, %%ymm14, %%ymm15\n\t" // [c00,c10,---,---]
	            "\n\t"
	            "addpd    %%xmm14, %%xmm15\n\t" // [c00,c10,---,---]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]        ), %%xmm0, %%xmm15\n\t"
	            "\n\t"
	            "movupd  %%xmm15, 0*8(%[c0]        )\n\t"
	            "\n\t"
	            "addq  $2*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"+r"(C)
	            :[alpha]"m"(alpha4),[ldc1]"r"(ldc1)
	        );

	        B = B - 1*K;

	    }
	    if( M & 1 ){

	        __asm__ __volatile__ (
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovupd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovupd   4*8(%[a]), %%ymm1 \n\t" // [a40,a50,a60,a70]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b40,b50,b60,b70]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm15\n\t"
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	        }
	        if( K & 4 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovupd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,---,---]
	                "\n\t"
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
	            "\n\t"
	            :[c0]"+r"(C)
	            :[alpha]"m"(alpha4),[ldc1]"r"(ldc1)
	        );

	        B = B - 1*K;

	    }
	    A = A - M*K;
	    B = B + 1*K;
	    C  = C - M + 1*ldc;

	}

	A = A + M*K;
	B = B - K*N;
	C  = C - ldc*N + M;
	// ---- Kernel


}

