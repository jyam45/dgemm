#include "myblas_internal.h"
#include <stdio.h>

void myblas_dgemm_kernel_detail(
         size_t M, size_t N, size_t K,
         double alpha, const double *A, const double *B, 
         double *C, size_t ldc )
{
	double a00,a01,a02,a03;
	double a10,a11,a12,a13;
	double a20,a21,a22,a23;
	double a30,a31,a32,a33;

	double b00,b01,b02,b03;
	double b10,b11,b12,b13;
	double b20,b21,b22,b23;
	double b30,b31,b32,b33;

	double c00,c01,c02,c03;
	double c10,c11,c12,c13;
	double c20,c21,c22,c23;
	double c30,c31,c32,c33;

	double *c0 = C;
	double *c1 = C + ldc;
	size_t ldc2 = ldc * 2 * sizeof(double);
	double alpha4[4] = {alpha,alpha,alpha,alpha};

	            //printf("START:\n");
	// Kernel ----
	if( N >> 2 ){
	            //printf("N4:\n");
	  size_t n4 = ( N >> 2 ); // unrolling N
	  while( n4-- ){
	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 ); // unrolling M
	      while( m4-- ){

	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        //c10=0e0;c11=0e0;c12=0e0;c13=0e0;
	        //c20=0e0;c21=0e0;c22=0e0;c23=0e0;
	        //c30=0e0;c31=0e0;c32=0e0;c33=0e0;

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
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*8 ); a10 = *(A + 1 + 0*8 ); a20 = *(A + 2 + 0*8 ); a30 = *(A + 3 + 0*8 ); // ymm0
	            //a01 = *(A + 0 + 1*8 ); a11 = *(A + 1 + 1*8 ); a21 = *(A + 2 + 1*8 ); a31 = *(A + 3 + 1*8 ); // ymm1
	            //a02 = *(A + 0 + 2*8 ); a12 = *(A + 1 + 2*8 ); a22 = *(A + 2 + 2*8 ); a32 = *(A + 3 + 2*8 ); // ymm2
	            //a03 = *(A + 0 + 3*8 ); a13 = *(A + 1 + 3*8 ); a23 = *(A + 2 + 3*8 ); a33 = *(A + 3 + 3*8 ); // ymm3
	            //b00 = *(B + 0 + 0*8 ); b10 = *(B + 1 + 0*8 ); b20 = *(B + 2 + 0*8 ); b30 = *(B + 3 + 0*8 ); // ymm4
	            //b01 = *(B + 0 + 1*8 ); b11 = *(B + 1 + 1*8 ); b21 = *(B + 2 + 1*8 ); b31 = *(B + 3 + 1*8 ); // ymm5
	            //b02 = *(B + 0 + 2*8 ); b12 = *(B + 1 + 2*8 ); b22 = *(B + 2 + 2*8 ); b32 = *(B + 3 + 2*8 ); // ymm6
	            //b03 = *(B + 0 + 3*8 ); b13 = *(B + 1 + 3*8 ); b23 = *(B + 2 + 3*8 ); b33 = *(B + 3 + 3*8 ); // ymm7
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //c02 += a00 * b02; c02 += a10 * b12; c02 += a20 * b22; c02 += a30 * b32; // ymm14
	            //c03 += a00 * b03; c03 += a10 * b13; c03 += a20 * b23; c03 += a30 * b33; // ymm15
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            //c12 += a01 * b02; c12 += a11 * b12; c12 += a21 * b22; c12 += a31 * b32; // ymm14
	            //c13 += a01 * b03; c13 += a11 * b13; c13 += a21 * b23; c13 += a31 * b33; // ymm15
	            //c20 += a02 * b00; c20 += a12 * b10; c20 += a22 * b20; c20 += a32 * b30; // ymm12
	            //c21 += a02 * b01; c21 += a12 * b11; c21 += a22 * b21; c21 += a32 * b31; // ymm13
	            //c22 += a02 * b02; c22 += a12 * b12; c22 += a22 * b22; c22 += a32 * b32; // ymm14
	            //c23 += a02 * b03; c23 += a12 * b13; c23 += a22 * b23; c23 += a32 * b33; // ymm15
	            //c30 += a03 * b00; c30 += a13 * b10; c30 += a23 * b20; c30 += a33 * b30; // ymm12
	            //c31 += a03 * b01; c31 += a13 * b11; c31 += a23 * b21; c31 += a33 * b31; // ymm13
	            //c32 += a03 * b02; c32 += a13 * b12; c32 += a23 * b22; c32 += a33 * b32; // ymm14
	            //c33 += a03 * b03; c33 += a13 * b13; c33 += a23 * b23; c33 += a33 * b33; // ymm15
	            //a00 = *(A + 4 + 0*8 ); a10 = *(A + 5 + 0*8 ); a20 = *(A + 6 + 0*8 ); a30 = *(A + 7 + 0*8 ); // ymm0
	            //a01 = *(A + 4 + 1*8 ); a11 = *(A + 5 + 1*8 ); a21 = *(A + 6 + 1*8 ); a31 = *(A + 7 + 1*8 ); // ymm1
	            //a02 = *(A + 4 + 2*8 ); a12 = *(A + 5 + 2*8 ); a22 = *(A + 6 + 2*8 ); a32 = *(A + 7 + 2*8 ); // ymm2
	            //a03 = *(A + 4 + 3*8 ); a13 = *(A + 5 + 3*8 ); a23 = *(A + 6 + 3*8 ); a33 = *(A + 7 + 3*8 ); // ymm3
	            //b00 = *(B + 4 + 0*8 ); b10 = *(B + 5 + 0*8 ); b20 = *(B + 6 + 0*8 ); b30 = *(B + 7 + 0*8 ); // ymm4
	            //b01 = *(B + 4 + 1*8 ); b11 = *(B + 5 + 1*8 ); b21 = *(B + 6 + 1*8 ); b31 = *(B + 7 + 1*8 ); // ymm5
	            //b02 = *(B + 4 + 2*8 ); b12 = *(B + 5 + 2*8 ); b22 = *(B + 6 + 2*8 ); b32 = *(B + 7 + 2*8 ); // ymm6
	            //b03 = *(B + 4 + 3*8 ); b13 = *(B + 5 + 3*8 ); b23 = *(B + 6 + 3*8 ); b33 = *(B + 7 + 3*8 ); // ymm7
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //c02 += a00 * b02; c02 += a10 * b12; c02 += a20 * b22; c02 += a30 * b32; // ymm14
	            //c03 += a00 * b03; c03 += a10 * b13; c03 += a20 * b23; c03 += a30 * b33; // ymm15
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            //c12 += a01 * b02; c12 += a11 * b12; c12 += a21 * b22; c12 += a31 * b32; // ymm14
	            //c13 += a01 * b03; c13 += a11 * b13; c13 += a21 * b23; c13 += a31 * b33; // ymm15
	            //c20 += a02 * b00; c20 += a12 * b10; c20 += a22 * b20; c20 += a32 * b30; // ymm12
	            //c21 += a02 * b01; c21 += a12 * b11; c21 += a22 * b21; c21 += a32 * b31; // ymm13
	            //c22 += a02 * b02; c22 += a12 * b12; c22 += a22 * b22; c22 += a32 * b32; // ymm14
	            //c23 += a02 * b03; c23 += a12 * b13; c23 += a22 * b23; c23 += a32 * b33; // ymm15
	            //c30 += a03 * b00; c30 += a13 * b10; c30 += a23 * b20; c30 += a33 * b30; // ymm12
	            //c31 += a03 * b01; c31 += a13 * b11; c31 += a23 * b21; c31 += a33 * b31; // ymm13
	            //c32 += a03 * b02; c32 += a13 * b12; c32 += a23 * b22; c32 += a33 * b32; // ymm14
	            //c33 += a03 * b03; c33 += a13 * b13; c33 += a23 * b23; c33 += a33 * b33; // ymm15
	            //A+=32;
	            //B+=32;

	            // 4x4x4x2  LD+PM/FMA = 6/8 = 0.75
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovapd   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
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
	                "vmovapd   8*8(%[b]), %%ymm4 \n\t" // [b20,b30,b21,b31]
	                "vmovapd  12*8(%[b]), %%ymm6 \n\t" // [b22,b32,b23,b33]
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
	                "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a40,a50,a41,a51]
	                "vmovapd  20*8(%[a]), %%ymm1 \n\t" // [a42,a52,a43,a53]
	                "vmovapd  16*8(%[b]), %%ymm4 \n\t" // [b40,b50,b41,b51]
	                "vmovapd  20*8(%[b]), %%ymm6 \n\t" // [b42,b52,b43,b53]
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
	                "vmovapd  24*8(%[a]), %%ymm2 \n\t" // [a60,a70,a61,a71]
	                "vmovapd  28*8(%[a]), %%ymm3 \n\t" // [a62,a72,a63,a73]
	                "vmovapd  24*8(%[b]), %%ymm4 \n\t" // [b60,b70,b61,b71]
	                "vmovapd  28*8(%[b]), %%ymm6 \n\t" // [b62,b72,b63,b73]
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
	                "addq  $32*8 , %[a]\n\t"
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
	            //a00 = *(A + 0 + 0*4 ); a10 = *(A + 1 + 0*4 ); a20 = *(A + 2 + 0*4 ); a30 = *(A + 3 + 0*4 ); // ymm0
	            //a01 = *(A + 0 + 1*4 ); a11 = *(A + 1 + 1*4 ); a21 = *(A + 2 + 1*4 ); a31 = *(A + 3 + 1*4 ); // ymm1
	            //a02 = *(A + 0 + 2*4 ); a12 = *(A + 1 + 2*4 ); a22 = *(A + 2 + 2*4 ); a32 = *(A + 3 + 2*4 ); // ymm2
	            //a03 = *(A + 0 + 3*4 ); a13 = *(A + 1 + 3*4 ); a23 = *(A + 2 + 3*4 ); a33 = *(A + 3 + 3*4 ); // ymm3
	            //b00 = *(B + 0 + 0*4 ); b10 = *(B + 1 + 0*4 ); b20 = *(B + 2 + 0*4 ); b30 = *(B + 3 + 0*4 ); // ymm4
	            //b01 = *(B + 0 + 1*4 ); b11 = *(B + 1 + 1*4 ); b21 = *(B + 2 + 1*4 ); b31 = *(B + 3 + 1*4 ); // ymm5
	            //b02 = *(B + 0 + 2*4 ); b12 = *(B + 1 + 2*4 ); b22 = *(B + 2 + 2*4 ); b32 = *(B + 3 + 2*4 ); // ymm6
	            //b03 = *(B + 0 + 3*4 ); b13 = *(B + 1 + 3*4 ); b23 = *(B + 2 + 3*4 ); b33 = *(B + 3 + 3*4 ); // ymm7
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //c02 += a00 * b02; c02 += a10 * b12; c02 += a20 * b22; c02 += a30 * b32; // ymm14
	            //c03 += a00 * b03; c03 += a10 * b13; c03 += a20 * b23; c03 += a30 * b33; // ymm15
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            //c12 += a01 * b02; c12 += a11 * b12; c12 += a21 * b22; c12 += a31 * b32; // ymm14
	            //c13 += a01 * b03; c13 += a11 * b13; c13 += a21 * b23; c13 += a31 * b33; // ymm15
	            //c20 += a02 * b00; c20 += a12 * b10; c20 += a22 * b20; c20 += a32 * b30; // ymm12
	            //c21 += a02 * b01; c21 += a12 * b11; c21 += a22 * b21; c21 += a32 * b31; // ymm13
	            //c22 += a02 * b02; c22 += a12 * b12; c22 += a22 * b22; c22 += a32 * b32; // ymm14
	            //c23 += a02 * b03; c23 += a12 * b13; c23 += a22 * b23; c23 += a32 * b33; // ymm15
	            //c30 += a03 * b00; c30 += a13 * b10; c30 += a23 * b20; c30 += a33 * b30; // ymm12
	            //c31 += a03 * b01; c31 += a13 * b11; c31 += a23 * b21; c31 += a33 * b31; // ymm13
	            //c32 += a03 * b02; c32 += a13 * b12; c32 += a23 * b22; c32 += a33 * b32; // ymm14
	            //c33 += a03 * b03; c33 += a13 * b13; c33 += a23 * b23; c33 += a33 * b33; // ymm15
	            //A+=16;
	            //B+=16;

	            // 4x4x4x2  LD+PM/FMA = 6/8 = 0.75
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovapd   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
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
	                "vmovapd   8*8(%[b]), %%ymm4 \n\t" // [b20,b30,b21,b31]
	                "vmovapd  12*8(%[b]), %%ymm6 \n\t" // [b22,b32,b23,b33]
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

	            //a00 = *(A + 0 + 0*2 ); a10 = *(A + 1 + 0*2 );
	            //a01 = *(A + 0 + 1*2 ); a11 = *(A + 1 + 1*2 );
	            //a02 = *(A + 0 + 2*2 ); a12 = *(A + 1 + 2*2 );
	            //a03 = *(A + 0 + 3*2 ); a13 = *(A + 1 + 3*2 );
	            //b00 = *(B + 0 + 0*2 ); b10 = *(B + 1 + 0*2 );
	            //b01 = *(B + 0 + 1*2 ); b11 = *(B + 1 + 1*2 );
	            //b02 = *(B + 0 + 2*2 ); b12 = *(B + 1 + 2*2 );
	            //b03 = *(B + 0 + 3*2 ); b13 = *(B + 1 + 3*2 );
	            //c00 += a00 * b00; c00 += a10 * b10;
	            //c01 += a00 * b01; c01 += a10 * b11;
	            //c02 += a00 * b02; c02 += a10 * b12;
	            //c03 += a00 * b03; c03 += a10 * b13;
	            //c10 += a01 * b00; c10 += a11 * b10;
	            //c11 += a01 * b01; c11 += a11 * b11;
	            //c12 += a01 * b02; c12 += a11 * b12;
	            //c13 += a01 * b03; c13 += a11 * b13;
	            //c20 += a02 * b00; c20 += a12 * b10;
	            //c21 += a02 * b01; c21 += a12 * b11;
	            //c22 += a02 * b02; c22 += a12 * b12;
	            //c23 += a02 * b03; c23 += a12 * b13;
	            //c30 += a03 * b00; c30 += a13 * b10;
	            //c31 += a03 * b01; c31 += a13 * b11;
	            //c32 += a03 * b02; c32 += a13 * b12;
	            //c33 += a03 * b03; c33 += a13 * b13;
	            //A+=8;
	            //B+=8;

	            // 4x4x4x2  LD+PM/FMA = 6/8 = 0.75
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovapd   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
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

	            //a00 = *(A + 0 + 0*1 );
	            //a01 = *(A + 0 + 1*1 );
	            //a02 = *(A + 0 + 2*1 );
	            //a03 = *(A + 0 + 3*1 );
	            //b00 = *(B + 0 + 0*1 );
	            //b01 = *(B + 0 + 1*1 );
	            //b02 = *(B + 0 + 2*1 );
	            //b03 = *(B + 0 + 3*1 );
	            //c00 += a00 * b00;
	            //c10 += a01 * b00;
	            //c20 += a02 * b00;
	            //c30 += a03 * b00;
	            //c01 += a00 * b01;
	            //c11 += a01 * b01;
	            //c21 += a02 * b01;
	            //c31 += a03 * b01;
	            //c02 += a00 * b02;
	            //c12 += a01 * b02;
	            //c22 += a02 * b02;
	            //c32 += a03 * b02;
	            //c03 += a00 * b03;
	            //c13 += a01 * b03;
	            //c23 += a02 * b03;
	            //c33 += a03 * b03;
	            //A+=4;
	            //B+=4;

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
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+0+1*ldc) += alpha*c01;
	        //*(C+0+2*ldc) += alpha*c02;
	        //*(C+0+3*ldc) += alpha*c03;
	        //*(C+1+0*ldc) += alpha*c10;
	        //*(C+1+1*ldc) += alpha*c11;
	        //*(C+1+2*ldc) += alpha*c12;
	        //*(C+1+3*ldc) += alpha*c13;
	        //*(C+2+0*ldc) += alpha*c20;
	        //*(C+2+1*ldc) += alpha*c21;
	        //*(C+2+2*ldc) += alpha*c22;
	        //*(C+2+3*ldc) += alpha*c23;
	        //*(C+3+0*ldc) += alpha*c30;
	        //*(C+3+1*ldc) += alpha*c31;
	        //*(C+3+2*ldc) += alpha*c32;
	        //*(C+3+3*ldc) += alpha*c33;
	        ////A = A - K + 4*K;
	        //B = B - 4*K;
	        //C+=4;
	        //////printf("pass through (n>>2)&(m>>2)\n");

	        __asm__ __volatile__ (
	            "\n\t"
	            //"vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c11,c11]
	            //"vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm9 \n\t" // [c01,c01,c10,c10]
	            //"vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm10\n\t" // [c02,c02,c13,c13]
	            //"vfmadd231pd  %%ymm0 , %%ymm7 , %%ymm11\n\t" // [c03,c03,c12,c12]
	            //"vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm12\n\t" // [c20,c20,c31,c31]
	            //"vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm13\n\t" // [c21,c21,c30,c30]
	            //"vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm14\n\t" // [c22,c22,c33,c33]
	            //"vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm15\n\t" // [c23,c23,c32,c32]
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
	            //"vhaddpd            %%ymm1 , %%ymm0 , %%ymm12\n\t" // [c00,c10,c20,c30] // slow down 5%
	            //"vhaddpd            %%ymm3 , %%ymm2 , %%ymm13\n\t" // [c01,c11,c21,c31] // slow down 5%
	            //"vhaddpd            %%ymm5 , %%ymm4 , %%ymm14\n\t" // [c02,c12,c22,c32] // slow down 5%
	            //"vhaddpd            %%ymm7 , %%ymm6 , %%ymm15\n\t" // [c03,c13,c23,c33] // slow down 5%
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
	        C+=4;

	      }
	    }
	    if( M & 2 ){

	            //printf("  M2:\n");
	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        //c10=0e0;c11=0e0;c12=0e0;c13=0e0;

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
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*8 ); a10 = *(A + 1 + 0*8 ); a20 = *(A + 2 + 0*8 ); a30 = *(A + 3 + 0*8 ); // ymm0
	            //a01 = *(A + 0 + 1*8 ); a11 = *(A + 1 + 1*8 ); a21 = *(A + 2 + 1*8 ); a31 = *(A + 3 + 1*8 ); // ymm1
	            //b00 = *(B + 0 + 0*8 ); b10 = *(B + 1 + 0*8 ); b20 = *(B + 2 + 0*8 ); b30 = *(B + 3 + 0*8 ); // ymm4
	            //b01 = *(B + 0 + 1*8 ); b11 = *(B + 1 + 1*8 ); b21 = *(B + 2 + 1*8 ); b31 = *(B + 3 + 1*8 ); // ymm5
	            //b02 = *(B + 0 + 2*8 ); b12 = *(B + 1 + 2*8 ); b22 = *(B + 2 + 2*8 ); b32 = *(B + 3 + 2*8 ); // ymm6
	            //b03 = *(B + 0 + 3*8 ); b13 = *(B + 1 + 3*8 ); b23 = *(B + 2 + 3*8 ); b33 = *(B + 3 + 3*8 ); // ymm7
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //c02 += a00 * b02; c02 += a10 * b12; c02 += a20 * b22; c02 += a30 * b32; // ymm14
	            //c03 += a00 * b03; c03 += a10 * b13; c03 += a20 * b23; c03 += a30 * b33; // ymm15
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            //c12 += a01 * b02; c12 += a11 * b12; c12 += a21 * b22; c12 += a31 * b32; // ymm14
	            //c13 += a01 * b03; c13 += a11 * b13; c13 += a21 * b23; c13 += a31 * b33; // ymm15
	            //a00 = *(A + 4 + 0*8 ); a10 = *(A + 5 + 0*8 ); a20 = *(A + 6 + 0*8 ); a30 = *(A + 7 + 0*8 ); // ymm0
	            //a01 = *(A + 4 + 1*8 ); a11 = *(A + 5 + 1*8 ); a21 = *(A + 6 + 1*8 ); a31 = *(A + 7 + 1*8 ); // ymm1
	            //b00 = *(B + 4 + 0*8 ); b10 = *(B + 5 + 0*8 ); b20 = *(B + 6 + 0*8 ); b30 = *(B + 7 + 0*8 ); // ymm4
	            //b01 = *(B + 4 + 1*8 ); b11 = *(B + 5 + 1*8 ); b21 = *(B + 6 + 1*8 ); b31 = *(B + 7 + 1*8 ); // ymm5
	            //b02 = *(B + 4 + 2*8 ); b12 = *(B + 5 + 2*8 ); b22 = *(B + 6 + 2*8 ); b32 = *(B + 7 + 2*8 ); // ymm6
	            //b03 = *(B + 4 + 3*8 ); b13 = *(B + 5 + 3*8 ); b23 = *(B + 6 + 3*8 ); b33 = *(B + 7 + 3*8 ); // ymm7
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //c02 += a00 * b02; c02 += a10 * b12; c02 += a20 * b22; c02 += a30 * b32; // ymm14
	            //c03 += a00 * b03; c03 += a10 * b13; c03 += a20 * b23; c03 += a30 * b33; // ymm15
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            //c12 += a01 * b02; c12 += a11 * b12; c12 += a21 * b22; c12 += a31 * b32; // ymm14
	            //c13 += a01 * b03; c13 += a11 * b13; c13 += a21 * b23; c13 += a31 * b33; // ymm15
	            //A+=16;
	            //B+=32;
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovapd   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
	                "vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm13\n\t" // [c10,c10,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c12,c12,c03,c03]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd   8*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
	                "vmovapd  12*8(%[b]), %%ymm7 \n\t" // [b22,b32,b23,b33]
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [a21,a31,a20,a30]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm13\n\t" // [c10,c10,c01,c01]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm14\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c12,c12,c03,c03]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm0 \n\t" // [a40,a50,a41,a51]
	                "vmovapd  16*8(%[b]), %%ymm4 \n\t" // [b40,b50,b41,b51]
	                "vmovapd  20*8(%[b]), %%ymm6 \n\t" // [b42,b52,b43,b53]
	                "vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm13\n\t" // [c10,c10,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c12,c12,c03,c03]
	                "\n\t"
	                "vmovapd  12*8(%[a]), %%ymm2 \n\t" // [a60,a70,a61,a71]
	                "vmovapd  24*8(%[b]), %%ymm5 \n\t" // [b60,b70,b61,b71]
	                "vmovapd  28*8(%[b]), %%ymm7 \n\t" // [b62,b72,b63,b73]
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
	        }
	        if( K & 4 ){
	        //if( K >> 2 ){
	        //    //printf("    K4:\n");
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){
	            //a00 = *(A + 0 + 0*4 ); a10 = *(A + 1 + 0*4 ); a20 = *(A + 2 + 0*4 ); a30 = *(A + 3 + 0*4 ); // ymm0
	            //a01 = *(A + 0 + 1*4 ); a11 = *(A + 1 + 1*4 ); a21 = *(A + 2 + 1*4 ); a31 = *(A + 3 + 1*4 ); // ymm1
	            //b00 = *(B + 0 + 0*4 ); b10 = *(B + 1 + 0*4 ); b20 = *(B + 2 + 0*4 ); b30 = *(B + 3 + 0*4 ); // ymm4
	            //b01 = *(B + 0 + 1*4 ); b11 = *(B + 1 + 1*4 ); b21 = *(B + 2 + 1*4 ); b31 = *(B + 3 + 1*4 ); // ymm5
	            //b02 = *(B + 0 + 2*4 ); b12 = *(B + 1 + 2*4 ); b22 = *(B + 2 + 2*4 ); b32 = *(B + 3 + 2*4 ); // ymm6
	            //b03 = *(B + 0 + 3*4 ); b13 = *(B + 1 + 3*4 ); b23 = *(B + 2 + 3*4 ); b33 = *(B + 3 + 3*4 ); // ymm7
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //c02 += a00 * b02; c02 += a10 * b12; c02 += a20 * b22; c02 += a30 * b32; // ymm14
	            //c03 += a00 * b03; c03 += a10 * b13; c03 += a20 * b23; c03 += a30 * b33; // ymm15
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            //c12 += a01 * b02; c12 += a11 * b12; c12 += a21 * b22; c12 += a31 * b32; // ymm14
	            //c13 += a01 * b03; c13 += a11 * b13; c13 += a21 * b23; c13 += a31 * b33; // ymm15
	            //A+=8;
	            //B+=16;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovapd   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
	                "vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm13\n\t" // [c10,c10,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c02,c02,c13,c13]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c12,c12,c03,c03]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd   8*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
	                "vmovapd  12*8(%[b]), %%ymm7 \n\t" // [b22,b32,b23,b33]
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
	            //printf("    K2:\n");

	            //a00 = *(A + 0 + 0*2 ); a10 = *(A + 1 + 0*2 );
	            //a01 = *(A + 0 + 1*2 ); a11 = *(A + 1 + 1*2 );
	            //b00 = *(B + 0 + 0*2 ); b10 = *(B + 1 + 0*2 );
	            //b01 = *(B + 0 + 1*2 ); b11 = *(B + 1 + 1*2 );
	            //b02 = *(B + 0 + 2*2 ); b12 = *(B + 1 + 2*2 );
	            //b03 = *(B + 0 + 3*2 ); b13 = *(B + 1 + 3*2 );
	            //c00 += a00 * b00; c00 += a10 * b10;
	            //c01 += a00 * b01; c01 += a10 * b11;
	            //c02 += a00 * b02; c02 += a10 * b12;
	            //c03 += a00 * b03; c03 += a10 * b13;
	            //c10 += a01 * b00; c10 += a11 * b10;
	            //c11 += a01 * b01; c11 += a11 * b11;
	            //c12 += a01 * b02; c12 += a11 * b12;
	            //c13 += a01 * b03; c13 += a11 * b13;
	            //A+=4;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovapd   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
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
	            //printf("    K1:\n");

	            //a00 = *(A + 0 + 0*1 );
	            //a01 = *(A + 0 + 1*1 );
	            //b00 = *(B + 0 + 0*1 );
	            //b01 = *(B + 0 + 1*1 );
	            //b02 = *(B + 0 + 2*1 );
	            //b03 = *(B + 0 + 3*1 );
	            //c00 += a00 * b00;
	            //c01 += a00 * b01;
	            //c02 += a00 * b02;
	            //c03 += a00 * b03;
	            //c10 += a01 * b00;
	            //c11 += a01 * b01;
	            //c12 += a01 * b02;
	            //c13 += a01 * b03;
	            //A+=2;
	            //B+=4;
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
	                "vperm2f128  $0x20, %%ymm5 , %%ymm4 , %%ymm4 \n\t" // [b00,  0,b01,  0]
	                "vperm2f128  $0x20, %%ymm7 , %%ymm6 , %%ymm6 \n\t" // [b02,  0,b03,  0]
	                "vperm2f128  $0x01, %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,  0,b00,  0]
	                "vperm2f128  $0x01, %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b03,  0,b02,  0]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm8 \n\t" // [c00,  0,c11,  0]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm9 \n\t" // [c01,  0,c10,  0]
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm10\n\t" // [c02,  0,c13,  0]
	                "vfmadd231pd  %%ymm0 , %%ymm7 , %%ymm11\n\t" // [c03,  0,c12,  0]
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);



	        }
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+0+1*ldc) += alpha*c01;
	        //*(C+0+2*ldc) += alpha*c02;
	        //*(C+0+3*ldc) += alpha*c03;
	        //*(C+1+0*ldc) += alpha*c10;
	        //*(C+1+1*ldc) += alpha*c11;
	        //*(C+1+2*ldc) += alpha*c12;
	        //*(C+1+3*ldc) += alpha*c13;
	        ////A = A - K + 4*K;
	        //B = B - 4*K;
	        //C+=2;
	        //c0+=2;
	        //c1+=2;
	        //////printf("pass through (n>>2)&(m&2)\n");

	            //printf("    END:\n");
	        __asm__ __volatile__ (
	            "\n\t"
	            //"vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	            //"vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm13\n\t" // [c10,c10,c01,c01]
	            //"vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c02,c02,c13,c13]
	            //"vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm15\n\t" // [c12,c12,c03,c03]
	            "\n\t"
	            "movupd %[alpha], %%xmm0\n\t"
	            "\n\t"
	            "vshufpd    $0x00 , %%ymm13, %%ymm12, %%ymm8 \n\t" // [c00,c10,c11,c01]
	            "vshufpd    $0x0f , %%ymm13, %%ymm12, %%ymm9 \n\t" // [c00,c10,c11,c01]
	            "vshufpd    $0x00 , %%ymm15, %%ymm14, %%ymm10\n\t" // [c02,c12,c13,c03]
	            "vshufpd    $0x0f , %%ymm14, %%ymm14, %%ymm11\n\t" // [c02,c12,c13,c03]
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
	        C+=2;

	    }
	    if( M & 1 ){
	            //printf("  M1:\n");

	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
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
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*8 ); a10 = *(A + 1 + 0*8 ); a20 = *(A + 2 + 0*8 ); a30 = *(A + 3 + 0*8 ); // ymm0
	            //b00 = *(B + 0 + 0*8 ); b10 = *(B + 1 + 0*8 ); b20 = *(B + 2 + 0*8 ); b30 = *(B + 3 + 0*8 ); // ymm4
	            //b01 = *(B + 0 + 1*8 ); b11 = *(B + 1 + 1*8 ); b21 = *(B + 2 + 1*8 ); b31 = *(B + 3 + 1*8 ); // ymm5
	            //b02 = *(B + 0 + 2*8 ); b12 = *(B + 1 + 2*8 ); b22 = *(B + 2 + 2*8 ); b32 = *(B + 3 + 2*8 ); // ymm6
	            //b03 = *(B + 0 + 3*8 ); b13 = *(B + 1 + 3*8 ); b23 = *(B + 2 + 3*8 ); b33 = *(B + 3 + 3*8 ); // ymm7
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //c02 += a00 * b02; c02 += a10 * b12; c02 += a20 * b22; c02 += a30 * b32; // ymm14
	            //c03 += a00 * b03; c03 += a10 * b13; c03 += a20 * b23; c03 += a30 * b33; // ymm15
	            //a00 = *(A + 4 + 0*8 ); a10 = *(A + 5 + 0*8 ); a20 = *(A + 6 + 0*8 ); a30 = *(A + 7 + 0*8 ); // ymm0
	            //b00 = *(B + 4 + 0*8 ); b10 = *(B + 5 + 0*8 ); b20 = *(B + 6 + 0*8 ); b30 = *(B + 7 + 0*8 ); // ymm4
	            //b01 = *(B + 4 + 1*8 ); b11 = *(B + 5 + 1*8 ); b21 = *(B + 6 + 1*8 ); b31 = *(B + 7 + 1*8 ); // ymm5
	            //b02 = *(B + 4 + 2*8 ); b12 = *(B + 5 + 2*8 ); b22 = *(B + 6 + 2*8 ); b32 = *(B + 7 + 2*8 ); // ymm6
	            //b03 = *(B + 4 + 3*8 ); b13 = *(B + 5 + 3*8 ); b23 = *(B + 6 + 3*8 ); b33 = *(B + 7 + 3*8 ); // ymm7
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //c02 += a00 * b02; c02 += a10 * b12; c02 += a20 * b22; c02 += a30 * b32; // ymm14
	            //c03 += a00 * b03; c03 += a10 * b13; c03 += a20 * b23; c03 += a30 * b33; // ymm15
	            //A+=8;
	            //B+=32;
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovapd   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
	                "vmovapd   8*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
	                "vmovapd  12*8(%[b]), %%ymm7 \n\t" // [b22,b32,b23,b33]
	                "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm8 \n\t" // [b00,b10,b20,b30]
	                "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm9 \n\t" // [b01,b11,b21,b31]
	                "vperm2f128 $0x20 , %%ymm7 , %%ymm6 , %%ymm10\n\t" // [b02,b12,b22,b32]
	                "vperm2f128 $0x31 , %%ymm7 , %%ymm6 , %%ymm11\n\t" // [b03,b13,b23,b33]
	                "vfmadd231pd  %%ymm0 , %%ymm8 , %%ymm12\n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm0 , %%ymm9 , %%ymm13\n\t" // [c01,c01,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm10, %%ymm14\n\t" // [c02,c02,c02,c02]
	                "vfmadd231pd  %%ymm0 , %%ymm11, %%ymm15\n\t" // [c03,c03,c03,c03]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a40,a50,a60,a70]
	                "vmovapd  16*8(%[b]), %%ymm4 \n\t" // [b40,b50,b41,b51]
	                "vmovapd  20*8(%[b]), %%ymm6 \n\t" // [b42,b52,b43,b53]
	                "vmovapd  24*8(%[b]), %%ymm5 \n\t" // [b60,b70,b61,b71]
	                "vmovapd  28*8(%[b]), %%ymm7 \n\t" // [b62,b72,b63,b73]
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
	        //    //printf("    K4:\n");
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){
	            //a00 = *(A + 0 + 0*4 ); a10 = *(A + 1 + 0*4 ); a20 = *(A + 2 + 0*4 ); a30 = *(A + 3 + 0*4 ); // ymm0
	            //b00 = *(B + 0 + 0*4 ); b10 = *(B + 1 + 0*4 ); b20 = *(B + 2 + 0*4 ); b30 = *(B + 3 + 0*4 ); // ymm4
	            //b01 = *(B + 0 + 1*4 ); b11 = *(B + 1 + 1*4 ); b21 = *(B + 2 + 1*4 ); b31 = *(B + 3 + 1*4 ); // ymm5
	            //b02 = *(B + 0 + 2*4 ); b12 = *(B + 1 + 2*4 ); b22 = *(B + 2 + 2*4 ); b32 = *(B + 3 + 2*4 ); // ymm6
	            //b03 = *(B + 0 + 3*4 ); b13 = *(B + 1 + 3*4 ); b23 = *(B + 2 + 3*4 ); b33 = *(B + 3 + 3*4 ); // ymm7
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //c02 += a00 * b02; c02 += a10 * b12; c02 += a20 * b22; c02 += a30 * b32; // ymm14
	            //c03 += a00 * b03; c03 += a10 * b13; c03 += a20 * b23; c03 += a30 * b33; // ymm15
	            //A+=4;
	            //B+=16;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovapd   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b03,b13]
	                "vmovapd   8*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
	                "vmovapd  12*8(%[b]), %%ymm7 \n\t" // [b22,b32,b23,b33]
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
	            //printf("    K2:\n");

	            //a00 = *(A + 0 + 0*2 ); a10 = *(A + 1 + 0*2 );
	            //b00 = *(B + 0 + 0*2 ); b10 = *(B + 1 + 0*2 );
	            //b01 = *(B + 0 + 1*2 ); b11 = *(B + 1 + 1*2 );
	            //b02 = *(B + 0 + 2*2 ); b12 = *(B + 1 + 2*2 );
	            //b03 = *(B + 0 + 3*2 ); b13 = *(B + 1 + 3*2 );
	            //c00 += a00 * b00; c00 += a10 * b10;
	            //c01 += a00 * b01; c01 += a10 * b11;
	            //c02 += a00 * b02; c02 += a10 * b12;
	            //c03 += a00 * b03; c03 += a10 * b13;
	            //A+=2;
	            //B+=8;

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
	            //printf("    K1:\n");

	            //a00 = *(A + 0 + 0*1 );
	            //b00 = *(B + 0 + 0*1 );
	            //b01 = *(B + 0 + 1*1 );
	            //b02 = *(B + 0 + 2*1 );
	            //b03 = *(B + 0 + 3*1 );
	            //c00 += a00 * b00;
	            //c01 += a00 * b01;
	            //c02 += a00 * b02;
	            //c03 += a00 * b03;
	            //A+=1;
	            //B+=4;

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
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+0+1*ldc) += alpha*c01;
	        //*(C+0+2*ldc) += alpha*c02;
	        //*(C+0+3*ldc) += alpha*c03;
	        ////A = A - K + 4*K;
	        //B = B - 4*K;
	        //C+=1;
	        //c0+=1;
	        //c1+=1;
	        //////printf("pass through (n>>2)&(m&1)\n");
	            //printf("    END:\n");

	        __asm__ __volatile__ (
	            "\n\t"
	            //"vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c00,c00]
	            //"vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm13\n\t" // [c01,c01,c01,c01]
	            //"vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm14\n\t" // [c02,c02,c02,c02]
	            //"vfmadd231pd  %%ymm0 , %%ymm7 , %%ymm15\n\t" // [c03,c03,c03,c03]
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
	            "vshufpd    $0x0f , %%ymm14, %%ymm14, %%ymm11\n\t" // [c02,c03,c02,c03]
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
	        C+=1;

	    }
	    //if( M & 3 ){
	    //  size_t mr = ( M & 3 ); // unrolling M
	    //  while( mr-- ){
	    ////  while( m-- ){
	    //    //AB=0e0;
	    //    c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	    //    size_t k = K;
	    //    while( k-- ){
	    //      a00  = *(A + 0*K );
	    //      b00  = *(B + 0*K );
	    //      b01  = *(B + 1*K );
	    //      b02  = *(B + 2*K );
	    //      b03  = *(B + 3*K );
	    //      c00 += a00 * b00;
	    //      c01 += a00 * b01;
	    //      c02 += a00 * b02;
	    //      c03 += a00 * b03;
	    //      //AB = AB + (*A)*(*B);
	    //      A++;
	    //      B++;
	    //    }
	    //    //*C = (*C) + alpha*AB;
	    //    *(C+0+0*ldc) += alpha*c00;
	    //    *(C+0+1*ldc) += alpha*c01;
	    //    *(C+0+2*ldc) += alpha*c02;
	    //    *(C+0+3*ldc) += alpha*c03;
	    //    B = B - K;
	    //    C++;
	    //  }
	    //}
	    A = A - M*K;
	    B = B + 4*K;
	    C  = C - M + 4*ldc;
	    c0 = c0- M + 4*ldc;
	    c1 = c1- M + 4*ldc;
	  }
	}
	if( N & 2 ){
	            //printf("N2:\n");

	    if( M >> 2 ){
	            //printf("  M4:\n");
	      size_t m4 = ( M >> 2 ); // unrolling M
	      while( m4-- ){
	        //AB=0e0;
	        //c00=0e0;c01=0e0;
	        //c10=0e0;c11=0e0;
	        //c20=0e0;c21=0e0;
	        //c30=0e0;c31=0e0;
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
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*8 ); a10 = *(A + 1 + 0*8 ); a20 = *(A + 2 + 0*8 ); a30 = *(A + 3 + 0*8 ); // ymm0
	            //a01 = *(A + 0 + 1*8 ); a11 = *(A + 1 + 1*8 ); a21 = *(A + 2 + 1*8 ); a31 = *(A + 3 + 1*8 ); // ymm1
	            //a02 = *(A + 0 + 2*8 ); a12 = *(A + 1 + 2*8 ); a22 = *(A + 2 + 2*8 ); a32 = *(A + 3 + 2*8 ); // ymm2
	            //a03 = *(A + 0 + 3*8 ); a13 = *(A + 1 + 3*8 ); a23 = *(A + 2 + 3*8 ); a33 = *(A + 3 + 3*8 ); // ymm3
	            //b00 = *(B + 0 + 0*8 ); b10 = *(B + 1 + 0*8 ); b20 = *(B + 2 + 0*8 ); b30 = *(B + 3 + 0*8 ); // ymm4
	            //b01 = *(B + 0 + 1*8 ); b11 = *(B + 1 + 1*8 ); b21 = *(B + 2 + 1*8 ); b31 = *(B + 3 + 1*8 ); // ymm5
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            //c20 += a02 * b00; c20 += a12 * b10; c20 += a22 * b20; c20 += a32 * b30; // ymm12
	            //c21 += a02 * b01; c21 += a12 * b11; c21 += a22 * b21; c21 += a32 * b31; // ymm13
	            //c30 += a03 * b00; c30 += a13 * b10; c30 += a23 * b20; c30 += a33 * b30; // ymm12
	            //c31 += a03 * b01; c31 += a13 * b11; c31 += a23 * b21; c31 += a33 * b31; // ymm13
	            //a00 = *(A + 4 + 0*8 ); a10 = *(A + 5 + 0*8 ); a20 = *(A + 6 + 0*8 ); a30 = *(A + 7 + 0*8 ); // ymm0
	            //a01 = *(A + 4 + 1*8 ); a11 = *(A + 5 + 1*8 ); a21 = *(A + 6 + 1*8 ); a31 = *(A + 7 + 1*8 ); // ymm1
	            //a02 = *(A + 4 + 2*8 ); a12 = *(A + 5 + 2*8 ); a22 = *(A + 6 + 2*8 ); a32 = *(A + 7 + 2*8 ); // ymm2
	            //a03 = *(A + 4 + 3*8 ); a13 = *(A + 5 + 3*8 ); a23 = *(A + 6 + 3*8 ); a33 = *(A + 7 + 3*8 ); // ymm3
	            //b00 = *(B + 4 + 0*8 ); b10 = *(B + 5 + 0*8 ); b20 = *(B + 6 + 0*8 ); b30 = *(B + 7 + 0*8 ); // ymm4
	            //b01 = *(B + 4 + 1*8 ); b11 = *(B + 5 + 1*8 ); b21 = *(B + 6 + 1*8 ); b31 = *(B + 7 + 1*8 ); // ymm5
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            //c20 += a02 * b00; c20 += a12 * b10; c20 += a22 * b20; c20 += a32 * b30; // ymm12
	            //c21 += a02 * b01; c21 += a12 * b11; c21 += a22 * b21; c21 += a32 * b31; // ymm13
	            //c30 += a03 * b00; c30 += a13 * b10; c30 += a23 * b20; c30 += a33 * b30; // ymm12
	            //c31 += a03 * b01; c31 += a13 * b11; c31 += a23 * b21; c31 += a33 * b31; // ymm13
	            //A+=32;
	            //B+=16;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,b11,b00,b10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm13\n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm14\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm15\n\t" // [c21,c21,c30,c30]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a22,a32,a23,a33]
	                "vmovapd   4*8(%[b]), %%ymm6 \n\t" // [b20,b30,b21,b31]
	                "vperm2f128 $0x01 , %%ymm6 , %%ymm6 , %%ymm7 \n\t" // [b21,b31,b20,b30]
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm13\n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm3 , %%ymm6 , %%ymm14\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c21,c21,c30,c30]
	                "\n\t"
	                "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a40,a50,a41,a51]
	                "vmovapd  20*8(%[a]), %%ymm1 \n\t" // [a42,a52,a43,a53]
	                "vmovapd   8*8(%[b]), %%ymm4 \n\t" // [b40,b50,b41,b51]
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,b11,b00,b10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm13\n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm14\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm15\n\t" // [c21,c21,c30,c30]
	                "\n\t"
	                "vmovapd  24*8(%[a]), %%ymm2 \n\t" // [a60,a70,a61,a71]
	                "vmovapd  28*8(%[a]), %%ymm3 \n\t" // [a62,a72,a63,a73]
	                "vmovapd  12*8(%[b]), %%ymm6 \n\t" // [b60,b70,b61,b71]
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
	        }
	        if( K & 4 ){
	        //if( K >> 2 ){
	        //    //printf("    K4:\n");
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){
	            //a00 = *(A + 0 + 0*4 ); a10 = *(A + 1 + 0*4 ); a20 = *(A + 2 + 0*4 ); a30 = *(A + 3 + 0*4 ); // ymm0
	            //a01 = *(A + 0 + 1*4 ); a11 = *(A + 1 + 1*4 ); a21 = *(A + 2 + 1*4 ); a31 = *(A + 3 + 1*4 ); // ymm1
	            //a02 = *(A + 0 + 2*4 ); a12 = *(A + 1 + 2*4 ); a22 = *(A + 2 + 2*4 ); a32 = *(A + 3 + 2*4 ); // ymm2
	            //a03 = *(A + 0 + 3*4 ); a13 = *(A + 1 + 3*4 ); a23 = *(A + 2 + 3*4 ); a33 = *(A + 3 + 3*4 ); // ymm3
	            //b00 = *(B + 0 + 0*4 ); b10 = *(B + 1 + 0*4 ); b20 = *(B + 2 + 0*4 ); b30 = *(B + 3 + 0*4 ); // ymm4
	            //b01 = *(B + 0 + 1*4 ); b11 = *(B + 1 + 1*4 ); b21 = *(B + 2 + 1*4 ); b31 = *(B + 3 + 1*4 ); // ymm5
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            //c20 += a02 * b00; c20 += a12 * b10; c20 += a22 * b20; c20 += a32 * b30; // ymm12
	            //c21 += a02 * b01; c21 += a12 * b11; c21 += a22 * b21; c21 += a32 * b31; // ymm13
	            //c30 += a03 * b00; c30 += a13 * b10; c30 += a23 * b20; c30 += a33 * b30; // ymm12
	            //c31 += a03 * b01; c31 += a13 * b11; c31 += a23 * b21; c31 += a33 * b31; // ymm13
	            //A+=16;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vperm2f128 $0x01 , %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,b11,b00,b10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm13\n\t" // [c01,c01,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm14\n\t" // [c20,c20,c31,c31]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm15\n\t" // [c21,c21,c30,c30]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a22,a32,a23,a33]
	                "vmovapd   4*8(%[b]), %%ymm6 \n\t" // [b20,b30,b21,b31]
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

	            //printf("    K2:\n");
	            //a00 = *(A + 0 + 0*2 ); a10 = *(A + 1 + 0*2 );
	            //a01 = *(A + 0 + 1*2 ); a11 = *(A + 1 + 1*2 );
	            //a02 = *(A + 0 + 2*2 ); a12 = *(A + 1 + 2*2 );
	            //a03 = *(A + 0 + 3*2 ); a13 = *(A + 1 + 3*2 );
	            //b00 = *(B + 0 + 0*2 ); b10 = *(B + 1 + 0*2 );
	            //b01 = *(B + 0 + 1*2 ); b11 = *(B + 1 + 1*2 );
	            //c00 += a00 * b00; c00 += a10 * b10;
	            //c01 += a00 * b01; c01 += a10 * b11;
	            //c10 += a01 * b00; c10 += a11 * b10;
	            //c11 += a01 * b01; c11 += a11 * b11;
	            //c20 += a02 * b00; c20 += a12 * b10;
	            //c21 += a02 * b01; c21 += a12 * b11;
	            //c30 += a03 * b00; c30 += a13 * b10;
	            //c31 += a03 * b01; c31 += a13 * b11;
	            //A+=8;
	            //B+=4;
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
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

	            //printf("    K1:\n");
	            //a00 = *(A + 0 + 0*1 );
	            //a01 = *(A + 0 + 1*1 );
	            //a02 = *(A + 0 + 2*1 );
	            //a03 = *(A + 0 + 3*1 );
	            //b00 = *(B + 0 + 0*1 );
	            //b01 = *(B + 0 + 1*1 );
	            //c00 += a00 * b00;
	            //c01 += a00 * b01;
	            //c10 += a01 * b00;
	            //c11 += a01 * b01;
	            //c20 += a02 * b00;
	            //c21 += a02 * b01;
	            //c30 += a03 * b00;
	            //c31 += a03 * b01;
	            //A+=4;
	            //B+=2;

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
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+0+1*ldc) += alpha*c01;
	        //*(C+1+0*ldc) += alpha*c10;
	        //*(C+1+1*ldc) += alpha*c11;
	        //*(C+2+0*ldc) += alpha*c20;
	        //*(C+2+1*ldc) += alpha*c21;
	        //*(C+3+0*ldc) += alpha*c30;
	        //*(C+3+1*ldc) += alpha*c31;
	        ////A = A - K + 4*K;
	        //B = B - 2*K;
	        //C+=4;
	        //////printf("pass through (n&2)&(m>>2)\n");

	            //printf("    END:\n");
	        __asm__ __volatile__ (
	            "\n\t"
	            //"vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c11,c11]
	            //"vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm13\n\t" // [c01,c01,c10,c10]
	            //"vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm14\n\t" // [c20,c20,c31,c31]
	            //"vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm15\n\t" // [c21,c21,c30,c30]
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
	        C+=4;

	      }
	    }
	    if( M & 2 ){

	            //printf("  M2:\n");
	        //c00=0e0;c01=0e0;
	        //c10=0e0;c11=0e0;
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
	            //a00 = *(A + 0 + 0*8 ); a10 = *(A + 1 + 0*8 ); a20 = *(A + 2 + 0*8 ); a30 = *(A + 3 + 0*8 ); // ymm0
	            //a01 = *(A + 0 + 1*8 ); a11 = *(A + 1 + 1*8 ); a21 = *(A + 2 + 1*8 ); a31 = *(A + 3 + 1*8 ); // ymm1
	            //b00 = *(B + 0 + 0*8 ); b10 = *(B + 1 + 0*8 ); b20 = *(B + 2 + 0*8 ); b30 = *(B + 3 + 0*8 ); // ymm4
	            //b01 = *(B + 0 + 1*8 ); b11 = *(B + 1 + 1*8 ); b21 = *(B + 2 + 1*8 ); b31 = *(B + 3 + 1*8 ); // ymm5
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            //a00 = *(A + 4 + 0*8 ); a10 = *(A + 5 + 0*8 ); a20 = *(A + 6 + 0*8 ); a30 = *(A + 7 + 0*8 ); // ymm0
	            //a01 = *(A + 4 + 1*8 ); a11 = *(A + 5 + 1*8 ); a21 = *(A + 6 + 1*8 ); a31 = *(A + 7 + 1*8 ); // ymm1
	            //b00 = *(B + 4 + 0*8 ); b10 = *(B + 5 + 0*8 ); b20 = *(B + 6 + 0*8 ); b30 = *(B + 7 + 0*8 ); // ymm4
	            //b01 = *(B + 4 + 1*8 ); b11 = *(B + 5 + 1*8 ); b21 = *(B + 6 + 1*8 ); b31 = *(B + 7 + 1*8 ); // ymm5
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            //A+=16;
	            //B+=16;
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm15\n\t" // [c10,c10,c01,c01]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd   4*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
	                "vperm2f128 $0x01 , %%ymm2 , %%ymm2 , %%ymm3 \n\t" // [a21,a31,a20,a30]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm14\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm15\n\t" // [c10,c10,c01,c01]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm8 \n\t" // [a40,a50,a41,a51]
	                "vmovapd   8*8(%[b]), %%ymm6 \n\t" // [b40,b50,b41,b51]
	                "vperm2f128 $0x01 , %%ymm8 , %%ymm8 , %%ymm9 \n\t" // [a01,a11,a00,a10]
	                "vfmadd231pd  %%ymm8 , %%ymm6 , %%ymm14\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm9 , %%ymm6 , %%ymm15\n\t" // [c10,c10,c01,c01]
	                "\n\t"
	                "vmovapd  12*8(%[a]), %%ymm10\n\t" // [a60,a70,a61,a71]
	                "vmovapd  12*8(%[b]), %%ymm7 \n\t" // [b60,b70,b61,b71]
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
	        //    //printf("    K4:\n");
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){
	            //a00 = *(A + 0 + 0*4 ); a10 = *(A + 1 + 0*4 ); a20 = *(A + 2 + 0*4 ); a30 = *(A + 3 + 0*4 ); // ymm0
	            //a01 = *(A + 0 + 1*4 ); a11 = *(A + 1 + 1*4 ); a21 = *(A + 2 + 1*4 ); a31 = *(A + 3 + 1*4 ); // ymm1
	            //b00 = *(B + 0 + 0*4 ); b10 = *(B + 1 + 0*4 ); b20 = *(B + 2 + 0*4 ); b30 = *(B + 3 + 0*4 ); // ymm4
	            //b01 = *(B + 0 + 1*4 ); b11 = *(B + 1 + 1*4 ); b21 = *(B + 2 + 1*4 ); b31 = *(B + 3 + 1*4 ); // ymm5
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            //A+=8;
	            //B+=8;
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vperm2f128 $0x01 , %%ymm0 , %%ymm0 , %%ymm1 \n\t" // [a01,a11,a00,a10]
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t" // [c00,c00,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm15\n\t" // [c10,c10,c01,c01]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd   4*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
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

	            //printf("    K2:\n");
	            //a00 = *(A + 0 + 0*2 ); a10 = *(A + 1 + 0*2 );
	            //a01 = *(A + 0 + 1*2 ); a11 = *(A + 1 + 1*2 );
	            //b00 = *(B + 0 + 0*2 ); b10 = *(B + 1 + 0*2 );
	            //b01 = *(B + 0 + 1*2 ); b11 = *(B + 1 + 1*2 );
	            //c00 += a00 * b00; c00 += a10 * b10;
	            //c01 += a00 * b01; c01 += a10 * b11;
	            //c10 += a01 * b00; c10 += a11 * b10;
	            //c11 += a01 * b01; c11 += a11 * b11;
	            //A+=4;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
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

	            //printf("    K1:\n");
	            //a00 = *(A + 0 + 0*1 );
	            //a01 = *(A + 0 + 1*1 );
	            //b00 = *(B + 0 + 0*1 );
	            //b01 = *(B + 0 + 1*1 );
	            //c00 += a00 * b00;
	            //c01 += a00 * b01;
	            //c10 += a01 * b00;
	            //c11 += a01 * b01;
	            //A+=2;
	            //B+=2;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[a]), %%xmm0 \n\t" // [a00,  0]
	                "vmovsd   1*8(%[a]), %%xmm1 \n\t" // [a01,  0]
	                "vmovsd   0*8(%[b]), %%xmm4 \n\t" // [b00,  0]
	                "vmovsd   1*8(%[b]), %%xmm5 \n\t" // [b01,  0]
	                "\n\t"
	                "vperm2f128  $0x20, %%ymm1 , %%ymm0 , %%ymm0 \n\t" // [a00,  0,a01,  0]
	                "vperm2f128  $0x20, %%ymm5 , %%ymm4 , %%ymm4 \n\t" // [b00,  0,b01,  0]
	                "vperm2f128  $0x01, %%ymm4 , %%ymm4 , %%ymm5 \n\t" // [b01,  0,b00,  0]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t" // [c00,  0,c11,  0]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm15\n\t" // [c01,  0,c10,  0]
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+0+1*ldc) += alpha*c01;
	        //*(C+1+0*ldc) += alpha*c10;
	        //*(C+1+1*ldc) += alpha*c11;
	        ////A = A - K + 4*K;
	        //B = B - 2*K;
	        //C+=2;
	        //////printf("pass through (n&2)&(m&2)\n");

	            //printf("    END:\n");
	        __asm__ __volatile__ (
	            "\n\t"
	            //"vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t" // [c00,c00,c11,c11]
	            //"vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm15\n\t" // [c10,c10,c01,c01]
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
	        C+=2;

	    }
	    if( M & 1 ){

	            //printf("  M1:\n");
	        //c00=0e0;c01=0e0;
	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*8 ); a10 = *(A + 1 + 0*8 ); a20 = *(A + 2 + 0*8 ); a30 = *(A + 3 + 0*8 ); // ymm0
	            //b00 = *(B + 0 + 0*8 ); b10 = *(B + 1 + 0*8 ); b20 = *(B + 2 + 0*8 ); b30 = *(B + 3 + 0*8 ); // ymm4
	            //b01 = *(B + 0 + 1*8 ); b11 = *(B + 1 + 1*8 ); b21 = *(B + 2 + 1*8 ); b31 = *(B + 3 + 1*8 ); // ymm5
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //a00 = *(A + 4 + 0*8 ); a10 = *(A + 5 + 0*8 ); a20 = *(A + 6 + 0*8 ); a30 = *(A + 7 + 0*8 ); // ymm0
	            //b00 = *(B + 4 + 0*8 ); b10 = *(B + 5 + 0*8 ); b20 = *(B + 6 + 0*8 ); b30 = *(B + 7 + 0*8 ); // ymm4
	            //b01 = *(B + 4 + 1*8 ); b11 = *(B + 5 + 1*8 ); b21 = *(B + 6 + 1*8 ); b31 = *(B + 7 + 1*8 ); // ymm5
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //A+=8;
	            //B+=16;
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovapd   4*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
	                "vperm2f128 $0x20 , %%ymm5 , %%ymm4 , %%ymm8 \n\t" // [b00,b10,b20,b30]
	                "vperm2f128 $0x31 , %%ymm5 , %%ymm4 , %%ymm9 \n\t" // [b01,b11,b21,b31]
	                "vfmadd231pd  %%ymm0 , %%ymm8 , %%ymm14\n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm0 , %%ymm9 , %%ymm15\n\t" // [c01,c01,c01,c01]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a40,a50,a60,a70]
	                "vmovapd   8*8(%[b]), %%ymm6 \n\t" // [b40,b50,b41,b51]
	                "vmovapd  12*8(%[b]), %%ymm7 \n\t" // [b60,b70,b61,b71]
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
	        //    //printf("    K4:\n");
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){
	            //a00 = *(A + 0 + 0*4 ); a10 = *(A + 1 + 0*4 ); a20 = *(A + 2 + 0*4 ); a30 = *(A + 3 + 0*4 ); // ymm0
	            //b00 = *(B + 0 + 0*4 ); b10 = *(B + 1 + 0*4 ); b20 = *(B + 2 + 0*4 ); b30 = *(B + 3 + 0*4 ); // ymm4
	            //b01 = *(B + 0 + 1*4 ); b11 = *(B + 1 + 1*4 ); b21 = *(B + 2 + 1*4 ); b31 = *(B + 3 + 1*4 ); // ymm5
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //A+=4;
	            //B+=8;
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b01,b11]
	                "vmovapd   4*8(%[b]), %%ymm5 \n\t" // [b20,b30,b21,b31]
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

	            //a00 = *(A + 0 + 0*2 ); a10 = *(A + 1 + 0*2 );
	            //b00 = *(B + 0 + 0*2 ); b10 = *(B + 1 + 0*2 );
	            //b01 = *(B + 0 + 1*2 ); b11 = *(B + 1 + 1*2 );
	            //c00 += a00 * b00; c00 += a10 * b10;
	            //c01 += a00 * b01; c01 += a10 * b11;
	            //A+=2;
	            //B+=4;
	            //printf("    K2:\n");
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

	            //printf("    K1:\n");
	            //a00 = *(A + 0 + 0*1 );
	            //b00 = *(B + 0 + 0*1 );
	            //b01 = *(B + 0 + 1*1 );
	            //c00 += a00 * b00;
	            //c01 += a00 * b01;
	            //A+=1;
	            //B+=2;

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
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+0+1*ldc) += alpha*c01;
	        ////A = A - K + 4*K;
	        //B = B - 2*K;
	        //C+=1;
	        //////printf("pass through (n&2)&(m&1)\n");

	            //printf("    END:\n");
	        __asm__ __volatile__ (
	            "\n\t"
	            //"vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t" // [c00,c00,c00,c00]
	            //"vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm15\n\t" // [c01,c01,c01,c01]
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
	        C+=1;
	    }
	    A = A - M*K;
	    B = B + 2*K;
	    C  = C - M + 2*ldc;
	    c0 = c0- M + 2*ldc;
	    c1 = c1- M + 2*ldc;

	}
	if( N & 1 ){
	            //printf("N1:\n");

	    if( M >> 2 ){
	            //printf("  M4:\n");
	      size_t m4 = ( M >> 2 ); // unrolling M
	      while( m4-- ){
	        //AB=0e0;
	        //c00=0e0;
	        //c10=0e0;
	        //c20=0e0;
	        //c30=0e0;
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
	            //a00 = *(A + 0 + 0*8 ); a10 = *(A + 1 + 0*8 ); a20 = *(A + 2 + 0*8 ); a30 = *(A + 3 + 0*8 ); // ymm0
	            //a01 = *(A + 0 + 1*8 ); a11 = *(A + 1 + 1*8 ); a21 = *(A + 2 + 1*8 ); a31 = *(A + 3 + 1*8 ); // ymm1
	            //a02 = *(A + 0 + 2*8 ); a12 = *(A + 1 + 2*8 ); a22 = *(A + 2 + 2*8 ); a32 = *(A + 3 + 2*8 ); // ymm2
	            //a03 = *(A + 0 + 3*8 ); a13 = *(A + 1 + 3*8 ); a23 = *(A + 2 + 3*8 ); a33 = *(A + 3 + 3*8 ); // ymm3
	            //b00 = *(B + 0 + 0*8 ); b10 = *(B + 1 + 0*8 ); b20 = *(B + 2 + 0*8 ); b30 = *(B + 3 + 0*8 ); // ymm4
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //c20 += a02 * b00; c20 += a12 * b10; c20 += a22 * b20; c20 += a32 * b30; // ymm12
	            //c30 += a03 * b00; c30 += a13 * b10; c30 += a23 * b20; c30 += a33 * b30; // ymm12
	            //a00 = *(A + 4 + 0*8 ); a10 = *(A + 5 + 0*8 ); a20 = *(A + 6 + 0*8 ); a30 = *(A + 7 + 0*8 ); // ymm0
	            //a01 = *(A + 4 + 1*8 ); a11 = *(A + 5 + 1*8 ); a21 = *(A + 6 + 1*8 ); a31 = *(A + 7 + 1*8 ); // ymm1
	            //a02 = *(A + 4 + 2*8 ); a12 = *(A + 5 + 2*8 ); a22 = *(A + 6 + 2*8 ); a32 = *(A + 7 + 2*8 ); // ymm2
	            //a03 = *(A + 4 + 3*8 ); a13 = *(A + 5 + 3*8 ); a23 = *(A + 6 + 3*8 ); a33 = *(A + 7 + 3*8 ); // ymm3
	            //b00 = *(B + 4 + 0*8 ); b10 = *(B + 5 + 0*8 ); b20 = *(B + 6 + 0*8 ); b30 = *(B + 7 + 0*8 ); // ymm4
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //c20 += a02 * b00; c20 += a12 * b10; c20 += a22 * b20; c20 += a32 * b30; // ymm12
	            //c30 += a03 * b00; c30 += a13 * b10; c30 += a23 * b20; c30 += a33 * b30; // ymm12
	            //A+=32;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a22,a32,a23,a33]
	                "vmovapd   0*8(%[b]), %%ymm8 \n\t" // [b00,b10,b20,b30]
	                "vperm2f128 $0x00 , %%ymm8 , %%ymm8 , %%ymm10\n\t" // [b00,b10,b00,b10]
	                "vperm2f128 $0x11 , %%ymm8 , %%ymm8 , %%ymm11\n\t" // [b20,b30,b20,b30]
	                "vfmadd231pd  %%ymm0 , %%ymm10, %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm10, %%ymm15\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm2 , %%ymm11, %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm3 , %%ymm11, %%ymm15\n\t" // [c20,c20,c30,c30]
	                "\n\t"
	                "vmovapd  16*8(%[a]), %%ymm4 \n\t" // [a40,a50,a41,a51]
	                "vmovapd  20*8(%[a]), %%ymm5 \n\t" // [a42,a52,a43,a53]
	                "vmovapd  24*8(%[a]), %%ymm6 \n\t" // [a60,a70,a61,a71]
	                "vmovapd  28*8(%[a]), %%ymm7 \n\t" // [a62,a72,a63,a73]
	                "vmovapd   4*8(%[b]), %%ymm9 \n\t" // [b40,b50,b60,b70]
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
	        //    //printf("    K4:\n");
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){
	            //a00 = *(A + 0 + 0*4 ); a10 = *(A + 1 + 0*4 ); a20 = *(A + 2 + 0*4 ); a30 = *(A + 3 + 0*4 ); // ymm0
	            //a01 = *(A + 0 + 1*4 ); a11 = *(A + 1 + 1*4 ); a21 = *(A + 2 + 1*4 ); a31 = *(A + 3 + 1*4 ); // ymm1
	            //a02 = *(A + 0 + 2*4 ); a12 = *(A + 1 + 2*4 ); a22 = *(A + 2 + 2*4 ); a32 = *(A + 3 + 2*4 ); // ymm2
	            //a03 = *(A + 0 + 3*4 ); a13 = *(A + 1 + 3*4 ); a23 = *(A + 2 + 3*4 ); a33 = *(A + 3 + 3*4 ); // ymm3
	            //b00 = *(B + 0 + 0*4 ); b10 = *(B + 1 + 0*4 ); b20 = *(B + 2 + 0*4 ); b30 = *(B + 3 + 0*4 ); // ymm4
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //c20 += a02 * b00; c20 += a12 * b10; c20 += a22 * b20; c20 += a32 * b30; // ymm12
	            //c30 += a03 * b00; c30 += a13 * b10; c30 += a23 * b20; c30 += a33 * b30; // ymm12
	            //A+=16;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a22,a32,a23,a33]
	                "vmovapd   0*8(%[b]), %%ymm8 \n\t" // [b00,b10,b20,b30]
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

	            //a00 = *(A + 0 + 0*2 ); a10 = *(A + 1 + 0*2 );
	            //a01 = *(A + 0 + 1*2 ); a11 = *(A + 1 + 1*2 );
	            //a02 = *(A + 0 + 2*2 ); a12 = *(A + 1 + 2*2 );
	            //a03 = *(A + 0 + 3*2 ); a13 = *(A + 1 + 3*2 );
	            //b00 = *(B + 0 + 0*2 ); b10 = *(B + 1 + 0*2 );
	            //c00 += a00 * b00; c00 += a10 * b10;
	            //c10 += a01 * b00; c10 += a11 * b10;
	            //c20 += a02 * b00; c20 += a12 * b10;
	            //c30 += a03 * b00; c30 += a13 * b10;
	            //A+=8;
	            //B+=2;
	            //printf("    K2:\n");

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

	            //printf("    K1:\n");
	            //a00 = *(A + 0 + 0*1 );
	            //a01 = *(A + 0 + 1*1 );
	            //a02 = *(A + 0 + 2*1 );
	            //a03 = *(A + 0 + 3*1 );
	            //b00 = *(B + 0 + 0*1 );
	            //c00 += a00 * b00;
	            //c10 += a01 * b00;
	            //c20 += a02 * b00;
	            //c30 += a03 * b00;
	            //A+=4;
	            //B+=1;
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
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+1+0*ldc) += alpha*c10;
	        //*(C+2+0*ldc) += alpha*c20;
	        //*(C+3+0*ldc) += alpha*c30;
	        ////A = A - K + 4*K;
	        //B = B - 1*K;
	        //C+=4;
	        //////printf("pass through (n&1)&(m>>2)\n");
	            //printf("    END:\n");
	        __asm__ __volatile__ (
	            "\n\t"
	            //"vfmadd231pd  %%ymm0 , %%ymm10, %%ymm14\n\t" // [c00,c00,c10,c10]
	            //"vfmadd231pd  %%ymm1 , %%ymm10, %%ymm15\n\t" // [c20,c20,c30,c30]
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
	            "vmovupd  %%ymm14, 0*8(%[c0]        )\n\t"
	            "\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "addq  $4*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 1*K;
	        C+=4;

	      }
	    }
	    if( M & 2 ){

	            //printf("  M2:\n");
	        //c00=0e0;
	        //c10=0e0;
	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*8 ); a10 = *(A + 1 + 0*8 ); a20 = *(A + 2 + 0*8 ); a30 = *(A + 3 + 0*8 ); // ymm0
	            //a01 = *(A + 0 + 1*8 ); a11 = *(A + 1 + 1*8 ); a21 = *(A + 2 + 1*8 ); a31 = *(A + 3 + 1*8 ); // ymm1
	            //b00 = *(B + 0 + 0*8 ); b10 = *(B + 1 + 0*8 ); b20 = *(B + 2 + 0*8 ); b30 = *(B + 3 + 0*8 ); // ymm4
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //a00 = *(A + 4 + 0*8 ); a10 = *(A + 5 + 0*8 ); a20 = *(A + 6 + 0*8 ); a30 = *(A + 7 + 0*8 ); // ymm0
	            //b00 = *(B + 4 + 0*8 ); b10 = *(B + 5 + 0*8 ); b20 = *(B + 6 + 0*8 ); b30 = *(B + 7 + 0*8 ); // ymm4
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //A+=16;
	            //B+=8;
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd   0*8(%[b]), %%ymm8 \n\t" // [b00,b10,b20,b30]
	                "vperm2f128 $0x00 , %%ymm8 , %%ymm8 , %%ymm10\n\t" // [b00,b10,b00,b10]
	                "vperm2f128 $0x11 , %%ymm8 , %%ymm8 , %%ymm11\n\t" // [b20,b30,b20,b30]
	                "vfmadd231pd  %%ymm0 , %%ymm10, %%ymm15\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm2 , %%ymm11, %%ymm15\n\t" // [c00,c00,c10,c10]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm4 \n\t" // [a40,a50,a41,a51]
	                "vmovapd  12*8(%[a]), %%ymm6 \n\t" // [a60,a70,a61,a71]
	                "vmovapd   4*8(%[b]), %%ymm9 \n\t" // [b40,b50,b60,b70]
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
	        //    //printf("    K4:\n");
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){
	            //a00 = *(A + 0 + 0*4 ); a10 = *(A + 1 + 0*4 ); a20 = *(A + 2 + 0*4 ); a30 = *(A + 3 + 0*4 ); // ymm0
	            //a01 = *(A + 0 + 1*4 ); a11 = *(A + 1 + 1*4 ); a21 = *(A + 2 + 1*4 ); a31 = *(A + 3 + 1*4 ); // ymm1
	            //b00 = *(B + 0 + 0*4 ); b10 = *(B + 1 + 0*4 ); b20 = *(B + 2 + 0*4 ); b30 = *(B + 3 + 0*4 ); // ymm4
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //A+=8;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd   0*8(%[b]), %%ymm8 \n\t" // [b00,b10,b20,b30]
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

	            //a00 = *(A + 0 + 0*2 ); a10 = *(A + 1 + 0*2 );
	            //a01 = *(A + 0 + 1*2 ); a11 = *(A + 1 + 1*2 );
	            //b00 = *(B + 0 + 0*2 ); b10 = *(B + 1 + 0*2 );
	            //c00 += a00 * b00; c00 += a10 * b10;
	            //c10 += a01 * b00; c10 += a11 * b10;
	            //A+=4;
	            //B+=2;

	            //printf("    K2:\n");
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

	            //a00 = *(A + 0 + 0*1 );
	            //a01 = *(A + 0 + 1*1 );
	            //b00 = *(B + 0 + 0*1 );
	            //c00 += a00 * b00;
	            //c10 += a01 * b00;
	            //A+=2;
	            //B+=1;

	            //printf("    K1:\n");
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
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $1*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+1+0*ldc) += alpha*c10;
	        ////A = A - K + 4*K;
	        //B = B - 1*K;
	        //C+=2;
	        //////printf("pass through (n&1)&(m&2)\n");
	            //printf("    END:\n");

	        __asm__ __volatile__ (
	            "\n\t"
	            //"vfmadd231pd  %%ymm0 , %%ymm10, %%ymm15\n\t" // [c00,c00,c10,c10]
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vperm2f128 $0x10 , %%ymm15, %%ymm15, %%ymm14\n\t" // [c10,c10,c00,c00] // 4% slow down
	            "vshufpd    $0x00 , %%ymm14, %%ymm15, %%ymm12\n\t" // [c00,c10,---,---] // 2% slow down
	            "vshufpd    $0x0f , %%ymm14, %%ymm15, %%ymm13\n\t" // [c00,c10,---,---] // 2% slow down
	            "vaddpd             %%ymm10, %%ymm11, %%ymm15\n\t" // [c00,c10,---,---]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]        ), %%xmm0, %%xmm15\n\t"
	            "\n\t"
	            "movupd  %%xmm15, 0*8(%[c0]        )\n\t"
	            "\n\t"
	            "addq  $2*8, %[c0]\n\t"
	            "addq  $2*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 1*K;
	        C+=2;

	    }
	    if( M & 1 ){

	            //printf("  M1:\n");
	        //c00=0e0;
	        __asm__ __volatile__ (
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*8 ); a10 = *(A + 1 + 0*8 ); a20 = *(A + 2 + 0*8 ); a30 = *(A + 3 + 0*8 ); // ymm0
	            //b00 = *(B + 0 + 0*8 ); b10 = *(B + 1 + 0*8 ); b20 = *(B + 2 + 0*8 ); b30 = *(B + 3 + 0*8 ); // ymm4
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //a00 = *(A + 4 + 0*8 ); a10 = *(A + 5 + 0*8 ); a20 = *(A + 6 + 0*8 ); a30 = *(A + 7 + 0*8 ); // ymm0
	            //b00 = *(B + 4 + 0*8 ); b10 = *(B + 5 + 0*8 ); b20 = *(B + 6 + 0*8 ); b30 = *(B + 7 + 0*8 ); // ymm4
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //A+=8;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   0*8(%[b]), %%ymm2 \n\t" // [b00,b10,b20,b30]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm15\n\t" // [c00,c00,c00,c00]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a40,a50,a60,a70]
	                "vmovapd   4*8(%[b]), %%ymm3 \n\t" // [b40,b50,b60,b70]
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
	        //    //printf("    K4:\n");
	        //  size_t k4 = ( K >> 2 ); // Unrolling K
	        //  while( k4-- ){
	            //a00 = *(A + 0 + 0*4 ); a10 = *(A + 1 + 0*4 ); a20 = *(A + 2 + 0*4 ); a30 = *(A + 3 + 0*4 ); // ymm0
	            //b00 = *(B + 0 + 0*4 ); b10 = *(B + 1 + 0*4 ); b20 = *(B + 2 + 0*4 ); b30 = *(B + 3 + 0*4 ); // ymm4
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //A+=4;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   0*8(%[b]), %%ymm2 \n\t" // [b00,b10,b20,b30]
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

	            //a00 = *(A + 0 + 0*2 ); a10 = *(A + 1 + 0*2 );
	            //b00 = *(B + 0 + 0*2 ); b10 = *(B + 1 + 0*2 );
	            //c00 += a00 * b00; c00 += a10 * b10;
	            //A+=2;
	            //B+=2;
	            //printf("    K2:\n");

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

	            //a00 = *(A + 0 + 0*1 );
	            //b00 = *(B + 0 + 0*1 );
	            //c00 += a00 * b00;
	            //A+=1;
	            //B+=1;
	            //printf("    K1:\n");

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
	        //*(C+0+0*ldc) += alpha*c00;
	        ////A = A - K + 4*K;
	        //B = B - 1*K;
	        //C+=1;
	            //printf("    END:\n");

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
	        C+=1;

	    }
	    A = A - M*K;
	    B = B + 1*K;
	    C  = C - M + 1*ldc;
	    c0 = c0- M + 1*ldc;
	    c1 = c1- M + 1*ldc;
	        ////printf("pass through (n&1)&(m&1)\n");

	}

	//if( N & 3 ){
	//  size_t nr = ( N & 3 ); // unrolling N
	//  while( nr-- ){
	//    size_t m = M;
	//    while( m-- ){
	//      c00=0e0;
	//      size_t k = K;
	//      while( k-- ){
	//        c00 = c00 + (*A)*(*B);
	//        A++;
	//        B++;
	//      }
	//      *C = (*C) + alpha*c00;
	//      B = B - K;
	//      C++;
	//    }
	//    A = A - M*K;
	//    B = B + K;
	//    C  = C - M + ldc;
	//  }
	//}

	A = A + M*K;
	B = B - K*N;
	C  = C - ldc*N + M;
	c0 = c0- ldc*N + M;
	c1 = c1- ldc*N + M;
	// ---- Kernel


}

