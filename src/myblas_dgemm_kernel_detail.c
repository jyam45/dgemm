#include "myblas_internal.h"
#include <stdio.h>

void myblas_dgemm_kernel_detail(
         size_t M, size_t N, size_t K,
         double alpha, const double *A, const double *B, 
         double *C, size_t ldc )
{
	//double a00,a01,a02,a03;
	//double a10,a11,a12,a13;
	//double a20,a21,a22,a23;
	//double a30,a31,a32,a33;

	//double b00,b01,b02,b03;
	//double b10,b11,b12,b13;
	//double b20,b21,b22,b23;
	//double b30,b31,b32,b33;

	//double c00,c01,c02,c03;
	//double c10,c11,c12,c13;
	//double c20,c21,c22,c23;
	//double c30,c31,c32,c33;

	double *c0 = C;
	double *c1 = C + ldc;
	size_t ldc2 = ldc * 2 * sizeof(double);
	double alpha4[4] = {alpha,alpha,alpha,alpha};

	size_t N3Q = N / 3;
	size_t N3R = N % 3;

	            //printf("START:\n");
	// Kernel ----
	if( N3Q ){
	            //printf("N4:\n");
	  size_t n3 = N3Q; // unrolling N
	  while( n3-- ){
	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 ); // unrolling M
	      while( m4-- ){

	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        //c10=0e0;c11=0e0;c12=0e0;c13=0e0;
	        //c20=0e0;c21=0e0;c22=0e0;c23=0e0;
	        //c30=0e0;c31=0e0;c32=0e0;c33=0e0;

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

	            // 4x3x4  LD/FMA = 7/12 = 0.58
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[b]), %%ymm1 \n\t" // [b00,b10,b20,b30]
	                "vmovapd   4*8(%[b]), %%ymm2 \n\t" // [b01,b11,b21,b31]
	                "vmovapd   8*8(%[b]), %%ymm3 \n\t" // [b02,b12,b22,b32]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c01,c01,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c02,c02,c02,c02]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%ymm0 \n\t" // [a01,a11,a21,a31]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c10,c10,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c11,c11,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c12,c12,c12,c12]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm0 \n\t" // [a02,a12,a22,a32]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c20,c20,c20,c20]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c21,c21,c21,c21]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c22,c22,c22,c22]
	                "\n\t"
	                "vmovapd  12*8(%[a]), %%ymm0 \n\t" // [a03,a13,a23,a33]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c30,c30,c30,c30]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c31,c31,c31,c31]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c32,c32,c32,c32]
	                "\n\t"
	                "vmovapd   0*8(%[b]), %%ymm1 \n\t" // [b40,b50,b60,b70]
	                "vmovapd   4*8(%[b]), %%ymm2 \n\t" // [b41,b51,b61,b71]
	                "vmovapd   8*8(%[b]), %%ymm3 \n\t" // [b42,b52,b62,b72]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a40,a50,a60,a70]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c01,c01,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c02,c02,c02,c02]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%ymm0 \n\t" // [a41,a51,a61,a71]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c10,c10,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c11,c11,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c12,c12,c12,c12]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm0 \n\t" // [a42,a52,a62,a72]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c20,c20,c20,c20]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c21,c21,c21,c21]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c22,c22,c22,c22]
	                "\n\t"
	                "vmovapd  12*8(%[a]), %%ymm0 \n\t" // [a43,a53,a63,a73]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c30,c30,c30,c30]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c31,c31,c31,c31]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c32,c32,c32,c32]
	                "\n\t"
	                "addq  $32*8 , %[a]\n\t"
	                "addq  $24*8 , %[b]\n\t"
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

	            // 4x3x4  LD/FMA = 7/12 = 0.58
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[b]), %%ymm1 \n\t" // [b00,b10,b20,b30]
	                "vmovapd   4*8(%[b]), %%ymm2 \n\t" // [b01,b11,b21,b31]
	                "vmovapd   8*8(%[b]), %%ymm3 \n\t" // [b02,b12,b22,b32]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c01,c01,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c02,c02,c02,c02]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%ymm0 \n\t" // [a01,a11,a21,a31]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c10,c10,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c11,c11,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c12,c12,c12,c12]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm0 \n\t" // [a02,a12,a22,a32]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c20,c20,c20,c20]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c21,c21,c21,c21]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c22,c22,c22,c22]
	                "\n\t"
	                "vmovapd  12*8(%[a]), %%ymm0 \n\t" // [a03,a13,a23,a33]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c30,c30,c30,c30]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c31,c31,c31,c31]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c32,c32,c32,c32]
	                "\n\t"
	                "addq  $16*8 , %[a]\n\t"
	                "addq  $12*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	            //// 3x3x4  LD/FMA = 6/9 = 0.66
	            //__asm__ __volatile__ (
	            //    "\n\t"
	            //    "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	            //    "vmovapd   4*8(%[b]), %%ymm5 \n\t" // [b01,b11,b21,b31]
	            //    "vmovapd   8*8(%[b]), %%ymm6 \n\t" // [b02,b12,b22,b32]
	            //    "\n\t"
	            //    "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	            //    "\n\t"
	            //    "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm7 \n\t" // [c00,c00,c00,c00]
	            //    "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm8 \n\t" // [c01,c01,c01,c01]
	            //    "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm9 \n\t" // [c02,c02,c02,c02]
	            //    "\n\t"
	            //    "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	            //    "\n\t"
	            //    "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm10\n\t" // [c10,c10,c10,c10]
	            //    "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm11\n\t" // [c11,c11,c11,c11]
	            //    "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm12\n\t" // [c12,c12,c12,c12]
	            //    "\n\t"
	            //    "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a02,a12,a22,a32]
	            //    "\n\t"
	            //    "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm13\n\t" // [c20,c20,c20,c20]
	            //    "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm14\n\t" // [c21,c21,c21,c21]
	            //    "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm15\n\t" // [c22,c22,c22,c22]
	            //    "\n\t"
	            //    "addq  $12*8 , %[a]\n\t"
	            //    "addq  $12*8 , %[b]\n\t"
	            //    "\n\t"
	            //    "\n\t"
	            //    :[a]"+r"(A),[b]"+r"(B)
	            //:);


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

	            // 4x3x4  LD/FMA = 7/12 = 0.58
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[b]), %%xmm1 \n\t" // [b00,b10,  0,  0]
	                "vmovapd   2*8(%[b]), %%xmm2 \n\t" // [b01,b11,  0,  0]
	                "vmovapd   4*8(%[b]), %%xmm3 \n\t" // [b02,b12,  0,  0]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c01,c01,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c02,c02,c02,c02]
	                "\n\t"
	                "vmovapd   2*8(%[a]), %%xmm0 \n\t" // [a01,a11,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c10,c10,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c11,c11,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c12,c12,c12,c12]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%xmm0 \n\t" // [a02,a12,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c20,c20,c20,c20]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c21,c21,c21,c21]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c22,c22,c22,c22]
	                "\n\t"
	                "vmovapd   6*8(%[a]), %%xmm0 \n\t" // [a03,a13,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c30,c30,c30,c30]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c31,c31,c31,c31]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c32,c32,c32,c32]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $6*8 , %[b]\n\t"
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

	            // 4x3x4  LD/FMA = 7/12 = 0.58
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[b]), %%xmm1 \n\t" // [b00,  0,  0,  0]
	                "vmovsd   1*8(%[b]), %%xmm2 \n\t" // [b01,  0,  0,  0]
	                "vmovsd   2*8(%[b]), %%xmm3 \n\t" // [b02,  0,  0,  0]
	                "\n\t"
	                "vmovsd   0*8(%[a]), %%xmm0 \n\t" // [a00,  0,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c00,c00,c00]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c01,c01,c01,c01]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c02,c02,c02,c02]
	                "\n\t"
	                "vmovsd   1*8(%[a]), %%xmm0 \n\t" // [a01,  0,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c10,c10,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c11,c11,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c12,c12,c12,c12]
	                "\n\t"
	                "vmovsd   2*8(%[a]), %%xmm0 \n\t" // [a02,  0,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c20,c20,c20,c20]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c21,c21,c21,c21]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c22,c22,c22,c22]
	                "\n\t"
	                "vmovsd   3*8(%[a]), %%xmm0 \n\t" // [a03,  0,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c30,c30,c30,c30]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c31,c31,c31,c31]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c32,c32,c32,c32]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $3*8 , %[b]\n\t"
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
	            "vshufpd    $0x00, %%ymm7 , %%ymm4 , %%ymm0 \n\t" // [c00,c10,c00,c10]
	            "vshufpd    $0x0f, %%ymm7 , %%ymm4 , %%ymm1 \n\t" // [c00,c10,c00,c10]
	            "vshufpd    $0x00, %%ymm13, %%ymm10, %%ymm2 \n\t" // [c20,c30,c20,c30]
	            "vshufpd    $0x0f, %%ymm13, %%ymm10, %%ymm3 \n\t" // [c20,c30,c20,c30]
	            "vaddpd            %%ymm1 , %%ymm0 , %%ymm0 \n\t" // [c00,c10,c00,c10]
	            "vaddpd            %%ymm3 , %%ymm2 , %%ymm2 \n\t" // [c20,c30,c20,c30]
	            "vperm2f128 $0x20, %%ymm2 , %%ymm0 , %%ymm1 \n\t" // [c00,c10,c20,c30]
	            "vperm2f128 $0x31, %%ymm2 , %%ymm0 , %%ymm3 \n\t" // [c00,c10,c20,c30]
	            "vaddpd            %%ymm3 , %%ymm1 , %%ymm13\n\t" // [c00,c10,c20,c30]
	            "\n\t"
	            "vshufpd    $0x00, %%ymm8 , %%ymm5 , %%ymm0 \n\t" // [c01,c11,c01,c11]
	            "vshufpd    $0x0f, %%ymm8 , %%ymm5 , %%ymm1 \n\t" // [c01,c11,c01,c11]
	            "vshufpd    $0x00, %%ymm14, %%ymm11, %%ymm2 \n\t" // [c21,c31,c21,c31]
	            "vshufpd    $0x0f, %%ymm14, %%ymm11, %%ymm3 \n\t" // [c21,c31,c21,c31]
	            "vaddpd            %%ymm1 , %%ymm0 , %%ymm0 \n\t" // [c01,c11,c01,c11]
	            "vaddpd            %%ymm3 , %%ymm2 , %%ymm2 \n\t" // [c21,c31,c21,c31]
	            "vperm2f128 $0x20, %%ymm2 , %%ymm0 , %%ymm1 \n\t" // [c01,c11,c21,c31]
	            "vperm2f128 $0x31, %%ymm2 , %%ymm0 , %%ymm3 \n\t" // [c01,c11,c21,c31]
	            "vaddpd            %%ymm3 , %%ymm1 , %%ymm14\n\t" // [c01,c11,c21,c31]
	            "\n\t"
	            "vshufpd    $0x00, %%ymm9 , %%ymm6 , %%ymm0 \n\t" // [c02,c12,c02,c12]
	            "vshufpd    $0x0f, %%ymm9 , %%ymm6 , %%ymm1 \n\t" // [c02,c12,c02,c12]
	            "vshufpd    $0x00, %%ymm15, %%ymm12, %%ymm2 \n\t" // [c22,c32,c22,c32]
	            "vshufpd    $0x0f, %%ymm15, %%ymm12, %%ymm3 \n\t" // [c22,c32,c22,c32]
	            "vaddpd            %%ymm1 , %%ymm0 , %%ymm0 \n\t" // [c02,c12,c02,c12]
	            "vaddpd            %%ymm3 , %%ymm2 , %%ymm2 \n\t" // [c22,c32,c22,c32]
	            "vperm2f128 $0x20, %%ymm2 , %%ymm0 , %%ymm1 \n\t" // [c02,c12,c22,c32]
	            "vperm2f128 $0x31, %%ymm2 , %%ymm0 , %%ymm3 \n\t" // [c02,c12,c22,c32]
	            "vaddpd            %%ymm3 , %%ymm1 , %%ymm15\n\t" // [c02,c12,c22,c32]
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]        ), %%ymm0, %%ymm13\n\t"
	            "vfmadd213pd 0*8(%[c1]        ), %%ymm0, %%ymm14\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc2]), %%ymm0, %%ymm15\n\t"
	            "\n\t"
	            "vmovupd  %%ymm13, 0*8(%[c0]        )\n\t"
	            "vmovupd  %%ymm14, 0*8(%[c1]        )\n\t"
	            "vmovupd  %%ymm15, 0*8(%[c0],%[ldc2])\n\t"
	            "\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "addq  $4*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 3*K;
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
/*
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){
	            a00 = *(A + 0 + 0*8 ); a10 = *(A + 1 + 0*8 ); a20 = *(A + 2 + 0*8 ); a30 = *(A + 3 + 0*8 ); // ymm0
	            a01 = *(A + 0 + 1*8 ); a11 = *(A + 1 + 1*8 ); a21 = *(A + 2 + 1*8 ); a31 = *(A + 3 + 1*8 ); // ymm1
	            b00 = *(B + 0 + 0*8 ); b10 = *(B + 1 + 0*8 ); b20 = *(B + 2 + 0*8 ); b30 = *(B + 3 + 0*8 ); // ymm4
	            b01 = *(B + 0 + 1*8 ); b11 = *(B + 1 + 1*8 ); b21 = *(B + 2 + 1*8 ); b31 = *(B + 3 + 1*8 ); // ymm5
	            b02 = *(B + 0 + 2*8 ); b12 = *(B + 1 + 2*8 ); b22 = *(B + 2 + 2*8 ); b32 = *(B + 3 + 2*8 ); // ymm6
	            b03 = *(B + 0 + 3*8 ); b13 = *(B + 1 + 3*8 ); b23 = *(B + 2 + 3*8 ); b33 = *(B + 3 + 3*8 ); // ymm7
	            c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            c02 += a00 * b02; c02 += a10 * b12; c02 += a20 * b22; c02 += a30 * b32; // ymm14
	            c03 += a00 * b03; c03 += a10 * b13; c03 += a20 * b23; c03 += a30 * b33; // ymm15
	            c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            c12 += a01 * b02; c12 += a11 * b12; c12 += a21 * b22; c12 += a31 * b32; // ymm14
	            c13 += a01 * b03; c13 += a11 * b13; c13 += a21 * b23; c13 += a31 * b33; // ymm15
	            a00 = *(A + 4 + 0*8 ); a10 = *(A + 5 + 0*8 ); a20 = *(A + 6 + 0*8 ); a30 = *(A + 7 + 0*8 ); // ymm0
	            a01 = *(A + 4 + 1*8 ); a11 = *(A + 5 + 1*8 ); a21 = *(A + 6 + 1*8 ); a31 = *(A + 7 + 1*8 ); // ymm1
	            b00 = *(B + 4 + 0*8 ); b10 = *(B + 5 + 0*8 ); b20 = *(B + 6 + 0*8 ); b30 = *(B + 7 + 0*8 ); // ymm4
	            b01 = *(B + 4 + 1*8 ); b11 = *(B + 5 + 1*8 ); b21 = *(B + 6 + 1*8 ); b31 = *(B + 7 + 1*8 ); // ymm5
	            b02 = *(B + 4 + 2*8 ); b12 = *(B + 5 + 2*8 ); b22 = *(B + 6 + 2*8 ); b32 = *(B + 7 + 2*8 ); // ymm6
	            b03 = *(B + 4 + 3*8 ); b13 = *(B + 5 + 3*8 ); b23 = *(B + 6 + 3*8 ); b33 = *(B + 7 + 3*8 ); // ymm7
	            c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            c02 += a00 * b02; c02 += a10 * b12; c02 += a20 * b22; c02 += a30 * b32; // ymm14
	            c03 += a00 * b03; c03 += a10 * b13; c03 += a20 * b23; c03 += a30 * b33; // ymm15
	            c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            c12 += a01 * b02; c12 += a11 * b12; c12 += a21 * b22; c12 += a31 * b32; // ymm14
	            c13 += a01 * b03; c13 += a11 * b13; c13 += a21 * b23; c13 += a31 * b33; // ymm15
	            A+=16;
	            B+=32;
	          }
	        }
	        if( K & 4 ){
*/
	        if( K >> 2 ){
	            //printf("    K4:\n");
	          size_t k4 = ( K >> 2 ); // Unrolling K
	          while( k4-- ){
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
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "\n\t"
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "\n\t"
	                "vmulpd   %%ymm0 , %%ymm4 , %%ymm6 \n\t" // [c00,c00,c00,c00]
	                "vmulpd   %%ymm1 , %%ymm4 , %%ymm7 \n\t" // [c10,c10,c10,c10]
	                "vshufpd  $0x00  , %%ymm7 , %%ymm6 , %%ymm5 \n\t" // [c00,c10,c00,c10]
	                "vshufpd  $0x0f  , %%ymm7 , %%ymm6 , %%ymm6 \n\t" // [c00,c10,c00,c10]
	                "vaddpd   %%ymm5 , %%ymm8 , %%ymm8 \n\t" // [c00,c10,c00,c10]
	                "vaddpd   %%ymm6 , %%ymm8 , %%ymm8 \n\t" // [c00,c10,c00,c10]
	                "\n\t"
	                "vmovapd   4*8(%[b]), %%ymm4 \n\t" // [b01,b11,b21,b31]
	                "\n\t"
	                "vmulpd   %%ymm0 , %%ymm4 , %%ymm6 \n\t" // [c01,c01,c01,c01]
	                "vmulpd   %%ymm1 , %%ymm4 , %%ymm7 \n\t" // [c11,c11,c11,c11]
	                "vshufpd  $0x00  , %%ymm7 , %%ymm6 , %%ymm5 \n\t" // [c01,c11,c01,c11]
	                "vshufpd  $0x0f  , %%ymm7 , %%ymm6 , %%ymm6 \n\t" // [c01,c11,c01,c11]
	                "vaddpd   %%ymm5 , %%ymm10, %%ymm10\n\t" // [c01,c11,c0h,c11]
	                "vaddpd   %%ymm6 , %%ymm10, %%ymm10\n\t" // [c01,c11,c01,c11]
	                "\n\t"
	                "vmovapd   8*8(%[b]), %%ymm4 \n\t" // [b02,b12,b22,b32]
	                "\n\t"
	                "vmulpd   %%ymm0 , %%ymm4 , %%ymm6 \n\t" // [c02,c02,c02,c02]
	                "vmulpd   %%ymm1 , %%ymm4 , %%ymm7 \n\t" // [c12,c12,c12,c12]
	                "vshufpd  $0x00  , %%ymm7 , %%ymm6 , %%ymm5 \n\t" // [c02,c12,c02,c12]
	                "vshufpd  $0x0f  , %%ymm7 , %%ymm6 , %%ymm6 \n\t" // [c02,c12,c02,c12]
	                "vaddpd   %%ymm5 , %%ymm12, %%ymm12\n\t" // [c02,c12,c02,c12]
	                "vaddpd   %%ymm6 , %%ymm12, %%ymm12\n\t" // [c02,c12,c02,c12]
	                "\n\t"
	                "vmovapd  12*8(%[b]), %%ymm4 \n\t" // [b03,b13,b23,b33]
	                "\n\t"
	                "vmulpd   %%ymm0 , %%ymm4 , %%ymm6 \n\t" // [c03,c03,c03,c03]
	                "vmulpd   %%ymm1 , %%ymm4 , %%ymm7 \n\t" // [c13,c13,c13,c13]
	                "vshufpd  $0x00  , %%ymm7 , %%ymm6 , %%ymm5 \n\t" // [c03,c13,c03,c13]
	                "vshufpd  $0x0f  , %%ymm7 , %%ymm6 , %%ymm6 \n\t" // [c03,c13,c03,c13]
	                "vaddpd   %%ymm5 , %%ymm14, %%ymm14\n\t" // [c03,c13,c03,c13]
	                "vaddpd   %%ymm6 , %%ymm14, %%ymm14\n\t" // [c03,c13,c03,c13]
	                "\n\t"
	                "addq  $8*8  , %[a]\n\t"
	                "addq  $16*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          }
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
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10]
	                "movapd   2*8(%[a]), %%xmm1 \n\t" // [a01,a11]
	                "\n\t"
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm4 , %%xmm6 \n\t" // [c00,c00]
	                "vmulpd   %%xmm1 , %%xmm4 , %%xmm7 \n\t" // [c10,c10]
	                "vshufpd  $0x00  , %%xmm7 , %%xmm6 , %%xmm5 \n\t" // [c00,c10]
	                "vshufpd  $0x0f  , %%xmm7 , %%xmm6 , %%xmm6 \n\t" // [c00,c10]
	                "addpd    %%xmm5 , %%xmm8 \n\t" // [c00,c10]
	                "addpd    %%xmm6 , %%xmm8 \n\t" // [c00,c10]
	                "\n\t"
	                "movapd   2*8(%[b]), %%xmm4 \n\t" // [b01,b11]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm4 , %%xmm6 \n\t" // [c01,c01]
	                "vmulpd   %%xmm1 , %%xmm4 , %%xmm7 \n\t" // [c11,c11]
	                "vshufpd  $0x00  , %%xmm7 , %%xmm6 , %%xmm5 \n\t" // [c01,c11]
	                "vshufpd  $0x0f  , %%xmm7 , %%xmm6 , %%xmm6 \n\t" // [c01,c11]
	                "addpd    %%xmm5 , %%xmm10\n\t" // [c01,c11]
	                "addpd    %%xmm6 , %%xmm10\n\t" // [c01,c11]
	                "\n\t"
	                "movapd   4*8(%[b]), %%xmm4 \n\t" // [b02,b12]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm4 , %%xmm6 \n\t" // [c02,c02]
	                "vmulpd   %%xmm1 , %%xmm4 , %%xmm7 \n\t" // [c12,c12]
	                "vshufpd  $0x00  , %%xmm7 , %%xmm6 , %%xmm5 \n\t" // [c02,c12]
	                "vshufpd  $0x0f  , %%xmm7 , %%xmm6 , %%xmm6 \n\t" // [c02,c12]
	                "addpd    %%xmm5 , %%xmm12\n\t" // [c02,c12]
	                "addpd    %%xmm6 , %%xmm12\n\t" // [c02,c12]
	                "\n\t"
	                "movapd   6*8(%[b]), %%xmm4 \n\t" // [b03,b13]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm4 , %%xmm6 \n\t" // [c03,c03]
	                "vmulpd   %%xmm1 , %%xmm4 , %%xmm7 \n\t" // [c13,c13]
	                "vshufpd  $0x00  , %%xmm7 , %%xmm6 , %%xmm5 \n\t" // [c03,c13]
	                "vshufpd  $0x0f  , %%xmm7 , %%xmm6 , %%xmm6 \n\t" // [c03,c13]
	                "addpd    %%xmm5 , %%xmm14\n\t" // [c03,c13]
	                "addpd    %%xmm6 , %%xmm14\n\t" // [c03,c13]
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
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a01]
	                "\n\t"
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b01]
	                "vshufpd  $0x00  , %%ymm4 , %%ymm4 , %%ymm2 \n\t" // [b00,b00]
	                "vshufpd  $0x03  , %%ymm4 , %%ymm4 , %%ymm3 \n\t" // [b01,b01]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm2 , %%xmm4 \n\t" // [c00,c10]
	                "vmulpd   %%xmm0 , %%xmm3 , %%xmm6 \n\t" // [c01,c11]
	                "addpd    %%xmm4 , %%xmm8 \n\t" // [c00,c10]
	                "addpd    %%xmm6 , %%xmm10\n\t" // [c01,c11]
	                "\n\t"
	                "movapd   2*8(%[b]), %%xmm4 \n\t" // [b02,b03]
	                "vshufpd  $0x00  , %%ymm4 , %%ymm4 , %%ymm2 \n\t" // [b02,b02]
	                "vshufpd  $0x03  , %%ymm4 , %%ymm4 , %%ymm3 \n\t" // [b03,b03]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm2 , %%xmm4 \n\t" // [c02,c12]
	                "vmulpd   %%xmm0 , %%xmm3 , %%xmm6 \n\t" // [c03,c13]
	                "addpd    %%xmm4 , %%xmm12\n\t" // [c02,c12]
	                "addpd    %%xmm6 , %%xmm14\n\t" // [c03,c13]
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
	            "movupd %[alpha], %%xmm9\n\t"
	            "\n\t"
	            "vperm2f128 $0x20, %%ymm8 , %%ymm8 , %%ymm0 \n\t" // [c00,c10,c00,c10]
	            "vperm2f128 $0x31, %%ymm8 , %%ymm8 , %%ymm1 \n\t" // [c00,c10,c00,c10]
	            "vperm2f128 $0x20, %%ymm10, %%ymm10, %%ymm2 \n\t" // [c01,c11,c01,c11]
	            "vperm2f128 $0x31, %%ymm10, %%ymm10, %%ymm3 \n\t" // [c01,c11,c01,c11]
	            "vperm2f128 $0x20, %%ymm12, %%ymm12, %%ymm4 \n\t" // [c02,c12,c02,c12]
	            "vperm2f128 $0x31, %%ymm12, %%ymm12, %%ymm5 \n\t" // [c02,c12,c02,c12]
	            "vperm2f128 $0x20, %%ymm14, %%ymm14, %%ymm6 \n\t" // [c03,c13,c03,c13]
	            "vperm2f128 $0x31, %%ymm14, %%ymm14, %%ymm7 \n\t" // [c03,c13,c03,c13]
	            "\n\t"
	            "addpd   %%xmm0 , %%xmm1 \n\t" // [c00,c10,--,--]
	            "addpd   %%xmm2 , %%xmm3 \n\t" // [c01,c11,--,--]
	            "addpd   %%xmm4 , %%xmm5 \n\t" // [c02,c12,--,--]
	            "addpd   %%xmm6 , %%xmm7 \n\t" // [c03,c13,--,--]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]        ), %%xmm9, %%xmm1\n\t"
	            "vfmadd213pd 0*8(%[c1]        ), %%xmm9, %%xmm3\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc2]), %%xmm9, %%xmm5\n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2]), %%xmm9, %%xmm7\n\t"
	            "\n\t"
	            "movupd  %%xmm1 , 0*8(%[c0]        )\n\t"
	            "movupd  %%xmm3 , 0*8(%[c1]        )\n\t"
	            "movupd  %%xmm5 , 0*8(%[c0],%[ldc2])\n\t"
	            "movupd  %%xmm7 , 0*8(%[c1],%[ldc2])\n\t"
	            "\n\t"
	            "addq  $2*8, %[c0]\n\t"
	            "addq  $2*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 3*K;
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


/*
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){
	            a00 = *(A + 0 + 0*8 ); a10 = *(A + 1 + 0*8 ); a20 = *(A + 2 + 0*8 ); a30 = *(A + 3 + 0*8 ); // ymm0
	            b00 = *(B + 0 + 0*8 ); b10 = *(B + 1 + 0*8 ); b20 = *(B + 2 + 0*8 ); b30 = *(B + 3 + 0*8 ); // ymm4
	            b01 = *(B + 0 + 1*8 ); b11 = *(B + 1 + 1*8 ); b21 = *(B + 2 + 1*8 ); b31 = *(B + 3 + 1*8 ); // ymm5
	            b02 = *(B + 0 + 2*8 ); b12 = *(B + 1 + 2*8 ); b22 = *(B + 2 + 2*8 ); b32 = *(B + 3 + 2*8 ); // ymm6
	            b03 = *(B + 0 + 3*8 ); b13 = *(B + 1 + 3*8 ); b23 = *(B + 2 + 3*8 ); b33 = *(B + 3 + 3*8 ); // ymm7
	            c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            c02 += a00 * b02; c02 += a10 * b12; c02 += a20 * b22; c02 += a30 * b32; // ymm14
	            c03 += a00 * b03; c03 += a10 * b13; c03 += a20 * b23; c03 += a30 * b33; // ymm15
	            a00 = *(A + 4 + 0*8 ); a10 = *(A + 5 + 0*8 ); a20 = *(A + 6 + 0*8 ); a30 = *(A + 7 + 0*8 ); // ymm0
	            b00 = *(B + 4 + 0*8 ); b10 = *(B + 5 + 0*8 ); b20 = *(B + 6 + 0*8 ); b30 = *(B + 7 + 0*8 ); // ymm4
	            b01 = *(B + 4 + 1*8 ); b11 = *(B + 5 + 1*8 ); b21 = *(B + 6 + 1*8 ); b31 = *(B + 7 + 1*8 ); // ymm5
	            b02 = *(B + 4 + 2*8 ); b12 = *(B + 5 + 2*8 ); b22 = *(B + 6 + 2*8 ); b32 = *(B + 7 + 2*8 ); // ymm6
	            b03 = *(B + 4 + 3*8 ); b13 = *(B + 5 + 3*8 ); b23 = *(B + 6 + 3*8 ); b33 = *(B + 7 + 3*8 ); // ymm7
	            c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            c02 += a00 * b02; c02 += a10 * b12; c02 += a20 * b22; c02 += a30 * b32; // ymm14
	            c03 += a00 * b03; c03 += a10 * b13; c03 += a20 * b23; c03 += a30 * b33; // ymm15
	            A+=8;
	            B+=32;
	          }
	        }
	        if( K & 4 ){
*/
	        if( K >> 2 ){
	            //printf("    K4:\n");
	          size_t k4 = ( K >> 2 ); // Unrolling K
	          while( k4-- ){
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
	                "vmovupd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "vmovapd   4*8(%[b]), %%ymm5 \n\t" // [b01,b11,b21,b31]
	                "vmovapd   8*8(%[b]), %%ymm6 \n\t" // [b02,b12,b22,b32]
	                "vmovapd  12*8(%[b]), %%ymm7 \n\t" // [b03,b13,b23,b33]
	                "\n\t"
	                "vmulpd   %%ymm0 , %%ymm4 , %%ymm4 \n\t" // [c00,c00,c00,c00]
	                "vmulpd   %%ymm0 , %%ymm5 , %%ymm5 \n\t" // [c01,c01,c01,c01]
	                "vmulpd   %%ymm0 , %%ymm6 , %%ymm6 \n\t" // [c02,c02,c02,c02]
	                "vmulpd   %%ymm0 , %%ymm7 , %%ymm7 \n\t" // [c03,c03,c03,c03]
	                "vaddpd   %%ymm4 , %%ymm12, %%ymm12\n\t" // [c00,c00,c00,c00]
	                "vaddpd   %%ymm5 , %%ymm13, %%ymm13\n\t" // [c01,c01,c01,c01]
	                "vaddpd   %%ymm6 , %%ymm14, %%ymm14\n\t" // [c02,c02,c02,c02]
	                "vaddpd   %%ymm7 , %%ymm15, %%ymm15\n\t" // [c03,c03,c03,c03]
	                "\n\t"
	                "addq  $4*8  , %[a]\n\t"
	                "addq  $16*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          }
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
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,--,--]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10,--,--]
	                "movapd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11,--,--]
	                "movapd   4*8(%[b]), %%xmm6 \n\t" // [b02,b12,--,--]
	                "movapd   6*8(%[b]), %%xmm7 \n\t" // [b03,b13,--,--]
	                "\n\t"
	                "mulpd   %%xmm0 , %%xmm4 \n\t" // [c00,c00,--,--]
	                "mulpd   %%xmm0 , %%xmm5 \n\t" // [c01,c01,--,--]
	                "mulpd   %%xmm0 , %%xmm6 \n\t" // [c02,c02,--,--]
	                "mulpd   %%xmm0 , %%xmm7 \n\t" // [c03,c03,--,--]
	                "addpd   %%xmm4 , %%xmm12\n\t" // [c00,c00,--,--]
	                "addpd   %%xmm5 , %%xmm13\n\t" // [c01,c01,--,--]
	                "addpd   %%xmm6 , %%xmm14\n\t" // [c02,c02,--,--]
	                "addpd   %%xmm7 , %%xmm15\n\t" // [c03,c03,--,--]
	                "\n\t"
	                "addq  $2*8  , %[a]\n\t"
	                "addq  $8*8  , %[b]\n\t"
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
	                "movsd   0*8(%[a]), %%xmm0 \n\t" // [a00,--,--,--]
	                "movsd   0*8(%[b]), %%xmm4 \n\t" // [b00,--,--,--]
	                "movsd   1*8(%[b]), %%xmm5 \n\t" // [b01,--,--,--]
	                "movsd   2*8(%[b]), %%xmm6 \n\t" // [b02,--,--,--]
	                "movsd   3*8(%[b]), %%xmm7 \n\t" // [b03,--,--,--]
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
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "movlpd   0*8(%[c0]        ), %%xmm4\n\t"
	            "movhpd   0*8(%[c1]        ), %%xmm4\n\t"
	            "movlpd   0*8(%[c0],%[ldc2]), %%xmm5\n\t"
	            "movhpd   0*8(%[c1],%[ldc2]), %%xmm5\n\t"
	            "\n\t"
	            "vhaddpd  %%ymm13, %%ymm12, %%ymm8 \n\t" // [c00,c01,c00,c01]
	            "vhaddpd  %%ymm15, %%ymm14, %%ymm9 \n\t" // [c02,c03,c02,c03]
	            "vperm2f128 $0x31, %%ymm8 , %%ymm8 , %%ymm10\n\t" // [c00,c01,c00,c01]
	            "vperm2f128 $0x31, %%ymm9 , %%ymm9 , %%ymm11\n\t" // [c02,c03,c02,c03]
	            "addpd    %%xmm10, %%xmm8 \n\t" // [c00,c01,---,---]
	            "addpd    %%xmm11, %%xmm9 \n\t" // [c02,c03,---,---]
	            "\n\t"
	            "mulpd    %%xmm0 , %%xmm8 \n\t" // [c00,c01,---,---]
	            "mulpd    %%xmm0 , %%xmm9 \n\t" // [c02,c03,---,---]
	            "addpd    %%xmm4 , %%xmm8 \n\t" // [c00,c01,---,---]
	            "addpd    %%xmm5 , %%xmm9 \n\t" // [c02,c03,---,---]
	            "\n\t"
	            "movlpd   %%xmm8 , 0*8(%[c0]        )\n\t"
	            "movhpd   %%xmm8 , 0*8(%[c1]        )\n\t"
	            "movlpd   %%xmm9 , 0*8(%[c0],%[ldc2])\n\t"
	            "movhpd   %%xmm9 , 0*8(%[c1],%[ldc2])\n\t"
	            "\n\t"
	            "addq  $1*8, %[c0]\n\t"
	            "addq  $1*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 3*K;
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
	    B = B + 3*K;
	    C  = C - M + 3*ldc;
	    c0 = c0- M + 3*ldc;
	    c1 = c1- M + 3*ldc;
	  }
	}
	if( N3R==2 ){
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
/*
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){
	            a00 = *(A + 0 + 0*8 ); a10 = *(A + 1 + 0*8 ); a20 = *(A + 2 + 0*8 ); a30 = *(A + 3 + 0*8 ); // ymm0
	            a01 = *(A + 0 + 1*8 ); a11 = *(A + 1 + 1*8 ); a21 = *(A + 2 + 1*8 ); a31 = *(A + 3 + 1*8 ); // ymm1
	            a02 = *(A + 0 + 2*8 ); a12 = *(A + 1 + 2*8 ); a22 = *(A + 2 + 2*8 ); a32 = *(A + 3 + 2*8 ); // ymm2
	            a03 = *(A + 0 + 3*8 ); a13 = *(A + 1 + 3*8 ); a23 = *(A + 2 + 3*8 ); a33 = *(A + 3 + 3*8 ); // ymm3
	            b00 = *(B + 0 + 0*8 ); b10 = *(B + 1 + 0*8 ); b20 = *(B + 2 + 0*8 ); b30 = *(B + 3 + 0*8 ); // ymm4
	            b01 = *(B + 0 + 1*8 ); b11 = *(B + 1 + 1*8 ); b21 = *(B + 2 + 1*8 ); b31 = *(B + 3 + 1*8 ); // ymm5
	            c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            c20 += a02 * b00; c20 += a12 * b10; c20 += a22 * b20; c20 += a32 * b30; // ymm12
	            c21 += a02 * b01; c21 += a12 * b11; c21 += a22 * b21; c21 += a32 * b31; // ymm13
	            c30 += a03 * b00; c30 += a13 * b10; c30 += a23 * b20; c30 += a33 * b30; // ymm12
	            c31 += a03 * b01; c31 += a13 * b11; c31 += a23 * b21; c31 += a33 * b31; // ymm13
	            a00 = *(A + 4 + 0*8 ); a10 = *(A + 5 + 0*8 ); a20 = *(A + 6 + 0*8 ); a30 = *(A + 7 + 0*8 ); // ymm0
	            a01 = *(A + 4 + 1*8 ); a11 = *(A + 5 + 1*8 ); a21 = *(A + 6 + 1*8 ); a31 = *(A + 7 + 1*8 ); // ymm1
	            a02 = *(A + 4 + 2*8 ); a12 = *(A + 5 + 2*8 ); a22 = *(A + 6 + 2*8 ); a32 = *(A + 7 + 2*8 ); // ymm2
	            a03 = *(A + 4 + 3*8 ); a13 = *(A + 5 + 3*8 ); a23 = *(A + 6 + 3*8 ); a33 = *(A + 7 + 3*8 ); // ymm3
	            b00 = *(B + 4 + 0*8 ); b10 = *(B + 5 + 0*8 ); b20 = *(B + 6 + 0*8 ); b30 = *(B + 7 + 0*8 ); // ymm4
	            b01 = *(B + 4 + 1*8 ); b11 = *(B + 5 + 1*8 ); b21 = *(B + 6 + 1*8 ); b31 = *(B + 7 + 1*8 ); // ymm5
	            c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            c20 += a02 * b00; c20 += a12 * b10; c20 += a22 * b20; c20 += a32 * b30; // ymm12
	            c21 += a02 * b01; c21 += a12 * b11; c21 += a22 * b21; c21 += a32 * b31; // ymm13
	            c30 += a03 * b00; c30 += a13 * b10; c30 += a23 * b20; c30 += a33 * b30; // ymm12
	            c31 += a03 * b01; c31 += a13 * b11; c31 += a23 * b21; c31 += a33 * b31; // ymm13
	            A+=32;
	            B+=16;
	          }
	        }
	        if( K & 4 ){
*/
	        if( K >> 2 ){
	            //printf("    K4:\n");
	          size_t k4 = ( K >> 2 ); // Unrolling K
	          while( k4-- ){
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
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a02,a12,a22,a32]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a03,a13,a23,a30]
	                "\n\t"
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "vmovapd   4*8(%[b]), %%ymm5 \n\t" // [b01,b11,b21,b31]
	                "\n\t"
	                "vmulpd   %%ymm0 , %%ymm4 , %%ymm6 \n\t" // [c00,c00,c00,c00]
	                "vmulpd   %%ymm1 , %%ymm4 , %%ymm7 \n\t" // [c10,c10,c10,c10]
	                "vaddpd   %%ymm6 , %%ymm8 , %%ymm8 \n\t" // [c00,c00,c00,c00]
	                "vaddpd   %%ymm7 , %%ymm9 , %%ymm9 \n\t" // [c10,c10,c10,c10]
	                "\n\t"
	                "vmulpd   %%ymm2 , %%ymm4 , %%ymm6 \n\t" // [c20,c20,c20,c20]
	                "vmulpd   %%ymm3 , %%ymm4 , %%ymm7 \n\t" // [c30,c30,c30,c30]
	                "vaddpd   %%ymm6 , %%ymm10, %%ymm10\n\t" // [c20,c20,c20,c20]
	                "vaddpd   %%ymm7 , %%ymm11, %%ymm11\n\t" // [c30,c30,c30,c30]
	                "\n\t"
	                "vmulpd   %%ymm0 , %%ymm5 , %%ymm6 \n\t" // [c01,c01,c01,c01]
	                "vmulpd   %%ymm1 , %%ymm5 , %%ymm7 \n\t" // [c11,c11,c11,c11]
	                "vaddpd   %%ymm6 , %%ymm12, %%ymm12\n\t" // [c01,c01,c01,c01]
	                "vaddpd   %%ymm7 , %%ymm13, %%ymm13\n\t" // [c11,c11,c11,c11]
	                "\n\t"
	                "vmulpd   %%ymm2 , %%ymm5 , %%ymm6 \n\t" // [c21,c21,c21,c21]
	                "vmulpd   %%ymm3 , %%ymm5 , %%ymm7 \n\t" // [c31,c31,c31,c31]
	                "vaddpd   %%ymm6 , %%ymm14, %%ymm14\n\t" // [c21,c21,c21,c21]
	                "vaddpd   %%ymm7 , %%ymm15, %%ymm15\n\t" // [c31,c31,c31,c31]
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          }
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
	            "vmovupd %[alpha], %%ymm0\n\t"
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
/*
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){
	            a00 = *(A + 0 + 0*8 ); a10 = *(A + 1 + 0*8 ); a20 = *(A + 2 + 0*8 ); a30 = *(A + 3 + 0*8 ); // ymm0
	            a01 = *(A + 0 + 1*8 ); a11 = *(A + 1 + 1*8 ); a21 = *(A + 2 + 1*8 ); a31 = *(A + 3 + 1*8 ); // ymm1
	            b00 = *(B + 0 + 0*8 ); b10 = *(B + 1 + 0*8 ); b20 = *(B + 2 + 0*8 ); b30 = *(B + 3 + 0*8 ); // ymm4
	            b01 = *(B + 0 + 1*8 ); b11 = *(B + 1 + 1*8 ); b21 = *(B + 2 + 1*8 ); b31 = *(B + 3 + 1*8 ); // ymm5
	            c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            a00 = *(A + 4 + 0*8 ); a10 = *(A + 5 + 0*8 ); a20 = *(A + 6 + 0*8 ); a30 = *(A + 7 + 0*8 ); // ymm0
	            a01 = *(A + 4 + 1*8 ); a11 = *(A + 5 + 1*8 ); a21 = *(A + 6 + 1*8 ); a31 = *(A + 7 + 1*8 ); // ymm1
	            b00 = *(B + 4 + 0*8 ); b10 = *(B + 5 + 0*8 ); b20 = *(B + 6 + 0*8 ); b30 = *(B + 7 + 0*8 ); // ymm4
	            b01 = *(B + 4 + 1*8 ); b11 = *(B + 5 + 1*8 ); b21 = *(B + 6 + 1*8 ); b31 = *(B + 7 + 1*8 ); // ymm5
	            c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            c11 += a01 * b01; c11 += a11 * b11; c11 += a21 * b21; c11 += a31 * b31; // ymm13
	            A+=16;
	            B+=16;
	          }
	        }
	        if( K & 4 ){
*/
	        if( K >> 2 ){
	            //printf("    K4:\n");
	          size_t k4 = ( K >> 2 ); // Unrolling K
	          while( k4-- ){
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
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "\n\t"
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "vmovapd   4*8(%[b]), %%ymm5 \n\t" // [b01,b11,b21,b31]
	                "\n\t"
	                "vmulpd   %%ymm0 , %%ymm4 , %%ymm6 \n\t" // [c00,c00,c00,c00]
	                "vmulpd   %%ymm1 , %%ymm4 , %%ymm7 \n\t" // [c10,c10,c10,c10]
	                "vaddpd   %%ymm6 , %%ymm12, %%ymm12\n\t" // [c00,c00,c00,c00]
	                "vaddpd   %%ymm7 , %%ymm13, %%ymm13\n\t" // [c10,c10,c10,c10]
	                "\n\t"
	                "vmulpd   %%ymm0 , %%ymm5 , %%ymm8 \n\t" // [c01,c01,c01,c01]
	                "vmulpd   %%ymm1 , %%ymm5 , %%ymm9 \n\t" // [c11,c11,c11,c11]
	                "vaddpd   %%ymm8 , %%ymm14, %%ymm14\n\t" // [c01,c01,c01,c01]
	                "vaddpd   %%ymm9 , %%ymm15, %%ymm15\n\t" // [c11,c11,c11,c11]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
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
	            "vfmadd213pd 0*8(%[c1]        ), %%xmm0, %%xmm14\n\t"
	            "\n\t"
	            "movupd  %%xmm12, 0*8(%[c0]        )\n\t"
	            "movupd  %%xmm14, 0*8(%[c1]        )\n\t"
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
/*
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){
	            a00 = *(A + 0 + 0*8 ); a10 = *(A + 1 + 0*8 ); a20 = *(A + 2 + 0*8 ); a30 = *(A + 3 + 0*8 ); // ymm0
	            b00 = *(B + 0 + 0*8 ); b10 = *(B + 1 + 0*8 ); b20 = *(B + 2 + 0*8 ); b30 = *(B + 3 + 0*8 ); // ymm4
	            b01 = *(B + 0 + 1*8 ); b11 = *(B + 1 + 1*8 ); b21 = *(B + 2 + 1*8 ); b31 = *(B + 3 + 1*8 ); // ymm5
	            c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            a00 = *(A + 4 + 0*8 ); a10 = *(A + 5 + 0*8 ); a20 = *(A + 6 + 0*8 ); a30 = *(A + 7 + 0*8 ); // ymm0
	            b00 = *(B + 4 + 0*8 ); b10 = *(B + 5 + 0*8 ); b20 = *(B + 6 + 0*8 ); b30 = *(B + 7 + 0*8 ); // ymm4
	            b01 = *(B + 4 + 1*8 ); b11 = *(B + 5 + 1*8 ); b21 = *(B + 6 + 1*8 ); b31 = *(B + 7 + 1*8 ); // ymm5
	            c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            A+=8;
	            B+=16;
	          }
	        }
	        if( K & 4 ){
*/
	        if( K >> 2 ){
	            //printf("    K4:\n");
	          size_t k4 = ( K >> 2 ); // Unrolling K
	          while( k4-- ){
	            //a00 = *(A + 0 + 0*4 ); a10 = *(A + 1 + 0*4 ); a20 = *(A + 2 + 0*4 ); a30 = *(A + 3 + 0*4 ); // ymm0
	            //b00 = *(B + 0 + 0*4 ); b10 = *(B + 1 + 0*4 ); b20 = *(B + 2 + 0*4 ); b30 = *(B + 3 + 0*4 ); // ymm4
	            //b01 = *(B + 0 + 1*4 ); b11 = *(B + 1 + 1*4 ); b21 = *(B + 2 + 1*4 ); b31 = *(B + 3 + 1*4 ); // ymm5
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c01 += a00 * b01; c01 += a10 * b11; c01 += a20 * b21; c01 += a30 * b31; // ymm13
	            //A+=4;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovupd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "\n\t"
	                "vmovapd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "vmovapd   4*8(%[b]), %%ymm5 \n\t" // [b01,b11,b21,b31]
	                "\n\t"
	                "vmulpd   %%ymm0 , %%ymm4 , %%ymm6 \n\t" // [c00,c00,c00,c00]
	                "vaddpd   %%ymm6 , %%ymm14, %%ymm14\n\t" // [c00,c00,c00,c00]
	                "\n\t"
	                "vmulpd   %%ymm0 , %%ymm5 , %%ymm8 \n\t" // [c01,c01,c01,c01]
	                "vaddpd   %%ymm8 , %%ymm15, %%ymm15\n\t" // [c01,c01,c01,c01]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
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
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,---,---]
	                "\n\t"
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10,---,---]
	                "movapd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11,---,---]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm4 , %%xmm6 \n\t" // [c00,c00,---,---]
	                "addpd    %%xmm6 , %%xmm14\n\t" // [c00,c00,---,---]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm5 , %%xmm8 \n\t" // [c01,c01,---,---]
	                "addpd    %%xmm8 , %%xmm15\n\t" // [c01,c01,---,---]
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
	                "movsd    0*8(%[a]), %%xmm0 \n\t" // [a00,---,---,---]
	                "\n\t"
	                "movsd    0*8(%[b]), %%xmm4 \n\t" // [b00,---,---,---]
	                "movsd    1*8(%[b]), %%xmm5 \n\t" // [b01,---,---,---]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm4 , %%xmm6 \n\t" // [c00,---,---,---]
	                "addsd    %%xmm6 , %%xmm14\n\t" // [c00,---,---,---]
	                "\n\t"
	                "vmulpd   %%xmm0 , %%xmm5 , %%xmm8 \n\t" // [c01,---,---,---]
	                "addsd    %%xmm8 , %%xmm15\n\t" // [c01,---,---,---]
	                "\n\t"
	                "addq  $1*8 , %[a]\n\t"
	                "addq  $2*8 , %[b]\n\t"
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
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "movsd  0*8(%[c0]), %%xmm2 \n\t"
	            "movsd  0*8(%[c1]), %%xmm3 \n\t"
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
	            "movsd   %%xmm15, 0*8(%[c1]        )\n\t"
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
	if( N3R==1 ){
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
/*
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){
	            a00 = *(A + 0 + 0*8 ); a10 = *(A + 1 + 0*8 ); a20 = *(A + 2 + 0*8 ); a30 = *(A + 3 + 0*8 ); // ymm0
	            a01 = *(A + 0 + 1*8 ); a11 = *(A + 1 + 1*8 ); a21 = *(A + 2 + 1*8 ); a31 = *(A + 3 + 1*8 ); // ymm1
	            a02 = *(A + 0 + 2*8 ); a12 = *(A + 1 + 2*8 ); a22 = *(A + 2 + 2*8 ); a32 = *(A + 3 + 2*8 ); // ymm2
	            a03 = *(A + 0 + 3*8 ); a13 = *(A + 1 + 3*8 ); a23 = *(A + 2 + 3*8 ); a33 = *(A + 3 + 3*8 ); // ymm3
	            b00 = *(B + 0 + 0*8 ); b10 = *(B + 1 + 0*8 ); b20 = *(B + 2 + 0*8 ); b30 = *(B + 3 + 0*8 ); // ymm4
	            c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            c20 += a02 * b00; c20 += a12 * b10; c20 += a22 * b20; c20 += a32 * b30; // ymm12
	            c30 += a03 * b00; c30 += a13 * b10; c30 += a23 * b20; c30 += a33 * b30; // ymm12
	            a00 = *(A + 4 + 0*8 ); a10 = *(A + 5 + 0*8 ); a20 = *(A + 6 + 0*8 ); a30 = *(A + 7 + 0*8 ); // ymm0
	            a01 = *(A + 4 + 1*8 ); a11 = *(A + 5 + 1*8 ); a21 = *(A + 6 + 1*8 ); a31 = *(A + 7 + 1*8 ); // ymm1
	            a02 = *(A + 4 + 2*8 ); a12 = *(A + 5 + 2*8 ); a22 = *(A + 6 + 2*8 ); a32 = *(A + 7 + 2*8 ); // ymm2
	            a03 = *(A + 4 + 3*8 ); a13 = *(A + 5 + 3*8 ); a23 = *(A + 6 + 3*8 ); a33 = *(A + 7 + 3*8 ); // ymm3
	            b00 = *(B + 4 + 0*8 ); b10 = *(B + 5 + 0*8 ); b20 = *(B + 6 + 0*8 ); b30 = *(B + 7 + 0*8 ); // ymm4
	            c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            c20 += a02 * b00; c20 += a12 * b10; c20 += a22 * b20; c20 += a32 * b30; // ymm12
	            c30 += a03 * b00; c30 += a13 * b10; c30 += a23 * b20; c30 += a33 * b30; // ymm12
	            A+=32;
	            B+=8;
	          }
	        }
	        if( K & 4 ){
*/
	        if( K >> 2 ){
	            //printf("    K4:\n");
	          size_t k4 = ( K >> 2 ); // Unrolling K
	          while( k4-- ){
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
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a02,a12,a22,a32]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a03,a13,a23,a30]
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "\n\t"
	                "vmulpd   %%ymm0 , %%ymm4 , %%ymm6 \n\t" // [c00,c00,c00,c00]
	                "vmulpd   %%ymm1 , %%ymm4 , %%ymm7 \n\t" // [c10,c10,c10,c10]
	                "vaddpd   %%ymm6 , %%ymm12, %%ymm12\n\t" // [c00,c00,c00,c00]
	                "vaddpd   %%ymm7 , %%ymm13, %%ymm13\n\t" // [c10,c10,c10,c10]
	                "\n\t"
	                "vmulpd   %%ymm2 , %%ymm4 , %%ymm6 \n\t" // [c20,c20,c20,c20]
	                "vmulpd   %%ymm3 , %%ymm4 , %%ymm7 \n\t" // [c30,c30,c30,c30]
	                "vaddpd   %%ymm6 , %%ymm14, %%ymm14\n\t" // [c20,c20,c20,c20]
	                "vaddpd   %%ymm7 , %%ymm15, %%ymm15\n\t" // [c30,c30,c30,c30]
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
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
/*
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){
	            a00 = *(A + 0 + 0*8 ); a10 = *(A + 1 + 0*8 ); a20 = *(A + 2 + 0*8 ); a30 = *(A + 3 + 0*8 ); // ymm0
	            a01 = *(A + 0 + 1*8 ); a11 = *(A + 1 + 1*8 ); a21 = *(A + 2 + 1*8 ); a31 = *(A + 3 + 1*8 ); // ymm1
	            b00 = *(B + 0 + 0*8 ); b10 = *(B + 1 + 0*8 ); b20 = *(B + 2 + 0*8 ); b30 = *(B + 3 + 0*8 ); // ymm4
	            c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            a00 = *(A + 4 + 0*8 ); a10 = *(A + 5 + 0*8 ); a20 = *(A + 6 + 0*8 ); a30 = *(A + 7 + 0*8 ); // ymm0
	            b00 = *(B + 4 + 0*8 ); b10 = *(B + 5 + 0*8 ); b20 = *(B + 6 + 0*8 ); b30 = *(B + 7 + 0*8 ); // ymm4
	            c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            A+=16;
	            B+=8;
	          }
	        }
	        if( K & 4 ){
*/
	        if( K >> 2 ){
	            //printf("    K4:\n");
	          size_t k4 = ( K >> 2 ); // Unrolling K
	          while( k4-- ){
	            //a00 = *(A + 0 + 0*4 ); a10 = *(A + 1 + 0*4 ); a20 = *(A + 2 + 0*4 ); a30 = *(A + 3 + 0*4 ); // ymm0
	            //a01 = *(A + 0 + 1*4 ); a11 = *(A + 1 + 1*4 ); a21 = *(A + 2 + 1*4 ); a31 = *(A + 3 + 1*4 ); // ymm1
	            //b00 = *(B + 0 + 0*4 ); b10 = *(B + 1 + 0*4 ); b20 = *(B + 2 + 0*4 ); b30 = *(B + 3 + 0*4 ); // ymm4
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //c10 += a01 * b00; c10 += a11 * b10; c10 += a21 * b20; c10 += a31 * b30; // ymm12
	            //A+=8;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "\n\t"
	                "vmulpd   %%ymm0 , %%ymm4 , %%ymm6 \n\t" // [c00,c00,c00,c00]
	                "vmulpd   %%ymm1 , %%ymm4 , %%ymm7 \n\t" // [c10,c10,c10,c10]
	                "vaddpd   %%ymm6 , %%ymm14, %%ymm14\n\t" // [c00,c00,c00,c00]
	                "vaddpd   %%ymm7 , %%ymm15, %%ymm15\n\t" // [c10,c10,c10,c10]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          }
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
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+1+0*ldc) += alpha*c10;
	        ////A = A - K + 4*K;
	        //B = B - 1*K;
	        //C+=2;
	        //////printf("pass through (n&1)&(m&2)\n");
	            //printf("    END:\n");

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
/*
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){
	            a00 = *(A + 0 + 0*8 ); a10 = *(A + 1 + 0*8 ); a20 = *(A + 2 + 0*8 ); a30 = *(A + 3 + 0*8 ); // ymm0
	            b00 = *(B + 0 + 0*8 ); b10 = *(B + 1 + 0*8 ); b20 = *(B + 2 + 0*8 ); b30 = *(B + 3 + 0*8 ); // ymm4
	            c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            a00 = *(A + 4 + 0*8 ); a10 = *(A + 5 + 0*8 ); a20 = *(A + 6 + 0*8 ); a30 = *(A + 7 + 0*8 ); // ymm0
	            b00 = *(B + 4 + 0*8 ); b10 = *(B + 5 + 0*8 ); b20 = *(B + 6 + 0*8 ); b30 = *(B + 7 + 0*8 ); // ymm4
	            c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            A+=8;
	            B+=8;
	          }
	        }
	        if( K & 4 ){
*/
	        if( K >> 2 ){
	            //printf("    K4:\n");
	          size_t k4 = ( K >> 2 ); // Unrolling K
	          while( k4-- ){
	            //a00 = *(A + 0 + 0*4 ); a10 = *(A + 1 + 0*4 ); a20 = *(A + 2 + 0*4 ); a30 = *(A + 3 + 0*4 ); // ymm0
	            //b00 = *(B + 0 + 0*4 ); b10 = *(B + 1 + 0*4 ); b20 = *(B + 2 + 0*4 ); b30 = *(B + 3 + 0*4 ); // ymm4
	            //c00 += a00 * b00; c00 += a10 * b10; c00 += a20 * b20; c00 += a30 * b30; // ymm12
	            //A+=4;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovupd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b20,b30]
	                "\n\t"
	                "vmulpd   %%ymm0 , %%ymm4 , %%ymm6 \n\t" // [c00,c00,c00,c00]
	                "vaddpd   %%ymm6 , %%ymm15, %%ymm15\n\t" // [c00,c00,c00,c00]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
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

