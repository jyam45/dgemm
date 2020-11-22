#include "myblas_internal.h"
#include <stdio.h>

/*  case 8x3x4x2 

for( size_t k=0; k<4; k++ ){
 for( size_t j=0; j<3; j++ ){
   for( size_t i=0; i<8; i++ ){
      for( size_t l=0; l<2; l++ ){
        c[i+8*j] += (*(A2+l+i*2+k*2*8))*(*(B2+l+j*2+k*2*3));
      }
   }
 }
}
A2+=8*4*2;
B2+=4*2*3;#

*/

void myblas_dgemm_kernel_detail(
         size_t M, size_t N, size_t K,
         double alpha, const double *A, const double *B, 
         double *C, size_t ldc )
{
	double *c0 = C;
	size_t ldc1 = ldc * 1 * sizeof(double);
	double alpha4[4] = {alpha,alpha,alpha,alpha};

	size_t N3Q = N / 3;
	size_t N3R = N % 3;

	// Kernel ----
	if( N3Q ){

	  size_t n3 = N3Q; // unrolling N
	  while( n3-- ){
	    if( M >> 3 ){
	      size_t m8 = ( M >> 3 ); // unrolling M
	      while( m8-- ){

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

	        __asm__ __volatile__ (
	            "\n\t"
	            "prefetcht0 0*8(%[c0]          )\n\t"
	            "prefetcht0 4*8(%[c0]          )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc1]  )\n\t"
	            "prefetcht0 4*8(%[c0],%[ldc1]  )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc1],2)\n\t"
	            "prefetcht0 4*8(%[c0],%[ldc1],2)\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[ldc1]"r"(ldc1)
	        );


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){
	            //
	            // c[i+8*j] += (*(A+l+i*2+k*2*8))*(*(B+l+j*2+k*2*3));
	            //
	            // b[k,j]
	            // a[k,i]
	            // c[i,j]
	            //
	            // b00 = *(B+0+0*2+0*2*3); b10 = *(B+1+0*2+0*2*3);
	            // b01 = *(B+0+1*2+0*2*3); b11 = *(B+1+1*2+0*2*3);
	            // b02 = *(B+0+2*2+0*2*3); b12 = *(B+1+2*2+0*2*3);
	            //                            
	            // a00 = *(A+0+0*2+0*2*8); a10 = *(A+1+0*2+0*2*8);
	            // a01 = *(A+0+1*2+0*2*8); a11 = *(A+1+1*2+0*2*8);
	            // c00 += a00 * b00; c00 += a10 * b10; c10 += a01 * b00; c10 += a11 * b10; 
	            // c01 += a00 * b01; c01 += a10 * b11; c11 += a01 * b01; c11 += a11 * b11; 
	            // c02 += a00 * b02; c02 += a10 * b12; c12 += a01 * b02; c12 += a11 * b12; 
	            //
	            // a02 = *(A+0+2*2+0*2*8); a12 = *(A+1+2*2+0*2*8);
	            // a03 = *(A+0+3*2+0*2*8); a13 = *(A+1+3*2+0*2*8);
	            // c20 += a02 * b00; c20 += a12 * b10; c30 += a03 * b00; c30 += a13 * b10; 
	            // c21 += a02 * b01; c21 += a12 * b11; c31 += a03 * b01; c31 += a13 * b11; 
	            // c22 += a02 * b02; c22 += a12 * b12; c32 += a03 * b02; c32 += a13 * b12; 
	            //
	            // a04 = *(A+0+4*2+0*2*8); a14 = *(A+1+4*2+0*2*8);
	            // a05 = *(A+0+5*2+0*2*8); a15 = *(A+1+5*2+0*2*8);
	            // c40 += a04 * b00; c40 += a14 * b10; c50 += a05 * b00; c50 += a15 * b10; 
	            // c41 += a04 * b01; c41 += a14 * b11; c51 += a05 * b01; c51 += a15 * b11; 
	            // c42 += a04 * b02; c42 += a14 * b12; c52 += a05 * b02; c52 += a15 * b12; 
	            //
	            // a06 = *(A+0+6*2+0*2*8); a16 = *(A+1+6*2+0*2*8);
	            // a07 = *(A+0+7*2+0*2*8); a17 = *(A+1+7*2+0*2*8);
	            // c60 += a06 * b00; c60 += a16 * b10; c70 += a07 * b00; c70 += a17 * b10; 
	            // c61 += a06 * b01; c61 += a16 * b11; c71 += a07 * b01; c71 += a17 * b11; 
	            // c62 += a06 * b02; c62 += a16 * b12; c72 += a07 * b02; c72 += a17 * b12; 
	            //
	            // a20 = *(A+0+0*2+1*2*8); a30 = *(A+1+0*2+1*2*8);
	            // a21 = *(A+0+1*2+1*2*8); a31 = *(A+1+1*2+1*2*8);
	            // a22 = *(A+0+2*2+1*2*8); a32 = *(A+1+2*2+1*2*8);
	            // a23 = *(A+0+3*2+1*2*8); a33 = *(A+1+3*2+1*2*8);
	            // a24 = *(A+0+4*2+1*2*8); a34 = *(A+1+4*2+1*2*8);
	            // a25 = *(A+0+5*2+1*2*8); a35 = *(A+1+5*2+1*2*8);
	            // a26 = *(A+0+6*2+1*2*8); a36 = *(A+1+6*2+1*2*8);
	            // a27 = *(A+0+7*2+1*2*8); a37 = *(A+1+7*2+1*2*8);
	            //                            
	            // b20 = *(B+0+0*2+1*2*3); b30 = *(B+1+0*2+1*2*3);
	            // b21 = *(B+0+1*2+1*2*3); b31 = *(B+1+1*2+1*2*3);
	            // b22 = *(B+0+2*2+1*2*3); b32 = *(B+1+2*2+1*2*3);
	            //
	            // c00 += a20 * b20; c00 += a30 * b30; c10 += a21 * b20; c10 += a31 * b30; 
	            // c20 += a22 * b20; c20 += a32 * b30; c30 += a23 * b20; c30 += a33 * b30; 
	            // c40 += a24 * b20; c40 += a34 * b30; c50 += a25 * b20; c50 += a35 * b30; 
	            // c60 += a26 * b20; c60 += a36 * b30; c70 += a27 * b20; c70 += a37 * b30; 
	            // c01 += a20 * b21; c01 += a30 * b31; c11 += a21 * b21; c11 += a31 * b31; 
	            // c21 += a22 * b21; c21 += a32 * b31; c31 += a23 * b21; c31 += a33 * b31; 
	            // c41 += a24 * b21; c41 += a34 * b31; c51 += a25 * b21; c51 += a35 * b31; 
	            // c61 += a26 * b21; c61 += a36 * b31; c71 += a27 * b21; c71 += a37 * b31; 
	            // c02 += a20 * b22; c02 += a30 * b32; c12 += a21 * b22; c12 += a31 * b32; 
	            // c22 += a22 * b22; c22 += a32 * b32; c32 += a23 * b22; c32 += a33 * b32; 
	            // c42 += a24 * b22; c42 += a34 * b32; c52 += a25 * b22; c52 += a35 * b32; 
	            // c62 += a26 * b22; c62 += a36 * b32; c72 += a27 * b22; c72 += a37 * b32; 
	            //
	            // a40 = *(A+0+0*2+0*2*8); a50 = *(A+1+0*2+0*2*8);
	            // a41 = *(A+0+1*2+0*2*8); a51 = *(A+1+1*2+0*2*8);
	            // a42 = *(A+0+2*2+0*2*8); a52 = *(A+1+2*2+0*2*8);
	            // a43 = *(A+0+3*2+0*2*8); a53 = *(A+1+3*2+0*2*8);
	            // a44 = *(A+0+4*2+0*2*8); a54 = *(A+1+4*2+0*2*8);
	            // a45 = *(A+0+5*2+0*2*8); a55 = *(A+1+5*2+0*2*8);
	            // a46 = *(A+0+6*2+0*2*8); a56 = *(A+1+6*2+0*2*8);
	            // a47 = *(A+0+7*2+0*2*8); a57 = *(A+1+7*2+0*2*8);
	            //                            
	            // b40 = *(B+0+0*2+2*2*3); b50 = *(B+1+0*2+2*2*3);
	            // b41 = *(B+0+1*2+2*2*3); b51 = *(B+1+1*2+2*2*3);
	            // b42 = *(B+0+2*2+2*2*3); b52 = *(B+1+2*2+2*2*3);
	            //
	            // c00 += a40 * b40; c00 += a50 * b50; c10 += a41 * b40; c10 += a51 * b50; 
	            // c20 += a42 * b40; c20 += a52 * b50; c30 += a43 * b40; c30 += a53 * b50; 
	            // c40 += a44 * b40; c40 += a54 * b50; c50 += a45 * b40; c50 += a55 * b50; 
	            // c60 += a46 * b40; c60 += a56 * b50; c70 += a47 * b40; c70 += a57 * b50; 
	            // c01 += a40 * b41; c01 += a50 * b51; c11 += a41 * b41; c11 += a51 * b51; 
	            // c21 += a42 * b41; c21 += a52 * b51; c31 += a43 * b41; c31 += a53 * b51; 
	            // c41 += a44 * b41; c41 += a54 * b51; c51 += a45 * b41; c51 += a55 * b51; 
	            // c61 += a46 * b41; c61 += a56 * b51; c71 += a47 * b41; c71 += a57 * b51; 
	            // c02 += a40 * b42; c02 += a50 * b52; c12 += a41 * b42; c12 += a51 * b52; 
	            // c22 += a42 * b42; c22 += a52 * b52; c32 += a43 * b42; c32 += a53 * b52; 
	            // c42 += a44 * b42; c42 += a54 * b52; c52 += a45 * b42; c52 += a55 * b52; 
	            // c62 += a46 * b42; c62 += a56 * b52; c72 += a47 * b42; c72 += a57 * b52; 
	            //
	            // a60 = *(A+0+0*2+1*2*8); a70 = *(A+1+0*2+1*2*8);
	            // a61 = *(A+0+1*2+1*2*8); a71 = *(A+1+1*2+1*2*8);
	            // a62 = *(A+0+2*2+1*2*8); a72 = *(A+1+2*2+1*2*8);
	            // a63 = *(A+0+3*2+1*2*8); a73 = *(A+1+3*2+1*2*8);
	            // a64 = *(A+0+4*2+1*2*8); a74 = *(A+1+4*2+1*2*8);
	            // a65 = *(A+0+5*2+1*2*8); a75 = *(A+1+5*2+1*2*8);
	            // a66 = *(A+0+6*2+1*2*8); a76 = *(A+1+6*2+1*2*8);
	            // a67 = *(A+0+7*2+1*2*8); a77 = *(A+1+7*2+1*2*8);
	            //
	            // b60 = *(B+0+0*2+3*2*3); b70 = *(B+1+0*2+3*2*3);
	            // b61 = *(B+0+1*2+3*2*3); b71 = *(B+1+1*2+3*2*3);
	            // b62 = *(B+0+2*2+3*2*3); b72 = *(B+1+2*2+3*2*3);
	            //
	            // c00 += a60 * b60; c00 += a70 * b70; c10 += a61 * b60; c10 += a71 * b70; 
	            // c20 += a62 * b60; c20 += a72 * b70; c30 += a63 * b60; c30 += a73 * b70; 
	            // c40 += a64 * b60; c40 += a74 * b70; c50 += a65 * b60; c50 += a75 * b70; 
	            // c60 += a66 * b60; c60 += a76 * b70; c70 += a67 * b60; c70 += a77 * b70; 
	            // c01 += a60 * b61; c01 += a70 * b71; c11 += a61 * b61; c11 += a71 * b71; 
	            // c21 += a62 * b61; c21 += a72 * b71; c31 += a63 * b61; c31 += a73 * b71; 
	            // c41 += a64 * b61; c41 += a74 * b71; c51 += a65 * b61; c51 += a75 * b71; 
	            // c61 += a66 * b61; c61 += a76 * b71; c71 += a67 * b61; c71 += a77 * b71; 
	            // c02 += a60 * b62; c02 += a70 * b72; c12 += a61 * b62; c12 += a71 * b72; 
	            // c22 += a62 * b62; c22 += a72 * b72; c32 += a63 * b62; c32 += a73 * b72; 
	            // c42 += a64 * b62; c42 += a74 * b72; c52 += a65 * b62; c52 += a75 * b72; 
	            // c62 += a66 * b62; c62 += a76 * b72; c72 += a67 * b62; c72 += a77 * b72; 
	            //
	            //A+=32;
	            //B+=32;

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0   192*8(%[a])\n\t"
	                "prefetcht0    24*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm1 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm2 \n\t" // [b01,b11,b01,b11]
	                "vbroadcastf128   4*8(%[b]), %%ymm3 \n\t" // [b02,b12,b02,b12]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c02,c02,c12,c12]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%ymm0 \n\t" // [a02,a12,a03,a13]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c22,c22,c32,c32]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm0 \n\t" // [a04,a14,a05,a15]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c41,c41,c51,c51]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c42,c42,c52,c52]
	                "\n\t"
	                "vmovapd  12*8(%[a]), %%ymm0 \n\t" // [a06,a16,a07,a17]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c60,c60,c70,c70]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c61,c61,c71,c71]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c62,c62,c72,c72]
	                "\n\t"
	                "prefetcht0   208*8(%[a])\n\t"
	                "prefetcht0    40*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   6*8(%[b]), %%ymm1 \n\t" // [b20,b30,b20,b30]
	                "vbroadcastf128   8*8(%[b]), %%ymm2 \n\t" // [b21,b31,b21,b31]
	                "vbroadcastf128  10*8(%[b]), %%ymm3 \n\t" // [b22,b32,b22,b32]
	                "\n\t"
	                "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a20,a30,a21,a31]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c02,c02,c12,c12]
	                "\n\t"
	                "vmovapd  20*8(%[a]), %%ymm0 \n\t" // [a22,a32,a23,a33]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c22,c22,c32,c32]
	                "\n\t"
	                "vmovapd  24*8(%[a]), %%ymm0 \n\t" // [a24,a34,a25,a35]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c41,c41,c51,c51]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c42,c42,c52,c52]
	                "\n\t"
	                "vmovapd  28*8(%[a]), %%ymm0 \n\t" // [a26,a36,a27,a37]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c60,c60,c70,c70]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c61,c61,c71,c71]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c62,c62,c72,c72]
	                "\n\t"
	                "prefetcht0   224*8(%[a])\n\t"
	                "prefetcht0    56*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128  12*8(%[b]), %%ymm1 \n\t" // [b40,b50,b40,b50]
	                "vbroadcastf128  14*8(%[b]), %%ymm2 \n\t" // [b41,b51,b41,b51]
	                "vbroadcastf128  16*8(%[b]), %%ymm3 \n\t" // [b42,b52,b42,b52]
	                "\n\t"
	                "vmovapd  32*8(%[a]), %%ymm0 \n\t" // [a40,a50,a41,a51]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c02,c02,c12,c12]
	                "\n\t"
	                "vmovapd  36*8(%[a]), %%ymm0 \n\t" // [a42,a52,a43,a53]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c22,c22,c32,c32]
	                "\n\t"
	                "vmovapd  40*8(%[a]), %%ymm0 \n\t" // [a44,a54,a45,a55]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c41,c41,c51,c51]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c42,c42,c52,c52]
	                "\n\t"
	                "vmovapd  44*8(%[a]), %%ymm0 \n\t" // [a46,a56,a47,a57]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c60,c60,c70,c70]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c61,c61,c71,c71]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c62,c62,c72,c72]
	                "\n\t"
	                "prefetcht0   240*8(%[a])\n\t"
	                "prefetcht0    72*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128  18*8(%[b]), %%ymm1 \n\t" // [b60,b70,b60,b70]
	                "vbroadcastf128  20*8(%[b]), %%ymm2 \n\t" // [b61,b71,b61,b71]
	                "vbroadcastf128  22*8(%[b]), %%ymm3 \n\t" // [b62,b72,b62,b72]
	                "\n\t"
	                "vmovapd  48*8(%[a]), %%ymm0 \n\t" // [a60,a70,a61,a71]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c02,c02,c12,c12]
	                "\n\t"
	                "vmovapd  52*8(%[a]), %%ymm0 \n\t" // [a62,a72,a63,a73]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c22,c22,c32,c32]
	                "\n\t"
	                "vmovapd  56*8(%[a]), %%ymm0 \n\t" // [a64,a74,a65,a75]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c41,c41,c51,c51]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c42,c42,c52,c52]
	                "\n\t"
	                "vmovapd  60*8(%[a]), %%ymm0 \n\t" // [a66,a76,a67,a77]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c60,c60,c70,c70]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c61,c61,c71,c71]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c62,c62,c72,c72]
	                "\n\t"
	                "addq  $64*8 , %[a]\n\t"
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

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0    32*8(%[a])\n\t"
	                "prefetcht0    12*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm1 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm2 \n\t" // [b01,b11,b01,b11]
	                "vbroadcastf128   4*8(%[b]), %%ymm3 \n\t" // [b02,b12,b02,b12]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c02,c02,c12,c12]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%ymm0 \n\t" // [a02,a12,a03,a13]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c22,c22,c32,c32]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm0 \n\t" // [a04,a14,a05,a15]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c41,c41,c51,c51]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c42,c42,c52,c52]
	                "\n\t"
	                "vmovapd  12*8(%[a]), %%ymm0 \n\t" // [a06,a16,a07,a17]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c60,c60,c70,c70]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c61,c61,c71,c71]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c62,c62,c72,c72]
	                "\n\t"
	                "prefetcht0    48*8(%[a])\n\t"
	                "prefetcht0    28*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   6*8(%[b]), %%ymm1 \n\t" // [b20,b30,b20,b30]
	                "vbroadcastf128   8*8(%[b]), %%ymm2 \n\t" // [b21,b31,b21,b31]
	                "vbroadcastf128  10*8(%[b]), %%ymm3 \n\t" // [b22,b32,b22,b32]
	                "\n\t"
	                "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a20,a30,a21,a31]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c02,c02,c12,c12]
	                "\n\t"
	                "vmovapd  20*8(%[a]), %%ymm0 \n\t" // [a22,a32,a23,a33]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c22,c22,c32,c32]
	                "\n\t"
	                "vmovapd  24*8(%[a]), %%ymm0 \n\t" // [a24,a34,a25,a35]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c41,c41,c51,c51]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c42,c42,c52,c52]
	                "\n\t"
	                "vmovapd  28*8(%[a]), %%ymm0 \n\t" // [a26,a36,a27,a37]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c60,c60,c70,c70]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c61,c61,c71,c71]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c62,c62,c72,c72]
	                "\n\t"
	                "addq  $32*8 , %[a]\n\t"
	                "addq  $12*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          //}
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm1 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm2 \n\t" // [b01,b11,b01,b11]
	                "vbroadcastf128   4*8(%[b]), %%ymm3 \n\t" // [b02,b12,b02,b12]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c02,c02,c12,c12]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%ymm0 \n\t" // [a02,a12,a03,a13]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c22,c22,c32,c32]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm0 \n\t" // [a04,a14,a05,a15]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c41,c41,c51,c51]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c42,c42,c52,c52]
	                "\n\t"
	                "vmovapd  12*8(%[a]), %%ymm0 \n\t" // [a06,a16,a07,a17]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c60,c60,c70,c70]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c61,c61,c71,c71]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c62,c62,c72,c72]
	                "\n\t"
	                "addq  $16*8 , %[a]\n\t"
	                "addq  $6*8  , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            // b00 = *(B+0+0*2+0*2*3);
	            // b01 = *(B+0+1*2+0*2*3);
	            // b02 = *(B+0+2*2+0*2*3);
	            //                            
	            // a00 = *(A+0+0*2+0*2*8);
	            // a01 = *(A+0+1*2+0*2*8);
	            // c00 += a00 * b00; ; c10 += a01 * b00; 
	            // c01 += a00 * b01; ; c11 += a01 * b01; 
	            // c02 += a00 * b02; ; c12 += a01 * b02; 
	            //
	            // a02 = *(A+0+2*2+0*2*8); 
	            // a03 = *(A+0+3*2+0*2*8); 
	            // c20 += a02 * b00; ; c30 += a03 * b00; 
	            // c21 += a02 * b01; ; c31 += a03 * b01; 
	            // c22 += a02 * b02; ; c32 += a03 * b02; 
	            //
	            // a04 = *(A+0+4*2+0*2*8);
	            // a05 = *(A+0+5*2+0*2*8);
	            // c40 += a04 * b00; ; c50 += a05 * b00; 
	            // c41 += a04 * b01; ; c51 += a05 * b01; 
	            // c42 += a04 * b02; ; c52 += a05 * b02; 
	            //
	            // a06 = *(A+0+6*2+0*2*8);
	            // a07 = *(A+0+7*2+0*2*8);
	            // c60 += a06 * b00; ; c70 += a07 * b00; 
	            // c61 += a06 * b01; ; c71 += a07 * b01; 
	            // c62 += a06 * b02; ; c72 += a07 * b02; 
	
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[b]), %%xmm1 \n\t" // [b00,  0,  0,  0]
	                "vmovsd   1*8(%[b]), %%xmm2 \n\t" // [b01,  0,  0,  0]
	                "vmovsd   2*8(%[b]), %%xmm3 \n\t" // [b02,  0,  0,  0]
	                "\n\t"
	                "vperm2f128 $0x00, %%ymm1 , %%ymm1 , %%ymm1 \n\t" // [b00,  0,b00,  0]
	                "vperm2f128 $0x00, %%ymm2 , %%ymm2 , %%ymm2 \n\t" // [b01,  0,b01,  0]
	                "vperm2f128 $0x00, %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b02,  0,b02,  0]
	                "\n\t"
	                "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t" // [a00,a01,a00,a01]
	                "vshufpd   $0x06, %%ymm0 , %%ymm0 , %%ymm0 \n\t" // [a00,a01,a01,a00]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c02,c02,c12,c12]
	                "\n\t"
	                "vbroadcastf128  2*8(%[a]), %%ymm0 \n\t" // [a02,a03,a02,a03]
	                "vshufpd   $0x06, %%ymm0 , %%ymm0 , %%ymm0 \n\t" // [a02,a03,a03,a02]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c22,c22,c32,c32]
	                "\n\t"
	                "vbroadcastf128  4*8(%[a]), %%ymm0 \n\t" // [a04,a05,a04,a05]
	                "vshufpd   $0x06, %%ymm0 , %%ymm0 , %%ymm0 \n\t" // [a04,a05,a05,a04]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c41,c41,c51,c51]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c42,c42,c52,c52]
	                "\n\t"
	                "vbroadcastf128  6*8(%[a]), %%ymm0 \n\t" // [a06,a07,a06,a07]
	                "vshufpd   $0x06, %%ymm0 , %%ymm0 , %%ymm0 \n\t" // [a06,a07,a07,a06]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c60,c60,c70,c70]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c61,c61,c71,c71]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c62,c62,c72,c72]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $3*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }

	        // %%ymm4  [c00,c00,c10,c10]
	        // %%ymm7  [c20,c20,c30,c30]
	        // %%ymm10 [c40,c40,c50,c50]
	        // %%ymm13 [c60,c60,c70,c70]
	        //
	        // %%ymm5  [c01,c01,c11,c11]
	        // %%ymm8  [c21,c21,c31,c31]
	        // %%ymm11 [c41,c41,c51,c51]
	        // %%ymm14 [c61,c61,c71,c71]
	        //
	        // %%ymm6  [c02,c02,c12,c12]
	        // %%ymm9  [c22,c22,c32,c32]
	        // %%ymm12 [c42,c42,c52,c52]
	        // %%ymm15 [c62,c62,c72,c72]
	
	        __asm__ __volatile__ (
	            "\n\t"
	            "vperm2f128 $0x20, %%ymm7 , %%ymm4 , %%ymm0 \n\t" // [c00,c00,c20,c20]
	            "vperm2f128 $0x31, %%ymm7 , %%ymm4 , %%ymm1 \n\t" // [c10,c10,c30,c30]
	            "vperm2f128 $0x20, %%ymm13, %%ymm10, %%ymm2 \n\t" // [c40,c40,c60,c60]
	            "vperm2f128 $0x31, %%ymm13, %%ymm10, %%ymm3 \n\t" // [c50,c50,c70,c70]
	            "\n\t"
	            "vshufpd    $0x00, %%ymm1 , %%ymm0 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	            "vshufpd    $0x0f, %%ymm1 , %%ymm0 , %%ymm7 \n\t" // [c00,c10,c20,c30]
	            "vshufpd    $0x00, %%ymm3 , %%ymm2 , %%ymm10\n\t" // [c40,c50,c60,c70]
	            "vshufpd    $0x0f, %%ymm3 , %%ymm2 , %%ymm13\n\t" // [c40,c50,c60,c70]
	            "\n\t"
	            "vaddpd            %%ymm4 , %%ymm7 , %%ymm7 \n\t" // [c00,c10,c20,c30]
	            "vaddpd            %%ymm10, %%ymm13, %%ymm13\n\t" // [c40,c50,c60,c70]
	            "\n\t"
	            "vperm2f128 $0x20, %%ymm8 , %%ymm5 , %%ymm0 \n\t" // [c01,c01,c21,c21]
	            "vperm2f128 $0x31, %%ymm8 , %%ymm5 , %%ymm1 \n\t" // [c11,c11,c31,c31]
	            "vperm2f128 $0x20, %%ymm14, %%ymm11, %%ymm2 \n\t" // [c41,c41,c61,c61]
	            "vperm2f128 $0x31, %%ymm14, %%ymm11, %%ymm3 \n\t" // [c51,c51,c71,c71]
	            "\n\t"
	            "vshufpd    $0x00, %%ymm1 , %%ymm0 , %%ymm5 \n\t" // [c01,c11,c21,c31]
	            "vshufpd    $0x0f, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [c01,c11,c21,c31]
	            "vshufpd    $0x00, %%ymm3 , %%ymm2 , %%ymm11\n\t" // [c41,c51,c61,c71]
	            "vshufpd    $0x0f, %%ymm3 , %%ymm2 , %%ymm14\n\t" // [c41,c51,c61,c71]
	            "\n\t"
	            "vaddpd            %%ymm5 , %%ymm8 , %%ymm8 \n\t" // [c01,c11,c21,c31]
	            "vaddpd            %%ymm11, %%ymm14, %%ymm14\n\t" // [c41,c51,c61,c71]
	            "\n\t"
	            "vperm2f128 $0x20, %%ymm9 , %%ymm6 , %%ymm0 \n\t" // [c00,c00,c20,c20]
	            "vperm2f128 $0x31, %%ymm9 , %%ymm6 , %%ymm1 \n\t" // [c10,c10,c30,c30]
	            "vperm2f128 $0x20, %%ymm15, %%ymm12, %%ymm2 \n\t" // [c40,c40,c60,c60]
	            "vperm2f128 $0x31, %%ymm15, %%ymm12, %%ymm3 \n\t" // [c50,c50,c70,c70]
	            "\n\t"
	            "vshufpd    $0x00, %%ymm1 , %%ymm0 , %%ymm6 \n\t" // [c00,c10,c20,c30]
	            "vshufpd    $0x0f, %%ymm1 , %%ymm0 , %%ymm9 \n\t" // [c00,c10,c20,c30]
	            "vshufpd    $0x00, %%ymm3 , %%ymm2 , %%ymm12\n\t" // [c40,c50,c60,c70]
	            "vshufpd    $0x0f, %%ymm3 , %%ymm2 , %%ymm15\n\t" // [c40,c50,c60,c70]
	            "\n\t"
	            "vaddpd            %%ymm6 , %%ymm9 , %%ymm9 \n\t" // [c00,c10,c20,c30]
	            "vaddpd            %%ymm12, %%ymm15, %%ymm15\n\t" // [c40,c50,c60,c70]
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%ymm0, %%ymm7 \n\t"
	            "vfmadd213pd 4*8(%[c0]          ), %%ymm0, %%ymm13\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc1]  ), %%ymm0, %%ymm8 \n\t"
	            "vfmadd213pd 4*8(%[c0],%[ldc1]  ), %%ymm0, %%ymm14\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc1],2), %%ymm0, %%ymm9 \n\t"
	            "vfmadd213pd 4*8(%[c0],%[ldc1],2), %%ymm0, %%ymm15\n\t"
	            "\n\t"
	            "vmovupd  %%ymm7 , 0*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm13, 4*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm8 , 0*8(%[c0],%[ldc1]  )\n\t"
	            "vmovupd  %%ymm14, 4*8(%[c0],%[ldc1]  )\n\t"
	            "vmovupd  %%ymm9 , 0*8(%[c0],%[ldc1],2)\n\t"
	            "vmovupd  %%ymm15, 4*8(%[c0],%[ldc1],2)\n\t"
	            "\n\t"
	            "addq  $8*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[alpha]"m"(alpha4),[ldc1]"r"(ldc1)
	        );

	        B = B - 3*K;

	      }
	    }
	    if( M & 4 ){
	    //kif( M >> 2 ){
	    //k  size_t m4 = ( M >> 2 ); // unrolling M
	    //k  while( m4-- ){

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
	            "\n\t"
	            "prefetcht0 0*8(%[c0]          )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc1]  )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc1],2)\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[ldc1]"r"(ldc1)
	        );

	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0    96*8(%[a])\n\t"
	                "prefetcht0    24*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm1 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm2 \n\t" // [b01,b11,b01,b11]
	                "vbroadcastf128   4*8(%[b]), %%ymm3 \n\t" // [b02,b12,b02,b12]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm4 \n\t" // [a02,a12,a03,a13]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c02,c02,c12,c12]
	                "vfmadd231pd  %%ymm4 , %%ymm1 , %%ymm13\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm4 , %%ymm2 , %%ymm14\n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm4 , %%ymm3 , %%ymm15\n\t" // [c22,c22,c32,c32]
	                "\n\t"
	                "prefetcht0   112*8(%[a])\n\t"
	                "prefetcht0    40*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   6*8(%[b]), %%ymm6 \n\t" // [b20,b30,b20,b30]
	                "vbroadcastf128   8*8(%[b]), %%ymm7 \n\t" // [b21,b31,b21,b31]
	                "vbroadcastf128  10*8(%[b]), %%ymm8 \n\t" // [b22,b32,b22,b32]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm5 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm9 \n\t" // [a22,a32,a23,a33]
	                "\n\t"
	                "vfmadd231pd  %%ymm5 , %%ymm6 , %%ymm10\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm5 , %%ymm7 , %%ymm11\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm5 , %%ymm8 , %%ymm12\n\t" // [c02,c02,c12,c12]
	                "vfmadd231pd  %%ymm9 , %%ymm6 , %%ymm13\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm9 , %%ymm7 , %%ymm14\n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm9 , %%ymm8 , %%ymm15\n\t" // [c22,c22,c32,c32]
	                "\n\t"
	                "prefetcht0   128*8(%[a])\n\t"
	                "prefetcht0    56*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128  12*8(%[b]), %%ymm1 \n\t" // [b40,b50,b40,b50]
	                "vbroadcastf128  14*8(%[b]), %%ymm2 \n\t" // [b41,b51,b41,b51]
	                "vbroadcastf128  16*8(%[b]), %%ymm3 \n\t" // [b42,b52,b42,b52]
	                "\n\t"
	                "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a40,a50,a41,a51]
	                "vmovapd  20*8(%[a]), %%ymm4 \n\t" // [a42,a52,a43,a53]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c02,c02,c12,c12]
	                "vfmadd231pd  %%ymm4 , %%ymm1 , %%ymm13\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm4 , %%ymm2 , %%ymm14\n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm4 , %%ymm3 , %%ymm15\n\t" // [c22,c22,c32,c32]
	                "\n\t"
	                "prefetcht0   144*8(%[a])\n\t"
	                "prefetcht0    72*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128  18*8(%[b]), %%ymm6 \n\t" // [b60,b70,b60,b70]
	                "vbroadcastf128  20*8(%[b]), %%ymm7 \n\t" // [b61,b71,b61,b71]
	                "vbroadcastf128  22*8(%[b]), %%ymm8 \n\t" // [b62,b72,b62,b72]
	                "\n\t"
	                "vmovapd  24*8(%[a]), %%ymm5 \n\t" // [a60,a70,a61,a71]
	                "vmovapd  28*8(%[a]), %%ymm9 \n\t" // [a62,a72,a63,a73]
	                "\n\t"
	                "vfmadd231pd  %%ymm5 , %%ymm6 , %%ymm10\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm5 , %%ymm7 , %%ymm11\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm5 , %%ymm8 , %%ymm12\n\t" // [c02,c02,c12,c12]
	                "vfmadd231pd  %%ymm9 , %%ymm6 , %%ymm13\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm9 , %%ymm7 , %%ymm14\n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm9 , %%ymm8 , %%ymm15\n\t" // [c22,c22,c32,c32]
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

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0    16*8(%[a])\n\t"
	                "prefetcht0    12*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm1 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm2 \n\t" // [b01,b11,b01,b11]
	                "vbroadcastf128   4*8(%[b]), %%ymm3 \n\t" // [b02,b12,b02,b12]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm4 \n\t" // [a02,a12,a03,a13]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c02,c02,c12,c12]
	                "vfmadd231pd  %%ymm4 , %%ymm1 , %%ymm13\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm4 , %%ymm2 , %%ymm14\n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm4 , %%ymm3 , %%ymm15\n\t" // [c22,c22,c32,c32]
	                "\n\t"
	                "prefetcht0    32*8(%[a])\n\t"
	                "prefetcht0    28*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   6*8(%[b]), %%ymm6 \n\t" // [b20,b30,b20,b30]
	                "vbroadcastf128   8*8(%[b]), %%ymm7 \n\t" // [b21,b31,b21,b31]
	                "vbroadcastf128  10*8(%[b]), %%ymm8 \n\t" // [b22,b32,b22,b32]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm5 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm9 \n\t" // [a22,a32,a23,a33]
	                "\n\t"
	                "vfmadd231pd  %%ymm5 , %%ymm6 , %%ymm10\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm5 , %%ymm7 , %%ymm11\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm5 , %%ymm8 , %%ymm12\n\t" // [c02,c02,c12,c12]
	                "vfmadd231pd  %%ymm9 , %%ymm6 , %%ymm13\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm9 , %%ymm7 , %%ymm14\n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm9 , %%ymm8 , %%ymm15\n\t" // [c22,c22,c32,c32]
	                "\n\t"
	                "addq  $16*8 , %[a]\n\t"
	                "addq  $12*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          //}
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm1 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm2 \n\t" // [b01,b11,b01,b11]
	                "vbroadcastf128   4*8(%[b]), %%ymm3 \n\t" // [b02,b12,b02,b12]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm4 \n\t" // [a02,a12,a03,a13]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c02,c02,c12,c12]
	                "vfmadd231pd  %%ymm4 , %%ymm1 , %%ymm13\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm4 , %%ymm2 , %%ymm14\n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm4 , %%ymm3 , %%ymm15\n\t" // [c22,c22,c32,c32]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $6*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	       
	        }
	        if( K & 1 ){

	            // b00 = *(B+0+0*2+0*2*3);
	            // b01 = *(B+0+1*2+0*2*3);
	            // b02 = *(B+0+2*2+0*2*3);
	            //                            
	            // a00 = *(A+0+0*2+0*2*8);
	            // a01 = *(A+0+1*2+0*2*8);
	            // c00 += a00 * b00; ; c10 += a01 * b00; 
	            // c01 += a00 * b01; ; c11 += a01 * b01; 
	            // c02 += a00 * b02; ; c12 += a01 * b02; 
	            //
	            // a02 = *(A+0+2*2+0*2*8); 
	            // a03 = *(A+0+3*2+0*2*8); 
	            // c20 += a02 * b00; ; c30 += a03 * b00; 
	            // c21 += a02 * b01; ; c31 += a03 * b01; 
	            // c22 += a02 * b02; ; c32 += a03 * b02; 
	            //
	
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[b]), %%xmm1 \n\t" // [b00,  0,  0,  0]
	                "vmovsd   1*8(%[b]), %%xmm2 \n\t" // [b01,  0,  0,  0]
	                "vmovsd   2*8(%[b]), %%xmm3 \n\t" // [b02,  0,  0,  0]
	                "\n\t"
	                "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t" // [a00,a01,a00,a01]
	                "vbroadcastf128  2*8(%[a]), %%ymm4 \n\t" // [a02,a03,a02,a03]
	                "\n\t"
	                "vperm2f128 $0x00, %%ymm1 , %%ymm1 , %%ymm1 \n\t" // [b00,  0,b00,  0]
	                "vperm2f128 $0x00, %%ymm2 , %%ymm2 , %%ymm2 \n\t" // [b01,  0,b01,  0]
	                "vperm2f128 $0x00, %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b02,  0,b02,  0]
	                "\n\t"
	                "vshufpd   $0x06, %%ymm0 , %%ymm0 , %%ymm0 \n\t" // [a00,a01,a01,a00]
	                "vshufpd   $0x06, %%ymm4 , %%ymm4 , %%ymm4 \n\t" // [a02,a03,a03,a02]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c02,c02,c12,c12]
	                "vfmadd231pd  %%ymm4 , %%ymm1 , %%ymm13\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm4 , %%ymm2 , %%ymm14\n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm4 , %%ymm3 , %%ymm15\n\t" // [c22,c22,c32,c32]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $3*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }

	        // %%ymm10 [c00,c00,c10,c10]
	        // %%ymm11 [c01,c01,c11,c11]
	        // %%ymm12 [c02,c02,c12,c12]
	        // %%ymm13 [c20,c20,c30,c30]
	        // %%ymm14 [c21,c21,c31,c31]
	        // %%ymm15 [c22,c22,c32,c32]
	
	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm6\n\t"
	            "\n\t"
	            "vperm2f128 $0x20, %%ymm13, %%ymm10, %%ymm0 \n\t" // [c00,c00,c20,c20]
	            "vperm2f128 $0x31, %%ymm13, %%ymm10, %%ymm1 \n\t" // [c10,c10,c30,c30]
	            "vperm2f128 $0x20, %%ymm14, %%ymm11, %%ymm2 \n\t" // [c01,c01,c21,c21]
	            "vperm2f128 $0x31, %%ymm14, %%ymm11, %%ymm3 \n\t" // [c11,c11,c31,c31]
	            "vperm2f128 $0x20, %%ymm15, %%ymm12, %%ymm4 \n\t" // [c00,c00,c20,c20]
	            "vperm2f128 $0x31, %%ymm15, %%ymm12, %%ymm5 \n\t" // [c10,c10,c30,c30]
	            "\n\t"
	            "vshufpd    $0x00, %%ymm1 , %%ymm0 , %%ymm10\n\t" // [c00,c10,c20,c30]
	            "vshufpd    $0x0f, %%ymm1 , %%ymm0 , %%ymm13\n\t" // [c00,c10,c20,c30]
	            "vshufpd    $0x00, %%ymm3 , %%ymm2 , %%ymm11\n\t" // [c01,c11,c21,c31]
	            "vshufpd    $0x0f, %%ymm3 , %%ymm2 , %%ymm14\n\t" // [c01,c11,c21,c31]
	            "vshufpd    $0x00, %%ymm5 , %%ymm4 , %%ymm12\n\t" // [c00,c10,c20,c30]
	            "vshufpd    $0x0f, %%ymm5 , %%ymm4 , %%ymm15\n\t" // [c00,c10,c20,c30]
	            "\n\t"
	            "vaddpd            %%ymm10, %%ymm13, %%ymm13\n\t" // [c00,c10,c20,c30]
	            "vaddpd            %%ymm11, %%ymm14, %%ymm14\n\t" // [c01,c11,c21,c31]
	            "vaddpd            %%ymm12, %%ymm15, %%ymm15\n\t" // [c00,c10,c20,c30]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%ymm6, %%ymm13\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc1]  ), %%ymm6, %%ymm14\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc1],2), %%ymm6, %%ymm15\n\t"
	            "\n\t"
	            "vmovupd  %%ymm13, 0*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm14, 0*8(%[c0],%[ldc1]  )\n\t"
	            "vmovupd  %%ymm15, 0*8(%[c0],%[ldc1],2)\n\t"
	            "\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[alpha]"m"(alpha4),[ldc1]"r"(ldc1)
	        );

	        B = B - 3*K;

	    //  }
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
	            "\n\t"
	            "prefetcht0 0*8(%[c0]          )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc1]  )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc1],2)\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[ldc1]"r"(ldc1)
	        );

	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0    48*8(%[a])\n\t"
	                "prefetcht0    24*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm1 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm2 \n\t" // [b01,b11,b01,b11]
	                "vbroadcastf128   4*8(%[b]), %%ymm3 \n\t" // [b02,b12,b02,b12]
	                "vbroadcastf128   6*8(%[b]), %%ymm5 \n\t" // [b20,b30,b20,b30]
	                "vbroadcastf128   8*8(%[b]), %%ymm6 \n\t" // [b21,b31,b21,b31]
	                "vbroadcastf128  10*8(%[b]), %%ymm7 \n\t" // [b22,b32,b22,b32]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm4 \n\t" // [a20,a30,a21,a31]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c02,c02,c12,c12]
	                "vfmadd231pd  %%ymm4 , %%ymm5 , %%ymm13\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm4 , %%ymm6 , %%ymm14\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm4 , %%ymm7 , %%ymm15\n\t" // [c02,c02,c12,c12]
	                "\n\t"
	                "prefetcht0    64*8(%[a])\n\t"
	                "prefetcht0    40*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128  12*8(%[b]), %%ymm1 \n\t" // [b40,b50,b40,b50]
	                "vbroadcastf128  14*8(%[b]), %%ymm2 \n\t" // [b41,b51,b41,b51]
	                "vbroadcastf128  16*8(%[b]), %%ymm3 \n\t" // [b42,b52,b42,b52]
	                "vbroadcastf128  18*8(%[b]), %%ymm5 \n\t" // [b60,b70,b60,b70]
	                "vbroadcastf128  20*8(%[b]), %%ymm6 \n\t" // [b61,b71,b61,b71]
	                "vbroadcastf128  22*8(%[b]), %%ymm7 \n\t" // [b62,b72,b62,b72]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm0 \n\t" // [a40,a50,a41,a51]
	                "vmovapd  12*8(%[a]), %%ymm4 \n\t" // [a60,a70,a61,a71]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c02,c02,c12,c12]
	                "vfmadd231pd  %%ymm4 , %%ymm5 , %%ymm13\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm4 , %%ymm6 , %%ymm14\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm4 , %%ymm7 , %%ymm15\n\t" // [c02,c02,c12,c12]
	                "\n\t"
	                "addq  $16*8 , %[a]\n\t"
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

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0    12*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm1 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm2 \n\t" // [b01,b11,b01,b11]
	                "vbroadcastf128   4*8(%[b]), %%ymm3 \n\t" // [b02,b12,b02,b12]
	                "vbroadcastf128   6*8(%[b]), %%ymm5 \n\t" // [b20,b30,b20,b30]
	                "vbroadcastf128   8*8(%[b]), %%ymm6 \n\t" // [b21,b31,b21,b31]
	                "vbroadcastf128  10*8(%[b]), %%ymm7 \n\t" // [b22,b32,b22,b32]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm4 \n\t" // [a20,a30,a21,a31]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c02,c02,c12,c12]
	                "vfmadd231pd  %%ymm4 , %%ymm5 , %%ymm13\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm4 , %%ymm6 , %%ymm14\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm4 , %%ymm7 , %%ymm15\n\t" // [c02,c02,c12,c12]
	                "\n\t"
	                "addq  $8*8  , %[a]\n\t"
	                "addq  $12*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          //}
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm1 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm2 \n\t" // [b01,b11,b01,b11]
	                "vbroadcastf128   4*8(%[b]), %%ymm3 \n\t" // [b02,b12,b02,b12]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c02,c02,c12,c12]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $6*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){

	            // b00 = *(B+0+0*2+0*2*3);
	            // b01 = *(B+0+1*2+0*2*3);
	            // b02 = *(B+0+2*2+0*2*3);
	            //                            
	            // a00 = *(A+0+0*2+0*2*8);
	            // a01 = *(A+0+1*2+0*2*8);
	            // c00 += a00 * b00; ; c10 += a01 * b00; 
	            // c01 += a00 * b01; ; c11 += a01 * b01; 
	            // c02 += a00 * b02; ; c12 += a01 * b02; 
	            //
	            // a02 = *(A+0+2*2+0*2*8); 
	            // a03 = *(A+0+3*2+0*2*8); 
	            // c20 += a02 * b00; ; c30 += a03 * b00; 
	            // c21 += a02 * b01; ; c31 += a03 * b01; 
	            // c22 += a02 * b02; ; c32 += a03 * b02; 
	            //
	
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[b]), %%xmm1 \n\t" // [b00,  0,  0,  0]
	                "vmovsd   1*8(%[b]), %%xmm2 \n\t" // [b01,  0,  0,  0]
	                "vmovsd   2*8(%[b]), %%xmm3 \n\t" // [b02,  0,  0,  0]
	                "\n\t"
	                "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t" // [a00,a01,a00,a01]
	                "\n\t"
	                "vperm2f128 $0x00, %%ymm1 , %%ymm1 , %%ymm1 \n\t" // [b00,  0,b00,  0]
	                "vperm2f128 $0x00, %%ymm2 , %%ymm2 , %%ymm2 \n\t" // [b01,  0,b01,  0]
	                "vperm2f128 $0x00, %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b02,  0,b02,  0]
	                "\n\t"
	                "vshufpd   $0x06, %%ymm0 , %%ymm0 , %%ymm0 \n\t" // [a00,a01,a01,a00]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c02,c02,c12,c12]
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $3*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }

	        // %%ymm13 [c00,c00,c10,c10]
	        // %%ymm14 [c01,c01,c11,c11]
	        // %%ymm15 [c02,c02,c12,c12]
	
	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm6\n\t"
	            "\n\t"
	            "vshufpd    $0x05, %%ymm13, %%ymm13, %%ymm10\n\t" // [c00,c00,c10,c10]
	            "vshufpd    $0x05, %%ymm14, %%ymm14, %%ymm11\n\t" // [c01,c01,c11,c11]
	            "vshufpd    $0x05, %%ymm15, %%ymm15, %%ymm12\n\t" // [c02,c02,c12,c12]
	            "vaddpd            %%ymm10, %%ymm13, %%ymm13\n\t" // [c00,c00,c10,c10]
	            "vaddpd            %%ymm11, %%ymm14, %%ymm14\n\t" // [c01,c01,c11,c11]
	            "vaddpd            %%ymm12, %%ymm15, %%ymm15\n\t" // [c02,c02,c12,c12]
	            "\n\t"
	            "vperm2f128 $0x01, %%ymm13, %%ymm13, %%ymm10\n\t" // [c10,c10,c00,c00]
	            "vperm2f128 $0x01, %%ymm14, %%ymm14, %%ymm11\n\t" // [c11,c11,c01,c01]
	            "vperm2f128 $0x01, %%ymm15, %%ymm15, %%ymm12\n\t" // [c12,c12,c02,c02]
	            "\n\t"
	            "vshufpd    $0x00, %%ymm10, %%ymm13, %%ymm13\n\t" // [c00,c10,c10,c00]
	            "vshufpd    $0x00, %%ymm11, %%ymm14, %%ymm14\n\t" // [c01,c11,c11,c01]
	            "vshufpd    $0x00, %%ymm12, %%ymm15, %%ymm15\n\t" // [c02,c12,c12,c02]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%xmm6, %%xmm13\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc1]  ), %%xmm6, %%xmm14\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc1],2), %%xmm6, %%xmm15\n\t"
	            "\n\t"
	            "vmovupd  %%xmm13, 0*8(%[c0]          )\n\t"
	            "vmovupd  %%xmm14, 0*8(%[c0],%[ldc1]  )\n\t"
	            "vmovupd  %%xmm15, 0*8(%[c0],%[ldc1],2)\n\t"
	            "\n\t"
	            "addq  $2*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[alpha]"m"(alpha4),[ldc1]"r"(ldc1)
	        );

	        B = B - 3*K;

	    }
	    if( M & 1 ){

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
	            "\n\t"
	            "prefetcht0 0*8(%[c0]          )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc1]  )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc1],2)\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[ldc1]"r"(ldc1)
	        );

	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0    24*8(%[b])\n\t"
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%xmm1 \n\t" // [b00,b10,  0,  0]
	                "vmovupd   2*8(%[b]), %%xmm2 \n\t" // [b01,b11,  0,  0]
	                "vmovupd   4*8(%[b]), %%xmm3 \n\t" // [b02,b12,  0,  0]
	                "vmovupd   6*8(%[b]), %%xmm5 \n\t" // [b20,b30,  0,  0]
	                "vmovupd   8*8(%[b]), %%xmm6 \n\t" // [b21,b31,  0,  0]
	                "vmovupd  10*8(%[b]), %%xmm7 \n\t" // [b22,b32,  0,  0]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,  0,  0]
	                "vmovapd   2*8(%[a]), %%xmm4 \n\t" // [a20,a30,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c00,c00,  0,  0]
	                "vfmadd231pd  %%xmm0 , %%xmm2 , %%xmm14\n\t" // [c01,c01,  0,  0]
	                "vfmadd231pd  %%xmm0 , %%xmm3 , %%xmm15\n\t" // [c02,c02,  0,  0]
	                "vfmadd231pd  %%xmm4 , %%xmm5 , %%xmm13\n\t" // [c00,c00,  0,  0]
	                "vfmadd231pd  %%xmm4 , %%xmm6 , %%xmm14\n\t" // [c01,c01,  0,  0]
	                "vfmadd231pd  %%xmm4 , %%xmm7 , %%xmm15\n\t" // [c02,c02,  0,  0]
	                "\n\t"
	                "prefetcht0    40*8(%[b])\n\t"
	                "\n\t"
	                "vmovupd  12*8(%[b]), %%xmm1 \n\t" // [b40,b50,  0,  0]
	                "vmovupd  14*8(%[b]), %%xmm2 \n\t" // [b41,b51,  0,  0]
	                "vmovupd  16*8(%[b]), %%xmm3 \n\t" // [b42,b52,  0,  0]
	                "vmovupd  18*8(%[b]), %%xmm5 \n\t" // [b60,b70,  0,  0]
	                "vmovupd  20*8(%[b]), %%xmm6 \n\t" // [b61,b71,  0,  0]
	                "vmovupd  22*8(%[b]), %%xmm7 \n\t" // [b62,b72,  0,  0]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%xmm0 \n\t" // [a40,a50,  0,  0]
	                "vmovapd   6*8(%[a]), %%xmm4 \n\t" // [a60,a70,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c00,c00,  0,  0]
	                "vfmadd231pd  %%xmm0 , %%xmm2 , %%xmm14\n\t" // [c01,c01,  0,  0]
	                "vfmadd231pd  %%xmm0 , %%xmm3 , %%xmm15\n\t" // [c02,c02,  0,  0]
	                "vfmadd231pd  %%xmm4 , %%xmm5 , %%xmm13\n\t" // [c00,c00,  0,  0]
	                "vfmadd231pd  %%xmm4 , %%xmm6 , %%xmm14\n\t" // [c01,c01,  0,  0]
	                "vfmadd231pd  %%xmm4 , %%xmm7 , %%xmm15\n\t" // [c02,c02,  0,  0]
	                "\n\t"
	                "addq  $8*8  , %[a]\n\t"
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

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0    12*8(%[b])\n\t"
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%xmm1 \n\t" // [b00,b10,  0,  0]
	                "vmovupd   2*8(%[b]), %%xmm2 \n\t" // [b01,b11,  0,  0]
	                "vmovupd   4*8(%[b]), %%xmm3 \n\t" // [b02,b12,  0,  0]
	                "vmovupd   6*8(%[b]), %%xmm5 \n\t" // [b20,b30,  0,  0]
	                "vmovupd   8*8(%[b]), %%xmm6 \n\t" // [b21,b31,  0,  0]
	                "vmovupd  10*8(%[b]), %%xmm7 \n\t" // [b22,b32,  0,  0]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,  0,  0]
	                "vmovapd   2*8(%[a]), %%xmm4 \n\t" // [a20,a30,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c00,c00,  0,  0]
	                "vfmadd231pd  %%xmm0 , %%xmm2 , %%xmm14\n\t" // [c01,c01,  0,  0]
	                "vfmadd231pd  %%xmm0 , %%xmm3 , %%xmm15\n\t" // [c02,c02,  0,  0]
	                "vfmadd231pd  %%xmm4 , %%xmm5 , %%xmm13\n\t" // [c00,c00,  0,  0]
	                "vfmadd231pd  %%xmm4 , %%xmm6 , %%xmm14\n\t" // [c01,c01,  0,  0]
	                "vfmadd231pd  %%xmm4 , %%xmm7 , %%xmm15\n\t" // [c02,c02,  0,  0]
	                "\n\t"
	                "addq  $4*8  , %[a]\n\t"
	                "addq  $12*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          //}
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%xmm1 \n\t" // [b00,b10,  0,  0]
	                "vmovupd   2*8(%[b]), %%xmm2 \n\t" // [b01,b11,  0,  0]
	                "vmovupd   4*8(%[b]), %%xmm3 \n\t" // [b02,b12,  0,  0]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c00,c00,  0,  0]
	                "vfmadd231pd  %%xmm0 , %%xmm2 , %%xmm14\n\t" // [c01,c01,  0,  0]
	                "vfmadd231pd  %%xmm0 , %%xmm3 , %%xmm15\n\t" // [c02,c02,  0,  0]
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $6*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[b]), %%xmm1 \n\t" // [b00,  0,  0,  0]
	                "vmovsd   1*8(%[b]), %%xmm2 \n\t" // [b01,  0,  0,  0]
	                "vmovsd   2*8(%[b]), %%xmm3 \n\t" // [b02,  0,  0,  0]
	                "\n\t"
	                "vmovsd   0*8(%[a]), %%xmm0 \n\t" // [a00,  0,  0,  0]
	                "\n\t"
	                "vfmadd231sd  %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c00,c00,  0,  0]
	                "vfmadd231sd  %%xmm0 , %%xmm2 , %%xmm14\n\t" // [c01,c01,  0,  0]
	                "vfmadd231sd  %%xmm0 , %%xmm3 , %%xmm15\n\t" // [c02,c02,  0,  0]
	                "\n\t"
	                "addq  $1*8 , %[a]\n\t"
	                "addq  $3*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }

	        // %%ymm13 [c00,c00,  0,  0]
	        // %%ymm14 [c01,c01,  0,  0]
	        // %%ymm15 [c02,c02,  0,  0]
	
	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm6\n\t"
	            "\n\t"
	            "vshufpd    $0x05, %%ymm13, %%ymm13, %%ymm10\n\t" // [c00,c00,  0,  0]
	            "vshufpd    $0x05, %%ymm14, %%ymm14, %%ymm11\n\t" // [c01,c01,  0,  0]
	            "vshufpd    $0x05, %%ymm15, %%ymm15, %%ymm12\n\t" // [c02,c02,  0,  0]
	            "vaddpd            %%ymm10, %%ymm13, %%ymm13\n\t" // [c00,---,  0,  0]
	            "vaddpd            %%ymm11, %%ymm14, %%ymm14\n\t" // [c01,---,  0,  0]
	            "vaddpd            %%ymm12, %%ymm15, %%ymm15\n\t" // [c02,---,  0,  0]
	            "\n\t"
	            "vfmadd213sd 0*8(%[c0]          ), %%xmm6, %%xmm13\n\t"
	            "vfmadd213sd 0*8(%[c0],%[ldc1]  ), %%xmm6, %%xmm14\n\t"
	            "vfmadd213sd 0*8(%[c0],%[ldc1],2), %%xmm6, %%xmm15\n\t"
	            "\n\t"
	            "vmovsd  %%xmm13, 0*8(%[c0]          )\n\t"
	            "vmovsd  %%xmm14, 0*8(%[c0],%[ldc1]  )\n\t"
	            "vmovsd  %%xmm15, 0*8(%[c0],%[ldc1],2)\n\t"
	            "\n\t"
	            "addq  $1*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[alpha]"m"(alpha4),[ldc1]"r"(ldc1)
	        );

	        B = B - 3*K;

	    }

	    A = A - M*K;
	    B = B + 3*K;
	    c0 = c0- M + 3*ldc;
	  }
	}
	if( N3R==2 ){

	    if( M >> 3 ){
	      size_t m8 = ( M >> 3 ); // unrolling M
	      while( m8-- ){

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

	        __asm__ __volatile__ (
	            "\n\t"
	            "prefetcht0 0*8(%[c0]          )\n\t"
	            "prefetcht0 4*8(%[c0]          )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc1]  )\n\t"
	            "prefetcht0 4*8(%[c0],%[ldc1]  )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[ldc1]"r"(ldc1)
	        );


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){
	            //
	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0   192*8(%[a])\n\t"
	                "prefetcht0    16*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm5 \n\t" // [b01,b11,b01,b11]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a04,a14,a05,a15]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a06,a16,a07,a17]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm9 \n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm10\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm11\n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm12\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm13\n\t" // [c41,c41,c51,c51]
	                "vfmadd231pd  %%ymm3 , %%ymm4 , %%ymm14\n\t" // [c60,c60,c70,c70]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm15\n\t" // [c61,c61,c71,c71]
	                "\n\t"
	                "prefetcht0   208*8(%[a])\n\t"
	                "prefetcht0    32*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   4*8(%[b]), %%ymm6 \n\t" // [b20,b30,b20,b30]
	                "vbroadcastf128   6*8(%[b]), %%ymm7 \n\t" // [b21,b31,b21,b31]
	                "\n\t"
	                "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  20*8(%[a]), %%ymm1 \n\t" // [a22,a32,a23,a33]
	                "vmovapd  24*8(%[a]), %%ymm2 \n\t" // [a24,a34,a25,a35]
	                "vmovapd  28*8(%[a]), %%ymm3 \n\t" // [a26,a36,a27,a37]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm8 \n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm7 , %%ymm9 \n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm10\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm11\n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm12\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm13\n\t" // [c41,c41,c51,c51]
	                "vfmadd231pd  %%ymm3 , %%ymm6 , %%ymm14\n\t" // [c60,c60,c70,c70]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c61,c61,c71,c71]
	                "\n\t"
	                "prefetcht0   224*8(%[a])\n\t"
	                "prefetcht0    48*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   8*8(%[b]), %%ymm4 \n\t" // [b40,b50,b40,b50]
	                "vbroadcastf128  10*8(%[b]), %%ymm5 \n\t" // [b41,b51,b41,b51]
	                "\n\t"
	                "vmovapd  32*8(%[a]), %%ymm0 \n\t" // [a40,a50,a41,a51]
	                "vmovapd  36*8(%[a]), %%ymm1 \n\t" // [a42,a52,a43,a53]
	                "vmovapd  40*8(%[a]), %%ymm2 \n\t" // [a44,a54,a45,a55]
	                "vmovapd  44*8(%[a]), %%ymm3 \n\t" // [a46,a56,a47,a57]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm9 \n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm10\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm11\n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm12\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm13\n\t" // [c41,c41,c51,c51]
	                "vfmadd231pd  %%ymm3 , %%ymm4 , %%ymm14\n\t" // [c60,c60,c70,c70]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm15\n\t" // [c61,c61,c71,c71]
	                "\n\t"
	                "prefetcht0   240*8(%[a])\n\t"
	                "prefetcht0    64*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128  12*8(%[b]), %%ymm6 \n\t" // [b60,b70,b60,b70]
	                "vbroadcastf128  14*8(%[b]), %%ymm7 \n\t" // [b61,b71,b61,b71]
	                "\n\t"
	                "vmovapd  48*8(%[a]), %%ymm0 \n\t" // [a60,a70,a61,a71]
	                "vmovapd  52*8(%[a]), %%ymm1 \n\t" // [a62,a72,a63,a73]
	                "vmovapd  56*8(%[a]), %%ymm2 \n\t" // [a64,a74,a65,a75]
	                "vmovapd  60*8(%[a]), %%ymm3 \n\t" // [a66,a76,a67,a77]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm8 \n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm7 , %%ymm9 \n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm10\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm11\n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm12\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm13\n\t" // [c41,c41,c51,c51]
	                "vfmadd231pd  %%ymm3 , %%ymm6 , %%ymm14\n\t" // [c60,c60,c70,c70]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c61,c61,c71,c71]
	                "\n\t"
	                "addq  $64*8 , %[a]\n\t"
	                "addq  $16*8 , %[b]\n\t"
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
	                "prefetcht0    32*8(%[a])\n\t"
	                "prefetcht0     8*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm5 \n\t" // [b01,b11,b01,b11]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a04,a14,a05,a15]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a06,a16,a07,a17]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm9 \n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm10\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm11\n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm12\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm13\n\t" // [c41,c41,c51,c51]
	                "vfmadd231pd  %%ymm3 , %%ymm4 , %%ymm14\n\t" // [c60,c60,c70,c70]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm15\n\t" // [c61,c61,c71,c71]
	                "\n\t"
	                "prefetcht0    48*8(%[a])\n\t"
	                "prefetcht0    24*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   4*8(%[b]), %%ymm6 \n\t" // [b20,b30,b20,b30]
	                "vbroadcastf128   6*8(%[b]), %%ymm7 \n\t" // [b21,b31,b21,b31]
	                "\n\t"
	                "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  20*8(%[a]), %%ymm1 \n\t" // [a22,a32,a23,a33]
	                "vmovapd  24*8(%[a]), %%ymm2 \n\t" // [a24,a34,a25,a35]
	                "vmovapd  28*8(%[a]), %%ymm3 \n\t" // [a26,a36,a27,a37]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm8 \n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm7 , %%ymm9 \n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm10\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm11\n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm12\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm13\n\t" // [c41,c41,c51,c51]
	                "vfmadd231pd  %%ymm3 , %%ymm6 , %%ymm14\n\t" // [c60,c60,c70,c70]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c61,c61,c71,c71]
	                "\n\t"
	                "addq  $32*8, %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          //}
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm5 \n\t" // [b01,b11,b01,b11]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a04,a14,a05,a15]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a06,a16,a07,a17]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm9 \n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm10\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm11\n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm12\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm13\n\t" // [c41,c41,c51,c51]
	                "vfmadd231pd  %%ymm3 , %%ymm4 , %%ymm14\n\t" // [c60,c60,c70,c70]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm15\n\t" // [c61,c61,c71,c71]
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[b]), %%xmm4 \n\t" // [b00,  0,  0,  0]
	                "vmovsd   1*8(%[b]), %%xmm5 \n\t" // [b01,  0,  0,  0]
	                "\n\t"
	                "vperm2f128 $0x00, %%ymm4 , %%ymm4 , %%ymm4 \n\t" // [b00,  0,b00,  0]
	                "vperm2f128 $0x00, %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b01,  0,b01,  0]
	                "\n\t"
	                "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t" // [a00,a01,a00,a01]
	                "vbroadcastf128  2*8(%[a]), %%ymm1 \n\t" // [a02,a03,a02,a03]
	                "vbroadcastf128  4*8(%[a]), %%ymm2 \n\t" // [a04,a05,a04,a05]
	                "vbroadcastf128  6*8(%[a]), %%ymm3 \n\t" // [a06,a07,a06,a07]
	                "\n\t"
	                "vshufpd   $0x06, %%ymm0 , %%ymm0 , %%ymm0 \n\t" // [a00,a01,a01,a00]
	                "vshufpd   $0x06, %%ymm1 , %%ymm1 , %%ymm1 \n\t" // [a02,a03,a03,a02]
	                "vshufpd   $0x06, %%ymm2 , %%ymm2 , %%ymm2 \n\t" // [a04,a05,a05,a04]
	                "vshufpd   $0x06, %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [a06,a07,a07,a06]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm8 \n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm9 \n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm10\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm11\n\t" // [c21,c21,c31,c31]
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm12\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm13\n\t" // [c41,c41,c51,c51]
	                "vfmadd231pd  %%ymm3 , %%ymm4 , %%ymm14\n\t" // [c60,c60,c70,c70]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm15\n\t" // [c61,c61,c71,c71]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $2*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }

	        // %%ymm8  [c00,c00,c10,c10]
	        // %%ymm10 [c20,c20,c30,c30]
	        // %%ymm12 [c40,c40,c50,c50]
	        // %%ymm14 [c60,c60,c70,c70]
	        //
	        // %%ymm9  [c01,c01,c11,c11]
	        // %%ymm11 [c21,c21,c31,c31]
	        // %%ymm13 [c41,c41,c51,c51]
	        // %%ymm15 [c61,c61,c71,c71]
	
	        __asm__ __volatile__ (
	            "\n\t"
	            "vperm2f128 $0x20, %%ymm10, %%ymm8 , %%ymm0 \n\t" // [c00,c00,c20,c20]
	            "vperm2f128 $0x31, %%ymm10, %%ymm8 , %%ymm1 \n\t" // [c10,c10,c30,c30]
	            "vperm2f128 $0x20, %%ymm14, %%ymm12, %%ymm2 \n\t" // [c40,c40,c60,c60]
	            "vperm2f128 $0x31, %%ymm14, %%ymm12, %%ymm3 \n\t" // [c50,c50,c70,c70]
	            "vperm2f128 $0x20, %%ymm11, %%ymm9 , %%ymm4 \n\t" // [c01,c01,c21,c21]
	            "vperm2f128 $0x31, %%ymm11, %%ymm9 , %%ymm5 \n\t" // [c11,c11,c31,c31]
	            "vperm2f128 $0x20, %%ymm15, %%ymm13, %%ymm6 \n\t" // [c41,c41,c61,c61]
	            "vperm2f128 $0x31, %%ymm15, %%ymm13, %%ymm7 \n\t" // [c51,c51,c71,c71]
	            "\n\t"
	            "vshufpd    $0x00, %%ymm1 , %%ymm0 , %%ymm8 \n\t" // [c00,c10,c20,c30]
	            "vshufpd    $0x0f, %%ymm1 , %%ymm0 , %%ymm10\n\t" // [c00,c10,c20,c30]
	            "vshufpd    $0x00, %%ymm3 , %%ymm2 , %%ymm12\n\t" // [c40,c50,c60,c70]
	            "vshufpd    $0x0f, %%ymm3 , %%ymm2 , %%ymm14\n\t" // [c40,c50,c60,c70]
	            "vshufpd    $0x00, %%ymm5 , %%ymm4 , %%ymm9 \n\t" // [c01,c11,c21,c31]
	            "vshufpd    $0x0f, %%ymm5 , %%ymm4 , %%ymm11\n\t" // [c01,c11,c21,c31]
	            "vshufpd    $0x00, %%ymm7 , %%ymm6 , %%ymm13\n\t" // [c41,c51,c61,c71]
	            "vshufpd    $0x0f, %%ymm7 , %%ymm6 , %%ymm15\n\t" // [c41,c51,c61,c71]
	            "\n\t"
	            "vaddpd            %%ymm8 , %%ymm10, %%ymm10\n\t" // [c00,c10,c20,c30]
	            "vaddpd            %%ymm12, %%ymm14, %%ymm14\n\t" // [c40,c50,c60,c70]
	            "vaddpd            %%ymm9 , %%ymm11, %%ymm11\n\t" // [c01,c11,c21,c31]
	            "vaddpd            %%ymm13, %%ymm15, %%ymm15\n\t" // [c41,c51,c61,c71]
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%ymm0, %%ymm10\n\t"
	            "vfmadd213pd 4*8(%[c0]          ), %%ymm0, %%ymm14\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc1]  ), %%ymm0, %%ymm11\n\t"
	            "vfmadd213pd 4*8(%[c0],%[ldc1]  ), %%ymm0, %%ymm15\n\t"
	            "\n\t"
	            "vmovupd  %%ymm10, 0*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm14, 4*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm11, 0*8(%[c0],%[ldc1]  )\n\t"
	            "vmovupd  %%ymm15, 4*8(%[c0],%[ldc1]  )\n\t"
	            "\n\t"
	            "addq  $8*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[alpha]"m"(alpha4),[ldc1]"r"(ldc1)
	        );

	        B = B - 2*K;

	      }
	    }
	    if( M & 4 ){
	    //kif( M >> 2 ){
	    //k  size_t m4 = ( M >> 2 ); // unrolling M
	    //k  while( m4-- ){

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
	            "\n\t"
	            "prefetcht0 0*8(%[c0]          )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc1]  )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[ldc1]"r"(ldc1)
	        );

	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0    96*8(%[a])\n\t"
	                "prefetcht0    16*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm5 \n\t" // [b01,b11,b01,b11]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm13\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm14\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm15\n\t" // [c21,c21,c31,c31]
	                "\n\t"
	                "prefetcht0   112*8(%[a])\n\t"
	                "prefetcht0    32*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   4*8(%[b]), %%ymm6 \n\t" // [b20,b30,b20,b30]
	                "vbroadcastf128   6*8(%[b]), %%ymm7 \n\t" // [b21,b31,b21,b31]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a22,a32,a23,a33]
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm12\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm13\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm6 , %%ymm14\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c21,c21,c31,c31]
	                "\n\t"
	                "prefetcht0   128*8(%[a])\n\t"
	                "prefetcht0    48*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   8*8(%[b]), %%ymm4 \n\t" // [b40,b50,b40,b50]
	                "vbroadcastf128  10*8(%[b]), %%ymm5 \n\t" // [b41,b51,b41,b51]
	                "\n\t"
	                "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a40,a50,a41,a51]
	                "vmovapd  20*8(%[a]), %%ymm1 \n\t" // [a42,a52,a43,a53]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm13\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm14\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm15\n\t" // [c21,c21,c31,c31]
	                "\n\t"
	                "prefetcht0   144*8(%[a])\n\t"
	                "prefetcht0    64*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128  12*8(%[b]), %%ymm6 \n\t" // [b60,b70,b60,b70]
	                "vbroadcastf128  14*8(%[b]), %%ymm7 \n\t" // [b61,b71,b61,b71]
	                "\n\t"
	                "vmovapd  24*8(%[a]), %%ymm2 \n\t" // [a60,a70,a61,a71]
	                "vmovapd  28*8(%[a]), %%ymm3 \n\t" // [a62,a72,a63,a73]
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm12\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm13\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm6 , %%ymm14\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c21,c21,c31,c31]
	                "\n\t"
	                "addq  $32*8 , %[a]\n\t"
	                "addq  $16*8 , %[b]\n\t"
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
	                "prefetcht0    48*8(%[a])\n\t"
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm5 \n\t" // [b01,b11,b01,b11]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm13\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm14\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm15\n\t" // [c21,c21,c31,c31]
	                "\n\t"
	                "prefetcht0    64*8(%[a])\n\t"
	                "\n\t"
	                "vbroadcastf128   4*8(%[b]), %%ymm6 \n\t" // [b20,b30,b20,b30]
	                "vbroadcastf128   6*8(%[b]), %%ymm7 \n\t" // [b21,b31,b21,b31]
	                "\n\t"
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a22,a32,a23,a33]
	                "\n\t"
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm12\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm13\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm6 , %%ymm14\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c21,c21,c31,c31]
	                "\n\t"
	                "addq  $16*8 , %[a]\n\t"
	                "addq  $8*8  , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          //}
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm5 \n\t" // [b01,b11,b01,b11]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm13\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm14\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm15\n\t" // [c21,c21,c31,c31]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            // b00 = *(B+0+0*2+0*2*3);
	            // b01 = *(B+0+1*2+0*2*3);
	            // b02 = *(B+0+2*2+0*2*3);
	            //                            
	            // a00 = *(A+0+0*2+0*2*8);
	            // a01 = *(A+0+1*2+0*2*8);
	            // c00 += a00 * b00; ; c10 += a01 * b00; 
	            // c01 += a00 * b01; ; c11 += a01 * b01; 
	            // c02 += a00 * b02; ; c12 += a01 * b02; 
	            //
	            // a02 = *(A+0+2*2+0*2*8); 
	            // a03 = *(A+0+3*2+0*2*8); 
	            // c20 += a02 * b00; ; c30 += a03 * b00; 
	            // c21 += a02 * b01; ; c31 += a03 * b01; 
	            // c22 += a02 * b02; ; c32 += a03 * b02; 
	            //

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[b]), %%xmm4 \n\t" // [b00,  0,  0,  0]
	                "vmovsd   1*8(%[b]), %%xmm5 \n\t" // [b01,  0,  0,  0]
	                "\n\t"
	                "vperm2f128 $0x00, %%ymm4 , %%ymm4 , %%ymm4 \n\t" // [b00,  0,b00,  0]
	                "vperm2f128 $0x00, %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b01,  0,b01,  0]
	                "\n\t"
	                "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t" // [a00,a01,a00,a01]
	                "vbroadcastf128  2*8(%[a]), %%ymm1 \n\t" // [a02,a03,a02,a03]
	                "\n\t"
	                "vshufpd   $0x06, %%ymm0 , %%ymm0 , %%ymm0 \n\t" // [a00,a01,a01,a00]
	                "vshufpd   $0x06, %%ymm1 , %%ymm1 , %%ymm1 \n\t" // [a02,a03,a03,a02]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm13\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm14\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm15\n\t" // [c21,c21,c31,c31]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $2*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }

	        // %%ymm10 [c00,c00,c10,c10]
	        // %%ymm11 [c01,c01,c11,c11]
	        // %%ymm12 [c02,c02,c12,c12]
	        // %%ymm13 [c20,c20,c30,c30]
	        // %%ymm14 [c21,c21,c31,c31]
	        // %%ymm15 [c22,c22,c32,c32]
	        __asm__ __volatile__ (
	            "\n\t"
	            "vperm2f128 $0x20, %%ymm14, %%ymm12, %%ymm0 \n\t" // [c00,c00,c20,c20]
	            "vperm2f128 $0x31, %%ymm14, %%ymm12, %%ymm1 \n\t" // [c10,c10,c30,c30]
	            "vperm2f128 $0x20, %%ymm15, %%ymm13, %%ymm4 \n\t" // [c01,c01,c21,c21]
	            "vperm2f128 $0x31, %%ymm15, %%ymm13, %%ymm5 \n\t" // [c11,c11,c31,c31]
	            "\n\t"
	            "vshufpd    $0x00, %%ymm1 , %%ymm0 , %%ymm12\n\t" // [c00,c10,c20,c30]
	            "vshufpd    $0x0f, %%ymm1 , %%ymm0 , %%ymm14\n\t" // [c00,c10,c20,c30]
	            "vshufpd    $0x00, %%ymm5 , %%ymm4 , %%ymm13\n\t" // [c01,c11,c21,c31]
	            "vshufpd    $0x0f, %%ymm5 , %%ymm4 , %%ymm15\n\t" // [c01,c11,c21,c31]
	            "\n\t"
	            "vaddpd            %%ymm12, %%ymm14, %%ymm14\n\t" // [c00,c10,c20,c30]
	            "vaddpd            %%ymm13, %%ymm15, %%ymm15\n\t" // [c01,c11,c21,c31]
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%ymm0, %%ymm14\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc1]  ), %%ymm0, %%ymm15\n\t"
	            "\n\t"
	            "vmovupd  %%ymm14, 0*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm15, 0*8(%[c0],%[ldc1]  )\n\t"
	            "\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[alpha]"m"(alpha4),[ldc1]"r"(ldc1)
	        );

	        B = B - 2*K;

	    //  }
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
	            "\n\t"
	            "prefetcht0 0*8(%[c0]          )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc1]  )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[ldc1]"r"(ldc1)
	        );


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0    48*8(%[a])\n\t"
	                "prefetcht0    16*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm5 \n\t" // [b01,b11,b01,b11]
	                "vbroadcastf128   4*8(%[b]), %%ymm6 \n\t" // [b20,b30,b20,b30]
	                "vbroadcastf128   6*8(%[b]), %%ymm7 \n\t" // [b21,b31,b21,b31]
	                "vbroadcastf128   8*8(%[b]), %%ymm8 \n\t" // [b40,b50,b40,b50]
	                "vbroadcastf128  10*8(%[b]), %%ymm9 \n\t" // [b41,b51,b41,b51]
	                "vbroadcastf128  12*8(%[b]), %%ymm10\n\t" // [b60,b70,b60,b70]
	                "vbroadcastf128  14*8(%[b]), %%ymm11\n\t" // [b61,b71,b61,b71]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a20,a30,a21,a31]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a40,a50,a41,a51]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a60,a70,a61,a71]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm15\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm15\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm2 , %%ymm8 , %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm2 , %%ymm9 , %%ymm15\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm3 , %%ymm10, %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm3 , %%ymm11, %%ymm15\n\t" // [c01,c01,c11,c11]
	                "\n\t"
	                "addq  $16*8 , %[a]\n\t"
	                "addq  $16*8 , %[b]\n\t"
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
	                "vbroadcastf128   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm5 \n\t" // [b01,b11,b01,b11]
	                "vbroadcastf128   4*8(%[b]), %%ymm6 \n\t" // [b20,b30,b20,b30]
	                "vbroadcastf128   6*8(%[b]), %%ymm7 \n\t" // [b21,b31,b21,b31]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a20,a30,a21,a31]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm15\n\t" // [c01,c01,c11,c11]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm15\n\t" // [c01,c01,c11,c11]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          //}
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm5 \n\t" // [b01,b11,b01,b11]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm15\n\t" // [c01,c01,c11,c11]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){

	            // b00 = *(B+0+0*2+0*2*3);
	            // b01 = *(B+0+1*2+0*2*3);
	            //                            
	            // a00 = *(A+0+0*2+0*2*8);
	            // a01 = *(A+0+1*2+0*2*8);
	            // c00 += a00 * b00; ; c10 += a01 * b00; 
	            // c01 += a00 * b01; ; c11 += a01 * b01; 
	            //
	            // a02 = *(A+0+2*2+0*2*8); 
	            // a03 = *(A+0+3*2+0*2*8); 
	            // c20 += a02 * b00; ; c30 += a03 * b00; 
	            // c21 += a02 * b01; ; c31 += a03 * b01; 
	            //
	
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[b]), %%xmm1 \n\t" // [b00,  0,  0,  0]
	                "vmovsd   1*8(%[b]), %%xmm2 \n\t" // [b01,  0,  0,  0]
	                "\n\t"
	                "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t" // [a00,a01,a00,a01]
	                "\n\t"
	                "vperm2f128 $0x00, %%ymm1 , %%ymm1 , %%ymm1 \n\t" // [b00,  0,b00,  0]
	                "vperm2f128 $0x00, %%ymm2 , %%ymm2 , %%ymm2 \n\t" // [b01,  0,b01,  0]
	                "\n\t"
	                "vshufpd   $0x06, %%ymm0 , %%ymm0 , %%ymm0 \n\t" // [a00,a01,a01,a00]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm0 , %%ymm2 , %%ymm15\n\t" // [c01,c01,c11,c11]
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $2*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }

	        // %%ymm13 [c00,c00,c10,c10]
	        // %%ymm14 [c01,c01,c11,c11]
	
	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm6\n\t"
	            "\n\t"
	            "vshufpd    $0x05, %%ymm14, %%ymm14, %%ymm10\n\t" // [c00,c00,c10,c10]
	            "vshufpd    $0x05, %%ymm15, %%ymm15, %%ymm11\n\t" // [c01,c01,c11,c11]
	            "vaddpd            %%ymm10, %%ymm14, %%ymm14\n\t" // [c00,c00,c10,c10]
	            "vaddpd            %%ymm11, %%ymm15, %%ymm15\n\t" // [c01,c01,c11,c11]
	            "\n\t"
	            "vperm2f128 $0x01, %%ymm14, %%ymm14, %%ymm10\n\t" // [c10,c10,c00,c00]
	            "vperm2f128 $0x01, %%ymm15, %%ymm15, %%ymm11\n\t" // [c11,c11,c01,c01]
	            "\n\t"
	            "vshufpd    $0x00, %%ymm10, %%ymm14, %%ymm14\n\t" // [c00,c10,c10,c00]
	            "vshufpd    $0x00, %%ymm11, %%ymm15, %%ymm15\n\t" // [c01,c11,c11,c01]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%xmm6, %%xmm14\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc1]  ), %%xmm6, %%xmm15\n\t"
	            "\n\t"
	            "vmovupd  %%xmm14, 0*8(%[c0]          )\n\t"
	            "vmovupd  %%xmm15, 0*8(%[c0],%[ldc1]  )\n\t"
	            "\n\t"
	            "addq  $2*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[alpha]"m"(alpha4),[ldc1]"r"(ldc1)
	        );

	        B = B - 2*K;

	    }
	    if( M & 1 ){

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
	            "\n\t"
	            "prefetcht0 0*8(%[c0]          )\n\t"
	            "prefetcht0 0*8(%[c0],%[ldc1]  )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[ldc1]"r"(ldc1)
	        );

	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0    16*8(%[b])\n\t"
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10,  0,  0]
	                "vmovupd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11,  0,  0]
	                "vmovupd   4*8(%[b]), %%xmm6 \n\t" // [b20,b30,  0,  0]
	                "vmovupd   6*8(%[b]), %%xmm7 \n\t" // [b21,b31,  0,  0]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,  0,  0]
	                "vmovapd   2*8(%[a]), %%xmm1 \n\t" // [a20,a30,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%xmm0 , %%xmm4 , %%xmm14\n\t" // [c00,c00,  0,  0]
	                "vfmadd231pd  %%xmm0 , %%xmm5 , %%xmm15\n\t" // [c01,c01,  0,  0]
	                "vfmadd231pd  %%xmm1 , %%xmm6 , %%xmm14\n\t" // [c00,c00,  0,  0]
	                "vfmadd231pd  %%xmm1 , %%xmm7 , %%xmm15\n\t" // [c01,c01,  0,  0]
	                "\n\t"
	                "vmovupd   8*8(%[b]), %%xmm4 \n\t" // [b40,b50,  0,  0]
	                "vmovupd  10*8(%[b]), %%xmm5 \n\t" // [b41,b51,  0,  0]
	                "vmovupd  12*8(%[b]), %%xmm6 \n\t" // [b60,b70,  0,  0]
	                "vmovupd  14*8(%[b]), %%xmm7 \n\t" // [b61,b71,  0,  0]
	                "\n\t"
	                "vmovapd   4*8(%[a]), %%xmm2 \n\t" // [a40,a50,  0,  0]
	                "vmovapd   6*8(%[a]), %%xmm3 \n\t" // [a60,a70,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%xmm2 , %%xmm4 , %%xmm14\n\t" // [c00,c00,  0,  0]
	                "vfmadd231pd  %%xmm2 , %%xmm5 , %%xmm15\n\t" // [c01,c01,  0,  0]
	                "vfmadd231pd  %%xmm3 , %%xmm6 , %%xmm14\n\t" // [c00,c00,  0,  0]
	                "vfmadd231pd  %%xmm3 , %%xmm7 , %%xmm15\n\t" // [c01,c01,  0,  0]
	                "\n\t"
	                "addq  $8*8  , %[a]\n\t"
	                "addq  $16*8 , %[b]\n\t"
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
	                "vmovupd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10,  0,  0]
	                "vmovupd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11,  0,  0]
	                "vmovupd   4*8(%[b]), %%xmm6 \n\t" // [b20,b30,  0,  0]
	                "vmovupd   6*8(%[b]), %%xmm7 \n\t" // [b21,b31,  0,  0]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,  0,  0]
	                "vmovapd   2*8(%[a]), %%xmm1 \n\t" // [a20,a30,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%xmm0 , %%xmm4 , %%xmm14\n\t" // [c00,c00,  0,  0]
	                "vfmadd231pd  %%xmm0 , %%xmm5 , %%xmm15\n\t" // [c01,c01,  0,  0]
	                "vfmadd231pd  %%xmm1 , %%xmm6 , %%xmm14\n\t" // [c00,c00,  0,  0]
	                "vfmadd231pd  %%xmm1 , %%xmm7 , %%xmm15\n\t" // [c01,c01,  0,  0]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          //}
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10,  0,  0]
	                "vmovupd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11,  0,  0]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%xmm0 , %%xmm4 , %%xmm14\n\t" // [c00,c00,  0,  0]
	                "vfmadd231pd  %%xmm0 , %%xmm5 , %%xmm15\n\t" // [c01,c01,  0,  0]
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[b]), %%xmm4 \n\t" // [b00,  0,  0,  0]
	                "vmovsd   1*8(%[b]), %%xmm5 \n\t" // [b01,  0,  0,  0]
	                "\n\t"
	                "vmovsd   0*8(%[a]), %%xmm0 \n\t" // [a00,  0,  0,  0]
	                "\n\t"
	                "vfmadd231sd  %%xmm0 , %%xmm4 , %%xmm14\n\t" // [c00,c00,  0,  0]
	                "vfmadd231sd  %%xmm0 , %%xmm5 , %%xmm15\n\t" // [c01,c01,  0,  0]
	                "\n\t"
	                "addq  $1*8 , %[a]\n\t"
	                "addq  $2*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }

	        // %%ymm13 [c00,c00,  0,  0]
	        // %%ymm14 [c01,c01,  0,  0]
	        // %%ymm15 [c02,c02,  0,  0]
	
	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm6\n\t"
	            "\n\t"
	            "vshufpd    $0x05, %%ymm14, %%ymm14, %%ymm10\n\t" // [c00,c00,  0,  0]
	            "vshufpd    $0x05, %%ymm15, %%ymm15, %%ymm11\n\t" // [c01,c01,  0,  0]
	            "vaddpd            %%ymm10, %%ymm14, %%ymm14\n\t" // [c00,---,  0,  0]
	            "vaddpd            %%ymm11, %%ymm15, %%ymm15\n\t" // [c01,---,  0,  0]
	            "\n\t"
	            "vfmadd213sd 0*8(%[c0]          ), %%xmm6, %%xmm14\n\t"
	            "vfmadd213sd 0*8(%[c0],%[ldc1]  ), %%xmm6, %%xmm15\n\t"
	            "\n\t"
	            "vmovsd  %%xmm14, 0*8(%[c0]          )\n\t"
	            "vmovsd  %%xmm15, 0*8(%[c0],%[ldc1]  )\n\t"
	            "\n\t"
	            "addq  $1*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[alpha]"m"(alpha4),[ldc1]"r"(ldc1)
	        );

	        B = B - 2*K;

	    }

	    A = A - M*K;
	    B = B + 2*K;
	    c0 = c0- M + 2*ldc;

	}
	if( N3R==1 ){

	    if( M >> 3 ){
	      size_t m8 = ( M >> 3 ); // unrolling M
	      while( m8-- ){

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

	        __asm__ __volatile__ (
	            "\n\t"
	            "prefetcht0 0*8(%[c0]          )\n\t"
	            "prefetcht0 4*8(%[c0]          )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[ldc1]"r"(ldc1)
	        );


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){
	            //
	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0   192*8(%[a])\n\t"
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm5 \n\t" // [b20,b30,b20,b30]
	                "vbroadcastf128   4*8(%[b]), %%ymm6 \n\t" // [b40,b50,b40,b50]
	                "vbroadcastf128   6*8(%[b]), %%ymm7 \n\t" // [b60,b70,b60,b70]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a04,a14,a05,a15]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a06,a16,a07,a17]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm13\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm14\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm3 , %%ymm4 , %%ymm15\n\t" // [c60,c60,c70,c70]
	                "\n\t"
	                "prefetcht0   208*8(%[a])\n\t"
	                "\n\t"
	                "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  20*8(%[a]), %%ymm1 \n\t" // [a22,a32,a23,a33]
	                "vmovapd  24*8(%[a]), %%ymm2 \n\t" // [a24,a34,a25,a35]
	                "vmovapd  28*8(%[a]), %%ymm3 \n\t" // [a26,a36,a27,a37]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm12\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm13\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm14\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm15\n\t" // [c60,c60,c70,c70]
	                "\n\t"
	                "prefetcht0   224*8(%[a])\n\t"
	                "\n\t"
	                "vmovapd  32*8(%[a]), %%ymm0 \n\t" // [a40,a50,a41,a51]
	                "vmovapd  36*8(%[a]), %%ymm1 \n\t" // [a42,a52,a43,a53]
	                "vmovapd  40*8(%[a]), %%ymm2 \n\t" // [a44,a54,a45,a55]
	                "vmovapd  44*8(%[a]), %%ymm3 \n\t" // [a46,a56,a47,a57]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm6 , %%ymm12\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm6 , %%ymm13\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm2 , %%ymm6 , %%ymm14\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm3 , %%ymm6 , %%ymm15\n\t" // [c60,c60,c70,c70]
	                "\n\t"
	                "prefetcht0   240*8(%[a])\n\t"
	                "\n\t"
	                "vmovapd  48*8(%[a]), %%ymm0 \n\t" // [a60,a70,a61,a71]
	                "vmovapd  52*8(%[a]), %%ymm1 \n\t" // [a62,a72,a63,a73]
	                "vmovapd  56*8(%[a]), %%ymm2 \n\t" // [a64,a74,a65,a75]
	                "vmovapd  60*8(%[a]), %%ymm3 \n\t" // [a66,a76,a67,a77]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm7 , %%ymm12\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm7 , %%ymm13\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm2 , %%ymm7 , %%ymm14\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm3 , %%ymm7 , %%ymm15\n\t" // [c60,c60,c70,c70]
	                "\n\t"
	                "addq  $64*8 , %[a]\n\t"
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
	                "prefetcht0    32*8(%[a])\n\t"
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm5 \n\t" // [b20,b30,b20,b30]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a04,a14,a05,a15]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a06,a16,a07,a17]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm13\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm14\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm3 , %%ymm4 , %%ymm15\n\t" // [c60,c60,c70,c70]
	                "\n\t"
	                "prefetcht0    48*8(%[a])\n\t"
	                "\n\t"
	                "vmovapd  16*8(%[a]), %%ymm0 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  20*8(%[a]), %%ymm1 \n\t" // [a22,a32,a23,a33]
	                "vmovapd  24*8(%[a]), %%ymm2 \n\t" // [a24,a34,a25,a35]
	                "vmovapd  28*8(%[a]), %%ymm3 \n\t" // [a26,a36,a27,a37]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm5 , %%ymm12\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm5 , %%ymm13\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm2 , %%ymm5 , %%ymm14\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm3 , %%ymm5 , %%ymm15\n\t" // [c60,c60,c70,c70]
	                "\n\t"
	                "addq  $32*8 , %[a]\n\t"
	                "addq  $4*8  , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          //}
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b00,b10]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a04,a14,a05,a15]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a06,a16,a07,a17]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm13\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm14\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm3 , %%ymm4 , %%ymm15\n\t" // [c60,c60,c70,c70]
	                "\n\t"
	                "addq  $16*8 , %[a]\n\t"
	                "addq  $2*8  , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[b]), %%xmm4 \n\t" // [b00,  0,  0,  0]
	                "\n\t"
	                "vperm2f128 $0x00, %%ymm4 , %%ymm4 , %%ymm4 \n\t" // [b00,  0,b00,  0]
	                "\n\t"
	                "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t" // [a00,a01,a00,a01]
	                "vbroadcastf128  2*8(%[a]), %%ymm1 \n\t" // [a02,a03,a02,a03]
	                "vbroadcastf128  4*8(%[a]), %%ymm2 \n\t" // [a04,a05,a04,a05]
	                "vbroadcastf128  6*8(%[a]), %%ymm3 \n\t" // [a06,a07,a06,a07]
	                "\n\t"
	                "vshufpd   $0x06, %%ymm0 , %%ymm0 , %%ymm0 \n\t" // [a00,a01,a01,a00]
	                "vshufpd   $0x06, %%ymm1 , %%ymm1 , %%ymm1 \n\t" // [a02,a03,a03,a02]
	                "vshufpd   $0x06, %%ymm2 , %%ymm2 , %%ymm2 \n\t" // [a04,a05,a05,a04]
	                "vshufpd   $0x06, %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [a06,a07,a07,a06]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm12\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm13\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm2 , %%ymm4 , %%ymm14\n\t" // [c40,c40,c50,c50]
	                "vfmadd231pd  %%ymm3 , %%ymm4 , %%ymm15\n\t" // [c60,c60,c70,c70]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $1*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }

	        // %%ymm12 [c00,c00,c10,c10]
	        // %%ymm13 [c20,c20,c30,c30]
	        // %%ymm14 [c40,c40,c50,c50]
	        // %%ymm15 [c60,c60,c70,c70]
	        //
	        __asm__ __volatile__ (
	            "\n\t"
	            "vperm2f128 $0x20, %%ymm13, %%ymm12, %%ymm0 \n\t" // [c00,c00,c20,c20]
	            "vperm2f128 $0x31, %%ymm13, %%ymm12, %%ymm1 \n\t" // [c10,c10,c30,c30]
	            "vperm2f128 $0x20, %%ymm15, %%ymm14, %%ymm2 \n\t" // [c40,c40,c60,c60]
	            "vperm2f128 $0x31, %%ymm15, %%ymm14, %%ymm3 \n\t" // [c50,c50,c70,c70]
	            "\n\t"
	            "vshufpd    $0x00, %%ymm1 , %%ymm0 , %%ymm12\n\t" // [c00,c10,c20,c30]
	            "vshufpd    $0x0f, %%ymm1 , %%ymm0 , %%ymm13\n\t" // [c00,c10,c20,c30]
	            "vshufpd    $0x00, %%ymm3 , %%ymm2 , %%ymm14\n\t" // [c40,c50,c60,c70]
	            "vshufpd    $0x0f, %%ymm3 , %%ymm2 , %%ymm15\n\t" // [c40,c50,c60,c70]
	            "\n\t"
	            "vaddpd            %%ymm12, %%ymm13, %%ymm13\n\t" // [c00,c10,c20,c30]
	            "vaddpd            %%ymm14, %%ymm15, %%ymm15\n\t" // [c40,c50,c60,c70]
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%ymm0, %%ymm13\n\t"
	            "vfmadd213pd 4*8(%[c0]          ), %%ymm0, %%ymm15\n\t"
	            "\n\t"
	            "vmovupd  %%ymm13, 0*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm15, 4*8(%[c0]          )\n\t"
	            "\n\t"
	            "addq  $8*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[alpha]"m"(alpha4),[ldc1]"r"(ldc1)
	        );

	        B = B - 1*K;

	      }
	    }
	    if( M & 4 ){
	    //kif( M >> 2 ){
	    //k  size_t m4 = ( M >> 2 ); // unrolling M
	    //k  while( m4-- ){

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
	            "\n\t"
	            "prefetcht0 0*8(%[c0]          )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[ldc1]"r"(ldc1)
	        );


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0    96*8(%[a])\n\t"
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm8 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm9 \n\t" // [b20,b30,b20,b30]
	                "vbroadcastf128   4*8(%[b]), %%ymm10\n\t" // [b40,b50,b40,b50]
	                "vbroadcastf128   6*8(%[b]), %%ymm11\n\t" // [b60,b70,b60,b70]
	                "\n\t"
	                "prefetcht0   112*8(%[a])\n\t"
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a22,a32,a23,a33]
	                "vmovapd  16*8(%[a]), %%ymm4 \n\t" // [a40,a50,a41,a51]
	                "vmovapd  20*8(%[a]), %%ymm5 \n\t" // [a42,a52,a43,a53]
	                "vmovapd  24*8(%[a]), %%ymm6 \n\t" // [a60,a70,a61,a71]
	                "vmovapd  28*8(%[a]), %%ymm7 \n\t" // [a62,a72,a63,a73]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm8 , %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm8 , %%ymm15\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm2 , %%ymm9 , %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm3 , %%ymm9 , %%ymm15\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm4 , %%ymm10, %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm5 , %%ymm10, %%ymm15\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm6 , %%ymm11, %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm7 , %%ymm11, %%ymm15\n\t" // [c20,c20,c30,c30]
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
	                "prefetcht0    16*8(%[a])\n\t"
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm8 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm9 \n\t" // [b20,b30,b20,b30]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a20,a30,a21,a31]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a22,a32,a23,a33]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm8 , %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm8 , %%ymm15\n\t" // [c20,c20,c30,c30]
	                "vfmadd231pd  %%ymm2 , %%ymm9 , %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm3 , %%ymm9 , %%ymm15\n\t" // [c20,c20,c30,c30]
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
	                "vbroadcastf128   0*8(%[b]), %%ymm8 \n\t" // [b00,b10,b00,b10]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a02,a12,a03,a13]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm8 , %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm8 , %%ymm15\n\t" // [c20,c20,c30,c30]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $2*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){

	            // b00 = *(B+0+0*2+0*2*3);
	            // b01 = *(B+0+1*2+0*2*3);
	            // b02 = *(B+0+2*2+0*2*3);
	            //                            
	            // a00 = *(A+0+0*2+0*2*8);
	            // a01 = *(A+0+1*2+0*2*8);
	            // c00 += a00 * b00; ; c10 += a01 * b00; 
	            // c01 += a00 * b01; ; c11 += a01 * b01; 
	            // c02 += a00 * b02; ; c12 += a01 * b02; 
	            //
	            // a02 = *(A+0+2*2+0*2*8); 
	            // a03 = *(A+0+3*2+0*2*8); 
	            // c20 += a02 * b00; ; c30 += a03 * b00; 
	            // c21 += a02 * b01; ; c31 += a03 * b01; 
	            // c22 += a02 * b02; ; c32 += a03 * b02; 
	            //

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[b]), %%xmm4 \n\t" // [b00,  0,  0,  0]
	                "\n\t"
	                "vperm2f128 $0x00, %%ymm4 , %%ymm4 , %%ymm4 \n\t" // [b00,  0,b00,  0]
	                "\n\t"
	                "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t" // [a00,a01,a00,a01]
	                "vbroadcastf128  2*8(%[a]), %%ymm1 \n\t" // [a02,a03,a02,a03]
	                "\n\t"
	                "vshufpd   $0x06, %%ymm0 , %%ymm0 , %%ymm0 \n\t" // [a00,a01,a01,a00]
	                "vshufpd   $0x06, %%ymm1 , %%ymm1 , %%ymm1 \n\t" // [a02,a03,a03,a02]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm4 , %%ymm14\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm4 , %%ymm15\n\t" // [c20,c20,c30,c30]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $2*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }

	        // %%ymm14 [c00,c00,c10,c10]
	        // %%ymm15 [c20,c20,c30,c30]
	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm6\n\t"
	            "\n\t"
	            "vperm2f128 $0x20, %%ymm15, %%ymm14, %%ymm0 \n\t" // [c00,c00,c20,c20]
	            "vperm2f128 $0x31, %%ymm15, %%ymm14, %%ymm1 \n\t" // [c10,c10,c30,c30]
	            "vshufpd    $0x00, %%ymm1 , %%ymm0 , %%ymm14\n\t" // [c00,c10,c20,c30]
	            "vshufpd    $0x0f, %%ymm1 , %%ymm0 , %%ymm15\n\t" // [c00,c10,c20,c30]
	            "vaddpd            %%ymm14, %%ymm15, %%ymm15\n\t" // [c00,c10,c20,c30]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%ymm6, %%ymm15\n\t"
	            "\n\t"
	            "vmovupd  %%ymm15, 0*8(%[c0]          )\n\t"
	            "\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[alpha]"m"(alpha4),[ldc1]"r"(ldc1)
	        );

	        B = B - 1*K;

	    //  }
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
	            "\n\t"
	            "prefetcht0 0*8(%[c0]          )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[ldc1]"r"(ldc1)
	        );


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0    48*8(%[a])\n\t"
	                "\n\t"
	                "vbroadcastf128   0*8(%[b]), %%ymm8 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm9 \n\t" // [b20,b30,b20,b30]
	                "vbroadcastf128   4*8(%[b]), %%ymm10\n\t" // [b40,b50,b40,b50]
	                "vbroadcastf128   6*8(%[b]), %%ymm11\n\t" // [b60,b70,b60,b70]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a20,a30,a21,a31]
	                "vmovapd   8*8(%[a]), %%ymm2 \n\t" // [a40,a50,a41,a51]
	                "vmovapd  12*8(%[a]), %%ymm3 \n\t" // [a60,a70,a61,a71]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm8 , %%ymm15\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm9 , %%ymm15\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm2 , %%ymm10, %%ymm15\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm3 , %%ymm11, %%ymm15\n\t" // [c00,c00,c10,c10]
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
	                "vbroadcastf128   0*8(%[b]), %%ymm8 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm9 \n\t" // [b20,b30,b20,b30]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "vmovapd   4*8(%[a]), %%ymm1 \n\t" // [a20,a30,a21,a31]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm8 , %%ymm15\n\t" // [c00,c00,c10,c10]
	                "vfmadd231pd  %%ymm1 , %%ymm9 , %%ymm15\n\t" // [c00,c00,c10,c10]
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
	                "vbroadcastf128   0*8(%[b]), %%ymm8 \n\t" // [b00,b10,b00,b10]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a01,a11]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm8 , %%ymm15\n\t" // [c00,c00,c10,c10]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $2*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){

	            // b00 = *(B+0+0*2+0*2*3);
	            // b01 = *(B+0+1*2+0*2*3);
	            //                            
	            // a00 = *(A+0+0*2+0*2*8);
	            // a01 = *(A+0+1*2+0*2*8);
	            // c00 += a00 * b00; ; c10 += a01 * b00; 
	            // c01 += a00 * b01; ; c11 += a01 * b01; 
	            //
	            // a02 = *(A+0+2*2+0*2*8); 
	            // a03 = *(A+0+3*2+0*2*8); 
	            // c20 += a02 * b00; ; c30 += a03 * b00; 
	            // c21 += a02 * b01; ; c31 += a03 * b01; 
	            //
	
	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[b]), %%xmm1 \n\t" // [b00,  0,  0,  0]
	                "\n\t"
	                "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t" // [a00,a01,a00,a01]
	                "\n\t"
	                "vperm2f128 $0x00, %%ymm1 , %%ymm1 , %%ymm1 \n\t" // [b00,  0,b00,  0]
	                "\n\t"
	                "vshufpd   $0x06, %%ymm0 , %%ymm0 , %%ymm0 \n\t" // [a00,a01,a01,a00]
	                "\n\t"
	                "vfmadd231pd  %%ymm0 , %%ymm1 , %%ymm15\n\t" // [c00,c00,c10,c10]
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $1*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        // %%ymm14 [c00,c00,c10,c10]
	        // %%ymm15 [c01,c01,c11,c11]
	
	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm6\n\t"
	            "\n\t"
	            "vshufpd    $0x05, %%ymm15, %%ymm15, %%ymm10\n\t" // [c00,c00,c10,c10]
	            "vaddpd            %%ymm10, %%ymm15, %%ymm15\n\t" // [c00,---,c10,---]
	            "\n\t"
	            "vperm2f128 $0x01, %%ymm15, %%ymm15, %%ymm10\n\t" // [c10,---,c00,---]
	            "\n\t"
	            "vshufpd    $0x00, %%ymm10, %%ymm15, %%ymm15\n\t" // [c00,c10,---,---]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%xmm6, %%xmm15\n\t"
	            "\n\t"
	            "vmovupd  %%xmm15, 0*8(%[c0]          )\n\t"
	            "\n\t"
	            "addq  $2*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[alpha]"m"(alpha4),[ldc1]"r"(ldc1)
	        );

	        B = B - 1*K;

	    }
	    if( M & 1 ){

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
	            "\n\t"
	            "prefetcht0 0*8(%[c0]          )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[ldc1]"r"(ldc1)
	        );

	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 ); // Unrolling K
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovupd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10,  0,  0]
	                "vmovupd   2*8(%[b]), %%xmm5 \n\t" // [b20,b30,  0,  0]
	                "vmovupd   4*8(%[b]), %%xmm6 \n\t" // [b40,b50,  0,  0]
	                "vmovupd   6*8(%[b]), %%xmm7 \n\t" // [b60,b70,  0,  0]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,  0,  0]
	                "vmovapd   2*8(%[a]), %%xmm1 \n\t" // [a20,a30,  0,  0]
	                "vmovapd   4*8(%[a]), %%xmm2 \n\t" // [a40,a50,  0,  0]
	                "vmovapd   6*8(%[a]), %%xmm3 \n\t" // [a60,a70,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%xmm0 , %%xmm4 , %%xmm15\n\t" // [c00,c00,  0,  0]
	                "vfmadd231pd  %%xmm1 , %%xmm5 , %%xmm15\n\t" // [c00,c00,  0,  0]
	                "vfmadd231pd  %%xmm2 , %%xmm6 , %%xmm15\n\t" // [c00,c00,  0,  0]
	                "vfmadd231pd  %%xmm3 , %%xmm7 , %%xmm15\n\t" // [c00,c00,  0,  0]
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
	                "vmovupd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10,  0,  0]
	                "vmovupd   2*8(%[b]), %%xmm5 \n\t" // [b20,b30,  0,  0]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,  0,  0]
	                "vmovapd   2*8(%[a]), %%xmm1 \n\t" // [a20,a30,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%xmm0 , %%xmm4 , %%xmm15\n\t" // [c00,c00,  0,  0]
	                "vfmadd231pd  %%xmm1 , %%xmm5 , %%xmm15\n\t" // [c00,c00,  0,  0]
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
	                "vmovupd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10,  0,  0]
	                "\n\t"
	                "vmovapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10,  0,  0]
	                "\n\t"
	                "vfmadd231pd  %%xmm0 , %%xmm4 , %%xmm15\n\t" // [c00,c00,  0,  0]
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $2*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovsd   0*8(%[b]), %%xmm4 \n\t" // [b00,  0,  0,  0]
	                "\n\t"
	                "vmovsd   0*8(%[a]), %%xmm0 \n\t" // [a00,  0,  0,  0]
	                "\n\t"
	                "vfmadd231sd  %%xmm0 , %%xmm4 , %%xmm15\n\t" // [c00,c00,  0,  0]
	                "\n\t"
	                "addq  $1*8 , %[a]\n\t"
	                "addq  $1*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }

	        // %%ymm15 [c00,c00,  0,  0]
	
	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm6\n\t"
	            "\n\t"
	            "vshufpd    $0x05, %%ymm15, %%ymm15, %%ymm10\n\t" // [c00,c00,  0,  0]
	            "vaddpd            %%ymm10, %%ymm15, %%ymm15\n\t" // [c00,---,  0,  0]
	            "\n\t"
	            "vfmadd213sd 0*8(%[c0]          ), %%xmm6, %%xmm15\n\t"
	            "\n\t"
	            "vmovsd  %%xmm15, 0*8(%[c0]          )\n\t"
	            "\n\t"
	            "addq  $1*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	            :[alpha]"m"(alpha4),[ldc1]"r"(ldc1)
	        );

	        B = B - 2*K;

	    }

	    A = A - M*K;
	    B = B + 1*K;
	    c0 = c0- M + 1*ldc;


	}

	A = A + M*K;
	B = B - K*N;
	c0 = c0- ldc*N + M;

}

