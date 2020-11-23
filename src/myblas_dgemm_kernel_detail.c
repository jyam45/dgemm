#include "myblas_internal.h"
#include <stdio.h>
#include <stdlib.h>

/*  case 12x4x8x1

for( size_t k=0; k<8; k++ ){
 for( size_t j=0; j<4; j++ ){
   for( size_t i=0; i<12; i++ ){
      for( size_t l=0; l<1; l++ ){
        c[i+8*j] += (*(A2+l+i*1+k*1*12))*(*(B2+l+j*1+k*1*4));
      }
   }
 }
}
A2+=12*8*1;
B2+=8*1*4;#

*/

void myblas_dgemm_kernel_detail(
         size_t M, size_t N, size_t K,
         double alpha, const double *A, const double *B, 
         double *C, size_t ldc )
{

	double *c0 = C;
	double *c1 = C + ldc;
	size_t ldc2 = ldc * 2 * sizeof(double);
	double alpha4[4] = {alpha,alpha,alpha,alpha};

	size_t M12Q = M / 12;
	size_t M12R = M % 12;

	// ---- Kernel
	if( N >> 2 ){
	  size_t n4 = ( N >> 2 );
	  while( n4-- ){

	    if( M12Q ){
	      size_t m = M12Q;
	      while( m-- ){
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
	            "prefetcht0   0*8(%[c0]          )\n\t"
	            "prefetcht0   4*8(%[c0]          )\n\t"
	            "prefetcht0   8*8(%[c0]          )\n\t"
	            "prefetcht0   0*8(%[c1]          )\n\t"
	            "prefetcht0   4*8(%[c1]          )\n\t"
	            "prefetcht0   8*8(%[c1]          )\n\t"
	            "prefetcht0   0*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0   4*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0   8*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0   0*8(%[c1],%[ldc2]  )\n\t"
	            "prefetcht0   4*8(%[c1],%[ldc2]  )\n\t"
	            "prefetcht0   8*8(%[c1],%[ldc2]  )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){

	           // 
	           // c[i+8*j] += (*(A+l+i*1+k*1*12))*(*(B+l+j*1+k*1*4));
	           // 
	           // a[i,k]
	           // 
	           // a00 = *(A+0+0*1+0*1*12)
	           // a10 = *(A+0+1*1+0*1*12)
	           // a20 = *(A+0+2*1+0*1*12)
	           // a30 = *(A+0+3*1+0*1*12)
	           // ...
	           // ab0 = *(A+0+12*1+0*1*12)
	           // a01 = *(A+0+0*1+1*1*12)
	           // a11 = *(A+0+0*1+1*1*12)
	           // ...
	           // ab7 = *(A+0+12*1+7*1*12)
	           // 
	           // b[j,k]
	           // 
	           // b00 = *(B+0+0*1+0*1*4) 
	           // b10 = *(B+0+1*1+0*1*4) 
	           // b20 = *(B+0+2*1+0*1*4) 
	           // b30 = *(B+0+3*1+0*1*4) 
	           // ...
	           // b37 = *(B+0+3*1+7*1*4) 
	           // 
	           // c[i,j]
	           // 
	           // k=0 block
	           // c00 += a00*b00; c10 += a10*b00; c20 += a20*b00; c30 += a30*b00;
	           // c01 += a00*b10; c11 += a10*b10; c21 += a20*b10; c31 += a30*b10;
	           // c02 += a00*b20; c12 += a10*b20; c22 += a20*b20; c32 += a30*b20;
	           // c03 += a00*b30; c13 += a10*b30; c23 += a20*b30; c33 += a30*b30;
	           // 
	           // k=1 block
	           // c00 += a01*b01; c10 += a11*b01; c20 += a21*b01; c30 += a31*b01;
	           // c01 += a01*b11; c11 += a11*b11; c21 += a21*b11; c31 += a31*b11;
	           // c02 += a01*b21; c12 += a11*b21; c22 += a21*b21; c32 += a31*b21;
	           // c03 += a01*b31; c13 += a11*b31; c23 += a21*b31; c33 += a31*b31;
	           // 
	           // ...
	           // 
	           // k=7 block
	           // c00 += a07*b07; c10 += a17*b07; c20 += a27*b07; c30 += a37*b07;
	           // c01 += a07*b17; c11 += a17*b17; c21 += a27*b17; c31 += a37*b17;
	           // c02 += a07*b27; c11 += a17*b27; c22 += a27*b27; c32 += a37*b27;
	           // c03 += a07*b37; c13 += a17*b37; c23 += a27*b37; c33 += a37*b37;
	           // 
	           // 
	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "prefetcht0  288*8(%[a])\n\t"
	                "prefetcht0   96*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd       0*8(%[a]), %%ymm1 \n\t" // [a00,a10,a20,a30]
	                "vmovapd       4*8(%[a]), %%ymm2 \n\t" // [a40,a50,a60,a70]
	                "vmovapd       8*8(%[a]), %%ymm3 \n\t" // [a80,a90,aA0,aB0]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b00,b00,b00,b00]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c80,c90,cA0,cB0]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b10,b10,b10,b10]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c81,c91,cA1,cB1]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b20,b20,b20,b20]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c82,c92,cA2,cB2]
	                "vbroadcastsd  3*8(%[b]), %%ymm7 \n\t" // [b30,b30,b30,b30]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c83,c93,cA3,cB3]
	                "\n\t" // k=1
	                "prefetcht0  304*8(%[a])\n\t"
	                "prefetcht0  112*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd      12*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vmovapd      16*8(%[a]), %%ymm2 \n\t" // [a41,a51,a61,a71]
	                "vmovapd      20*8(%[a]), %%ymm3 \n\t" // [a81,a91,aA1,aB1]
	                "vbroadcastsd  4*8(%[b]), %%ymm0 \n\t" // [b01,b01,b01,b01]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c80,c90,cA0,cB0]
	                "vbroadcastsd  5*8(%[b]), %%ymm0 \n\t" // [b11,b11,b11,b11]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c81,c91,cA1,cB1]
	                "vbroadcastsd  6*8(%[b]), %%ymm0 \n\t" // [b21,b21,b21,b21]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c82,c92,cA2,cB2]
	                "vbroadcastsd  7*8(%[b]), %%ymm0 \n\t" // [b31,b31,b31,b31]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c83,c93,cA3,cB3]
	                "\n\t" // k=2
	                "prefetcht0  320*8(%[a])\n\t"
	                "prefetcht0  128*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd      24*8(%[a]), %%ymm1 \n\t" // [a02,a12,a22,a32]
	                "vmovapd      28*8(%[a]), %%ymm2 \n\t" // [a42,a52,a62,a72]
	                "vmovapd      32*8(%[a]), %%ymm3 \n\t" // [a82,a92,aA2,aB2]
	                "vbroadcastsd  8*8(%[b]), %%ymm0 \n\t" // [b02,b02,b02,b02]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c80,c90,cA0,cB0]
	                "vbroadcastsd  9*8(%[b]), %%ymm0 \n\t" // [b12,b12,b12,b12]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c81,c91,cA1,cB1]
	                "vbroadcastsd 10*8(%[b]), %%ymm0 \n\t" // [b22,b22,b22,b22]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c82,c92,cA2,cB2]
	                "vbroadcastsd 11*8(%[b]), %%ymm0 \n\t" // [b32,b32,b32,b32]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c83,c93,cA3,cB3]
	                "\n\t" // k=3
	                "prefetcht0  336*8(%[a])\n\t"
	                "prefetcht0  144*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd      36*8(%[a]), %%ymm1 \n\t" // [a03,a13,a23,a33]
	                "vmovapd      40*8(%[a]), %%ymm2 \n\t" // [a43,a53,a63,a73]
	                "vmovapd      44*8(%[a]), %%ymm3 \n\t" // [a83,a93,aA3,aB3]
	                "vbroadcastsd 12*8(%[b]), %%ymm0 \n\t" // [b03,b03,b03,b03]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c80,c90,cA0,cB0]
	                "vbroadcastsd 13*8(%[b]), %%ymm0 \n\t" // [b13,b13,b13,b13]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c81,c91,cA1,cB1]
	                "vbroadcastsd 14*8(%[b]), %%ymm0 \n\t" // [b23,b23,b23,b23]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c82,c92,cA2,cB2]
	                "vbroadcastsd 15*8(%[b]), %%ymm0 \n\t" // [b33,b33,b33,b33]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c83,c93,cA3,cB3]
	                "\n\t"
	                "addq  $48*8, %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                "\n\t" // k=4
	                "prefetcht0  288*8(%[a])\n\t"
	                "prefetcht0   96*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd       0*8(%[a]), %%ymm1 \n\t" // [a04,a14,a24,a34]
	                "vmovapd       4*8(%[a]), %%ymm2 \n\t" // [a44,a54,a64,a74]
	                "vmovapd       8*8(%[a]), %%ymm3 \n\t" // [a84,a94,aA4,aB4]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b04,b04,b04,b04]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c80,c90,cA0,cB0]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b14,b14,b14,b14]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c81,c91,cA1,cB1]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b24,b24,b24,b24]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c82,c92,cA2,cB2]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b34,b34,b34,b34]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c83,c93,cA3,cB3]
	                "\n\t" // k=5
	                "prefetcht0  304*8(%[a])\n\t"
	                "prefetcht0  112*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd      12*8(%[a]), %%ymm1 \n\t" // [a05,a15,a25,a35]
	                "vmovapd      16*8(%[a]), %%ymm2 \n\t" // [a45,a55,a65,a75]
	                "vmovapd      20*8(%[a]), %%ymm3 \n\t" // [a85,a95,aA5,aB5]
	                "vbroadcastsd  4*8(%[b]), %%ymm0 \n\t" // [b05,b05,b05,b05]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c80,c90,cA0,cB0]
	                "vbroadcastsd  5*8(%[b]), %%ymm0 \n\t" // [b15,b15,b15,b15]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c81,c91,cA1,cB1]
	                "vbroadcastsd  6*8(%[b]), %%ymm0 \n\t" // [b25,b25,b25,b25]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c82,c92,cA2,cB2]
	                "vbroadcastsd  7*8(%[b]), %%ymm0 \n\t" // [b35,b35,b35,b35]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c83,c93,cA3,cB3]
	                "\n\t" // k=6
	                "prefetcht0  320*8(%[a])\n\t"
	                "prefetcht0  128*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd      24*8(%[a]), %%ymm1 \n\t" // [a06,a16,a26,a36]
	                "vmovapd      28*8(%[a]), %%ymm2 \n\t" // [a46,a56,a66,a76]
	                "vmovapd      32*8(%[a]), %%ymm3 \n\t" // [a86,a96,aA6,aB6]
	                "vbroadcastsd  8*8(%[b]), %%ymm0 \n\t" // [b06,b06,b06,b06]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c80,c90,cA0,cB0]
	                "vbroadcastsd  9*8(%[b]), %%ymm0 \n\t" // [b16,b16,b16,b16]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c81,c91,cA1,cB1]
	                "vbroadcastsd 10*8(%[b]), %%ymm0 \n\t" // [b26,b26,b26,b26]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c82,c92,cA2,cB2]
	                "vbroadcastsd 11*8(%[b]), %%ymm0 \n\t" // [b36,b36,b36,b36]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c83,c93,cA3,cB3]
	                "\n\t" // k=7
	                "prefetcht0  336*8(%[a])\n\t"
	                "prefetcht0  144*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd      36*8(%[a]), %%ymm1 \n\t" // [a07,a17,a27,a37]
	                "vmovapd      40*8(%[a]), %%ymm2 \n\t" // [a47,a57,a67,a77]
	                "vmovapd      44*8(%[a]), %%ymm3 \n\t" // [a87,a97,aA7,aB7]
	                "vbroadcastsd 12*8(%[b]), %%ymm0 \n\t" // [b07,b07,b07,b07]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c80,c90,cA0,cB0]
	                "vbroadcastsd 13*8(%[b]), %%ymm0 \n\t" // [b17,b17,b17,b17]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c81,c91,cA1,cB1]
	                "vbroadcastsd 14*8(%[b]), %%ymm0 \n\t" // [b27,b27,b27,b27]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c82,c92,cA2,cB2]
	                "vbroadcastsd 15*8(%[b]), %%ymm0 \n\t" // [b37,b37,b37,b37]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c83,c93,cA3,cB3]
	                "\n\t"
	                "addq  $48*8, %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 4 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd       0*8(%[a]), %%ymm1 \n\t" // [a00,a10,a20,a30]
	                "vmovapd       4*8(%[a]), %%ymm2 \n\t" // [a40,a50,a60,a70]
	                "vmovapd       8*8(%[a]), %%ymm3 \n\t" // [a80,a90,aA0,aB0]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b00,b00,b00,b00]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c80,c90,cA0,cB0]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b10,b10,b10,b10]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c81,c91,cA1,cB1]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b20,b20,b20,b20]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c82,c92,cA2,cB2]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b30,b30,b30,b30]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c83,c93,cA3,cB3]
	                "\n\t" // k=1
	                "vmovapd      12*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vmovapd      16*8(%[a]), %%ymm2 \n\t" // [a41,a51,a61,a71]
	                "vmovapd      20*8(%[a]), %%ymm3 \n\t" // [a81,a91,aA1,aB1]
	                "vbroadcastsd  4*8(%[b]), %%ymm0 \n\t" // [b01,b01,b01,b01]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c80,c90,cA0,cB0]
	                "vbroadcastsd  5*8(%[b]), %%ymm0 \n\t" // [b11,b11,b11,b11]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c81,c91,cA1,cB1]
	                "vbroadcastsd  6*8(%[b]), %%ymm0 \n\t" // [b21,b21,b21,b21]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c82,c92,cA2,cB2]
	                "vbroadcastsd  7*8(%[b]), %%ymm0 \n\t" // [b31,b31,b31,b31]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c83,c93,cA3,cB3]
	                "\n\t" // k=2
	                "vmovapd      24*8(%[a]), %%ymm1 \n\t" // [a02,a12,a22,a32]
	                "vmovapd      28*8(%[a]), %%ymm2 \n\t" // [a42,a52,a62,a72]
	                "vmovapd      32*8(%[a]), %%ymm3 \n\t" // [a82,a92,aA2,aB2]
	                "vbroadcastsd  8*8(%[b]), %%ymm0 \n\t" // [b02,b02,b02,b02]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c80,c90,cA0,cB0]
	                "vbroadcastsd  9*8(%[b]), %%ymm0 \n\t" // [b12,b12,b12,b12]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c81,c91,cA1,cB1]
	                "vbroadcastsd 10*8(%[b]), %%ymm0 \n\t" // [b22,b22,b22,b22]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c82,c92,cA2,cB2]
	                "vbroadcastsd 11*8(%[b]), %%ymm0 \n\t" // [b32,b32,b32,b32]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c83,c93,cA3,cB3]
	                "\n\t" // k=3
	                "vmovapd      36*8(%[a]), %%ymm1 \n\t" // [a03,a13,a23,a33]
	                "vmovapd      40*8(%[a]), %%ymm2 \n\t" // [a43,a53,a63,a73]
	                "vmovapd      44*8(%[a]), %%ymm3 \n\t" // [a83,a93,aA3,aB3]
	                "vbroadcastsd 12*8(%[b]), %%ymm0 \n\t" // [b03,b03,b03,b03]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c80,c90,cA0,cB0]
	                "vbroadcastsd 13*8(%[b]), %%ymm0 \n\t" // [b13,b13,b13,b13]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c81,c91,cA1,cB1]
	                "vbroadcastsd 14*8(%[b]), %%ymm0 \n\t" // [b23,b23,b23,b23]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c82,c92,cA2,cB2]
	                "vbroadcastsd 15*8(%[b]), %%ymm0 \n\t" // [b33,b33,b33,b33]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c83,c93,cA3,cB3]
	                "\n\t"
	                "addq  $48*8, %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd       0*8(%[a]), %%ymm1 \n\t" // [a00,a10,a20,a30]
	                "vmovapd       4*8(%[a]), %%ymm2 \n\t" // [a40,a50,a60,a70]
	                "vmovapd       8*8(%[a]), %%ymm3 \n\t" // [a80,a90,aA0,aB0]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b00,b00,b00,b00]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c80,c90,cA0,cB0]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b10,b10,b10,b10]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c81,c91,cA1,cB1]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b20,b20,b20,b20]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c82,c92,cA2,cB2]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b30,b30,b30,b30]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c83,c93,cA3,cB3]
	                "\n\t" // k=1
	                "vmovapd      12*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vmovapd      16*8(%[a]), %%ymm2 \n\t" // [a41,a51,a61,a71]
	                "vmovapd      20*8(%[a]), %%ymm3 \n\t" // [a81,a91,aA1,aB1]
	                "vbroadcastsd  4*8(%[b]), %%ymm0 \n\t" // [b01,b01,b01,b01]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c80,c90,cA0,cB0]
	                "vbroadcastsd  5*8(%[b]), %%ymm0 \n\t" // [b11,b11,b11,b11]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c81,c91,cA1,cB1]
	                "vbroadcastsd  6*8(%[b]), %%ymm0 \n\t" // [b21,b21,b21,b21]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c82,c92,cA2,cB2]
	                "vbroadcastsd  7*8(%[b]), %%ymm0 \n\t" // [b31,b31,b31,b31]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c83,c93,cA3,cB3]
	                "\n\t"
	                "addq  $24*8, %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd       0*8(%[a]), %%ymm1 \n\t" // [a00,a10,a20,a30]
	                "vmovapd       4*8(%[a]), %%ymm2 \n\t" // [a40,a50,a60,a70]
	                "vmovapd       8*8(%[a]), %%ymm3 \n\t" // [a80,a90,aA0,aB0]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b00,b00,b00,b00]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c80,c90,cA0,cB0]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b10,b10,b10,b10]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm9 \n\t" // [c81,c91,cA1,cB1]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b20,b20,b20,b20]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm12\n\t" // [c82,c92,cA2,cB2]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b30,b30,b30,b30]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm15\n\t" // [c83,c93,cA3,cB3]
	                "\n\t"
	                "addq  $12*8, %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%ymm0, %%ymm4 \n\t"
	            "vfmadd213pd 4*8(%[c0]          ), %%ymm0, %%ymm5 \n\t"
	            "vfmadd213pd 8*8(%[c0]          ), %%ymm0, %%ymm6 \n\t"
	            "vfmadd213pd 0*8(%[c1]          ), %%ymm0, %%ymm7 \n\t"
	            "vfmadd213pd 4*8(%[c1]          ), %%ymm0, %%ymm8 \n\t"
	            "vfmadd213pd 8*8(%[c1]          ), %%ymm0, %%ymm9 \n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc2]  ), %%ymm0, %%ymm10\n\t"
	            "vfmadd213pd 4*8(%[c0],%[ldc2]  ), %%ymm0, %%ymm11\n\t"
	            "vfmadd213pd 8*8(%[c0],%[ldc2]  ), %%ymm0, %%ymm12\n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2]  ), %%ymm0, %%ymm13\n\t"
	            "vfmadd213pd 4*8(%[c1],%[ldc2]  ), %%ymm0, %%ymm14\n\t"
	            "vfmadd213pd 8*8(%[c1],%[ldc2]  ), %%ymm0, %%ymm15\n\t"
	            "\n\t"
	            "vmovupd  %%ymm4 , 0*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm5 , 4*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm6 , 8*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm7 , 0*8(%[c1]          )\n\t"
	            "vmovupd  %%ymm8 , 4*8(%[c1]          )\n\t"
	            "vmovupd  %%ymm9 , 8*8(%[c1]          )\n\t"
	            "vmovupd  %%ymm10, 0*8(%[c0],%[ldc2]  )\n\t"
	            "vmovupd  %%ymm11, 4*8(%[c0],%[ldc2]  )\n\t"
	            "vmovupd  %%ymm12, 8*8(%[c0],%[ldc2]  )\n\t"
	            "vmovupd  %%ymm13, 0*8(%[c1],%[ldc2]  )\n\t"
	            "vmovupd  %%ymm14, 4*8(%[c1],%[ldc2]  )\n\t"
	            "vmovupd  %%ymm15, 8*8(%[c1],%[ldc2]  )\n\t"
	            "\n\t"
	            "addq  $12*8, %[c0]\n\t"
	            "addq  $12*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 4*K;

	      }
	    }
	    if( M12R >> 3 ){
	      size_t m = ( M12R >> 3 );
	      while( m-- ){
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
	            "prefetcht0   0*8(%[c0]          )\n\t"
	            "prefetcht0   4*8(%[c0]          )\n\t"
	            "prefetcht0   0*8(%[c1]          )\n\t"
	            "prefetcht0   4*8(%[c1]          )\n\t"
	            "prefetcht0   0*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0   4*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0   0*8(%[c1],%[ldc2]  )\n\t"
	            "prefetcht0   4*8(%[c1],%[ldc2]  )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );

	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "prefetcht0  192*8(%[a])\n\t"
	                "prefetcht0   96*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd       0*8(%[a]), %%ymm1 \n\t" // [a00,a10,a20,a30]
	                "vmovapd       4*8(%[a]), %%ymm2 \n\t" // [a40,a50,a60,a70]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b00,b00,b00,b00]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b10,b10,b10,b10]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b20,b20,b20,b20]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b30,b30,b30,b30]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "\n\t" // k=1
	                "prefetcht0  208*8(%[a])\n\t"
	                "prefetcht0  112*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd       8*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vmovapd      12*8(%[a]), %%ymm2 \n\t" // [a41,a51,a61,a71]
	                "vbroadcastsd  4*8(%[b]), %%ymm0 \n\t" // [b01,b01,b01,b01]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vbroadcastsd  5*8(%[b]), %%ymm0 \n\t" // [b11,b11,b11,b11]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastsd  6*8(%[b]), %%ymm0 \n\t" // [b21,b21,b21,b21]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vbroadcastsd  7*8(%[b]), %%ymm0 \n\t" // [b31,b31,b31,b31]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "\n\t" // k=2
	                "prefetcht0  224*8(%[a])\n\t"
	                "prefetcht0  128*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd      16*8(%[a]), %%ymm1 \n\t" // [a02,a12,a22,a32]
	                "vmovapd      20*8(%[a]), %%ymm2 \n\t" // [a42,a52,a62,a72]
	                "vbroadcastsd  8*8(%[b]), %%ymm0 \n\t" // [b02,b02,b02,b02]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vbroadcastsd  9*8(%[b]), %%ymm0 \n\t" // [b12,b12,b12,b12]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastsd 10*8(%[b]), %%ymm0 \n\t" // [b22,b22,b22,b22]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vbroadcastsd 11*8(%[b]), %%ymm0 \n\t" // [b32,b32,b32,b32]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "\n\t" // k=3
	                "prefetcht0  240*8(%[a])\n\t"
	                "prefetcht0  144*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd      24*8(%[a]), %%ymm1 \n\t" // [a03,a13,a23,a33]
	                "vmovapd      28*8(%[a]), %%ymm2 \n\t" // [a43,a53,a63,a73]
	                "vbroadcastsd 12*8(%[b]), %%ymm0 \n\t" // [b03,b03,b03,b03]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vbroadcastsd 13*8(%[b]), %%ymm0 \n\t" // [b13,b13,b13,b13]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastsd 14*8(%[b]), %%ymm0 \n\t" // [b23,b23,b23,b23]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vbroadcastsd 15*8(%[b]), %%ymm0 \n\t" // [b33,b33,b33,b33]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "\n\t"
	                "addq  $32*8, %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                "\n\t" // k=4
	                "prefetcht0  192*8(%[a])\n\t"
	                "prefetcht0   96*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd       0*8(%[a]), %%ymm1 \n\t" // [a04,a14,a24,a34]
	                "vmovapd       4*8(%[a]), %%ymm2 \n\t" // [a44,a54,a64,a74]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b04,b04,b04,b04]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b14,b14,b14,b14]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b24,b24,b24,b24]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b34,b34,b34,b34]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "\n\t" // k=5
	                "prefetcht0  208*8(%[a])\n\t"
	                "prefetcht0  112*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd       8*8(%[a]), %%ymm1 \n\t" // [a05,a15,a25,a35]
	                "vmovapd      12*8(%[a]), %%ymm2 \n\t" // [a45,a55,a65,a75]
	                "vbroadcastsd  4*8(%[b]), %%ymm0 \n\t" // [b05,b05,b05,b05]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vbroadcastsd  5*8(%[b]), %%ymm0 \n\t" // [b15,b15,b15,b15]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastsd  6*8(%[b]), %%ymm0 \n\t" // [b25,b25,b25,b25]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vbroadcastsd  7*8(%[b]), %%ymm0 \n\t" // [b35,b35,b35,b35]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "\n\t" // k=6
	                "prefetcht0  224*8(%[a])\n\t"
	                "prefetcht0  128*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd      16*8(%[a]), %%ymm1 \n\t" // [a06,a16,a26,a36]
	                "vmovapd      20*8(%[a]), %%ymm2 \n\t" // [a46,a56,a66,a76]
	                "vbroadcastsd  8*8(%[b]), %%ymm0 \n\t" // [b06,b06,b06,b06]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vbroadcastsd  9*8(%[b]), %%ymm0 \n\t" // [b16,b16,b16,b16]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastsd 10*8(%[b]), %%ymm0 \n\t" // [b26,b26,b26,b26]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vbroadcastsd 11*8(%[b]), %%ymm0 \n\t" // [b36,b36,b36,b36]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "\n\t" // k=7
	                "prefetcht0  240*8(%[a])\n\t"
	                "prefetcht0  144*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd      24*8(%[a]), %%ymm1 \n\t" // [a07,a17,a27,a37]
	                "vmovapd      28*8(%[a]), %%ymm2 \n\t" // [a47,a57,a67,a77]
	                "vbroadcastsd 12*8(%[b]), %%ymm0 \n\t" // [b07,b07,b07,b07]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vbroadcastsd 13*8(%[b]), %%ymm0 \n\t" // [b17,b17,b17,b17]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastsd 14*8(%[b]), %%ymm0 \n\t" // [b27,b27,b27,b27]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vbroadcastsd 15*8(%[b]), %%ymm0 \n\t" // [b37,b37,b37,b37]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "\n\t"
	                "addq  $32*8, %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 4 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd       0*8(%[a]), %%ymm1 \n\t" // [a00,a10,a20,a30]
	                "vmovapd       4*8(%[a]), %%ymm2 \n\t" // [a40,a50,a60,a70]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b00,b00,b00,b00]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b10,b10,b10,b10]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b20,b20,b20,b20]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b30,b30,b30,b30]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "\n\t" // k=1
	                "vmovapd       8*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vmovapd      12*8(%[a]), %%ymm2 \n\t" // [a41,a51,a61,a71]
	                "vbroadcastsd  4*8(%[b]), %%ymm0 \n\t" // [b01,b01,b01,b01]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vbroadcastsd  5*8(%[b]), %%ymm0 \n\t" // [b11,b11,b11,b11]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastsd  6*8(%[b]), %%ymm0 \n\t" // [b21,b21,b21,b21]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vbroadcastsd  7*8(%[b]), %%ymm0 \n\t" // [b31,b31,b31,b31]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "\n\t" // k=2
	                "vmovapd      16*8(%[a]), %%ymm1 \n\t" // [a02,a12,a22,a32]
	                "vmovapd      20*8(%[a]), %%ymm2 \n\t" // [a42,a52,a62,a72]
	                "vbroadcastsd  8*8(%[b]), %%ymm0 \n\t" // [b02,b02,b02,b02]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vbroadcastsd  9*8(%[b]), %%ymm0 \n\t" // [b12,b12,b12,b12]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastsd 10*8(%[b]), %%ymm0 \n\t" // [b22,b22,b22,b22]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vbroadcastsd 11*8(%[b]), %%ymm0 \n\t" // [b32,b32,b32,b32]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "\n\t" // k=3
	                "vmovapd      24*8(%[a]), %%ymm1 \n\t" // [a03,a13,a23,a33]
	                "vmovapd      28*8(%[a]), %%ymm2 \n\t" // [a43,a53,a63,a73]
	                "vbroadcastsd 12*8(%[b]), %%ymm0 \n\t" // [b03,b03,b03,b03]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vbroadcastsd 13*8(%[b]), %%ymm0 \n\t" // [b13,b13,b13,b13]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastsd 14*8(%[b]), %%ymm0 \n\t" // [b23,b23,b23,b23]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vbroadcastsd 15*8(%[b]), %%ymm0 \n\t" // [b33,b33,b33,b33]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "\n\t"
	                "addq  $32*8, %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd       0*8(%[a]), %%ymm1 \n\t" // [a00,a10,a20,a30]
	                "vmovapd       4*8(%[a]), %%ymm2 \n\t" // [a40,a50,a60,a70]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b00,b00,b00,b00]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b10,b10,b10,b10]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b20,b20,b20,b20]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b30,b30,b30,b30]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "\n\t" // k=1
	                "vmovapd       8*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vmovapd      12*8(%[a]), %%ymm2 \n\t" // [a41,a51,a61,a71]
	                "vbroadcastsd  4*8(%[b]), %%ymm0 \n\t" // [b01,b01,b01,b01]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vbroadcastsd  5*8(%[b]), %%ymm0 \n\t" // [b11,b11,b11,b11]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastsd  6*8(%[b]), %%ymm0 \n\t" // [b21,b21,b21,b21]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vbroadcastsd  7*8(%[b]), %%ymm0 \n\t" // [b31,b31,b31,b31]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd       0*8(%[a]), %%ymm1 \n\t" // [a00,a10,a20,a30]
	                "vmovapd       4*8(%[a]), %%ymm2 \n\t" // [a40,a50,a60,a70]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b00,b00,b00,b00]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b10,b10,b10,b10]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b20,b20,b20,b20]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm11\n\t" // [c42,c52,c62,c72]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b30,b30,b30,b30]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm14\n\t" // [c43,c53,c63,c73]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%ymm0, %%ymm4 \n\t"
	            "vfmadd213pd 4*8(%[c0]          ), %%ymm0, %%ymm5 \n\t"
	            "vfmadd213pd 0*8(%[c1]          ), %%ymm0, %%ymm7 \n\t"
	            "vfmadd213pd 4*8(%[c1]          ), %%ymm0, %%ymm8 \n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc2]  ), %%ymm0, %%ymm10\n\t"
	            "vfmadd213pd 4*8(%[c0],%[ldc2]  ), %%ymm0, %%ymm11\n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2]  ), %%ymm0, %%ymm13\n\t"
	            "vfmadd213pd 4*8(%[c1],%[ldc2]  ), %%ymm0, %%ymm14\n\t"
	            "\n\t"
	            "vmovupd  %%ymm4 , 0*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm5 , 4*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm7 , 0*8(%[c1]          )\n\t"
	            "vmovupd  %%ymm8 , 4*8(%[c1]          )\n\t"
	            "vmovupd  %%ymm10, 0*8(%[c0],%[ldc2]  )\n\t"
	            "vmovupd  %%ymm11, 4*8(%[c0],%[ldc2]  )\n\t"
	            "vmovupd  %%ymm13, 0*8(%[c1],%[ldc2]  )\n\t"
	            "vmovupd  %%ymm14, 4*8(%[c1],%[ldc2]  )\n\t"
	            "\n\t"
	            "addq  $8*8, %[c0]\n\t"
	            "addq  $8*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 4*K;

	      }
	    }
	    if( M12R & 4 ){

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
	            "prefetcht0   0*8(%[c0]          )\n\t"
	            "prefetcht0   0*8(%[c1]          )\n\t"
	            "prefetcht0   0*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0   0*8(%[c1],%[ldc2]  )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "prefetcht0   96*8(%[a])\n\t"
	                "prefetcht0   96*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd       0*8(%[a]), %%ymm1 \n\t" // [a00,a10,a20,a30]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b00,b00,b00,b00]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b10,b10,b10,b10]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b20,b20,b20,b20]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b30,b30,b30,b30]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t" // k=1
	                "prefetcht0  112*8(%[a])\n\t"
	                "prefetcht0  112*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd       4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vbroadcastsd  4*8(%[b]), %%ymm0 \n\t" // [b01,b01,b01,b01]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vbroadcastsd  5*8(%[b]), %%ymm0 \n\t" // [b11,b11,b11,b11]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vbroadcastsd  6*8(%[b]), %%ymm0 \n\t" // [b21,b21,b21,b21]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vbroadcastsd  7*8(%[b]), %%ymm0 \n\t" // [b31,b31,b31,b31]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t" // k=2
	                "prefetcht0  128*8(%[a])\n\t"
	                "prefetcht0  128*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd       8*8(%[a]), %%ymm1 \n\t" // [a02,a12,a22,a32]
	                "vbroadcastsd  8*8(%[b]), %%ymm0 \n\t" // [b02,b02,b02,b02]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vbroadcastsd  9*8(%[b]), %%ymm0 \n\t" // [b12,b12,b12,b12]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vbroadcastsd 10*8(%[b]), %%ymm0 \n\t" // [b22,b22,b22,b22]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vbroadcastsd 11*8(%[b]), %%ymm0 \n\t" // [b32,b32,b32,b32]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t" // k=3
	                "prefetcht0  144*8(%[a])\n\t"
	                "prefetcht0  144*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd      12*8(%[a]), %%ymm1 \n\t" // [a03,a13,a23,a33]
	                "vbroadcastsd 12*8(%[b]), %%ymm0 \n\t" // [b03,b03,b03,b03]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vbroadcastsd 13*8(%[b]), %%ymm0 \n\t" // [b13,b13,b13,b13]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vbroadcastsd 14*8(%[b]), %%ymm0 \n\t" // [b23,b23,b23,b23]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vbroadcastsd 15*8(%[b]), %%ymm0 \n\t" // [b33,b33,b33,b33]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                "\n\t" // k=4
	                "prefetcht0   96*8(%[a])\n\t"
	                "prefetcht0   96*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd       0*8(%[a]), %%ymm1 \n\t" // [a04,a14,a24,a34]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b04,b04,b04,b04]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b14,b14,b14,b14]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b24,b24,b24,b24]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b34,b34,b34,b34]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t" // k=5
	                "prefetcht0  112*8(%[a])\n\t"
	                "prefetcht0  112*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd       4*8(%[a]), %%ymm1 \n\t" // [a05,a15,a25,a35]
	                "vbroadcastsd  4*8(%[b]), %%ymm0 \n\t" // [b05,b05,b05,b05]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vbroadcastsd  5*8(%[b]), %%ymm0 \n\t" // [b15,b15,b15,b15]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vbroadcastsd  6*8(%[b]), %%ymm0 \n\t" // [b25,b25,b25,b25]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vbroadcastsd  7*8(%[b]), %%ymm0 \n\t" // [b35,b35,b35,b35]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t" // k=6
	                "prefetcht0  128*8(%[a])\n\t"
	                "prefetcht0  128*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd       8*8(%[a]), %%ymm1 \n\t" // [a06,a16,a26,a36]
	                "vbroadcastsd  8*8(%[b]), %%ymm0 \n\t" // [b06,b06,b06,b06]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vbroadcastsd  9*8(%[b]), %%ymm0 \n\t" // [b16,b16,b16,b16]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vbroadcastsd 10*8(%[b]), %%ymm0 \n\t" // [b26,b26,b26,b26]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vbroadcastsd 11*8(%[b]), %%ymm0 \n\t" // [b36,b36,b36,b36]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t" // k=7
	                "prefetcht0  144*8(%[a])\n\t"
	                "prefetcht0  144*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd      12*8(%[a]), %%ymm1 \n\t" // [a07,a17,a27,a37]
	                "vbroadcastsd 12*8(%[b]), %%ymm0 \n\t" // [b07,b07,b07,b07]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vbroadcastsd 13*8(%[b]), %%ymm0 \n\t" // [b17,b17,b17,b17]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vbroadcastsd 14*8(%[b]), %%ymm0 \n\t" // [b27,b27,b27,b27]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vbroadcastsd 15*8(%[b]), %%ymm0 \n\t" // [b37,b37,b37,b37]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 4 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd       0*8(%[a]), %%ymm1 \n\t" // [a00,a10,a20,a30]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b00,b00,b00,b00]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b10,b10,b10,b10]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b20,b20,b20,b20]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b30,b30,b30,b30]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t" // k=1
	                "vmovapd       4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vbroadcastsd  4*8(%[b]), %%ymm0 \n\t" // [b01,b01,b01,b01]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vbroadcastsd  5*8(%[b]), %%ymm0 \n\t" // [b11,b11,b11,b11]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vbroadcastsd  6*8(%[b]), %%ymm0 \n\t" // [b21,b21,b21,b21]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vbroadcastsd  7*8(%[b]), %%ymm0 \n\t" // [b31,b31,b31,b31]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t" // k=2
	                "vmovapd       8*8(%[a]), %%ymm1 \n\t" // [a02,a12,a22,a32]
	                "vbroadcastsd  8*8(%[b]), %%ymm0 \n\t" // [b02,b02,b02,b02]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vbroadcastsd  9*8(%[b]), %%ymm0 \n\t" // [b12,b12,b12,b12]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vbroadcastsd 10*8(%[b]), %%ymm0 \n\t" // [b22,b22,b22,b22]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vbroadcastsd 11*8(%[b]), %%ymm0 \n\t" // [b32,b32,b32,b32]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t" // k=3
	                "vmovapd      12*8(%[a]), %%ymm1 \n\t" // [a03,a13,a23,a33]
	                "vbroadcastsd 12*8(%[b]), %%ymm0 \n\t" // [b03,b03,b03,b03]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vbroadcastsd 13*8(%[b]), %%ymm0 \n\t" // [b13,b13,b13,b13]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vbroadcastsd 14*8(%[b]), %%ymm0 \n\t" // [b23,b23,b23,b23]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vbroadcastsd 15*8(%[b]), %%ymm0 \n\t" // [b33,b33,b33,b33]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd       0*8(%[a]), %%ymm1 \n\t" // [a00,a10,a20,a30]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b00,b00,b00,b00]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b10,b10,b10,b10]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b20,b20,b20,b20]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b30,b30,b30,b30]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t" // k=1
	                "vmovapd       4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vbroadcastsd  4*8(%[b]), %%ymm0 \n\t" // [b01,b01,b01,b01]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vbroadcastsd  5*8(%[b]), %%ymm0 \n\t" // [b11,b11,b11,b11]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vbroadcastsd  6*8(%[b]), %%ymm0 \n\t" // [b21,b21,b21,b21]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vbroadcastsd  7*8(%[b]), %%ymm0 \n\t" // [b31,b31,b31,b31]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd       0*8(%[a]), %%ymm1 \n\t" // [a00,a10,a20,a30]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b00,b00,b00,b00]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b10,b10,b10,b10]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm7 \n\t" // [c01,c11,c21,c31]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b20,b20,b20,b20]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm10\n\t" // [c02,c12,c22,c32]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b30,b30,b30,b30]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%ymm0, %%ymm4 \n\t"
	            "vfmadd213pd 0*8(%[c1]          ), %%ymm0, %%ymm7 \n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc2]  ), %%ymm0, %%ymm10\n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2]  ), %%ymm0, %%ymm13\n\t"
	            "\n\t"
	            "vmovupd  %%ymm4 , 0*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm7 , 0*8(%[c1]          )\n\t"
	            "vmovupd  %%ymm10, 0*8(%[c0],%[ldc2]  )\n\t"
	            "vmovupd  %%ymm13, 0*8(%[c1],%[ldc2]  )\n\t"
	            "\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "addq  $4*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 4*K;


	    }
	    if( M12R & 2 ){

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
	            "prefetcht0   0*8(%[c0]          )\n\t"
	            "prefetcht0   0*8(%[c1]          )\n\t"
	            "prefetcht0   0*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0   0*8(%[c1],%[ldc2]  )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "prefetcht0   48*8(%[a])\n\t"
	                "prefetcht0   96*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd       0*8(%[a]), %%xmm1 \n\t" // [a00,a10,---,---]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b00,b00,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm4 \n\t" // [c00,c10,---,---]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b10,b10,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm7 \n\t" // [c01,c11,---,---]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b20,b20,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm10\n\t" // [c02,c12,---,---]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b30,b30,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t" // k=1
	                "vmovapd       2*8(%[a]), %%xmm1 \n\t" // [a01,a11,---,---]
	                "vbroadcastsd  4*8(%[b]), %%ymm0 \n\t" // [b01,b01,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm4 \n\t" // [c00,c10,---,---]
	                "vbroadcastsd  5*8(%[b]), %%ymm0 \n\t" // [b11,b11,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm7 \n\t" // [c01,c11,---,---]
	                "vbroadcastsd  6*8(%[b]), %%ymm0 \n\t" // [b21,b21,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm10\n\t" // [c02,c12,---,---]
	                "vbroadcastsd  7*8(%[b]), %%ymm0 \n\t" // [b31,b31,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t" // k=2
	                "prefetcht0   56*8(%[a])\n\t"
	                "prefetcht0  112*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd       4*8(%[a]), %%xmm1 \n\t" // [a02,a12,---,---]
	                "vbroadcastsd  8*8(%[b]), %%ymm0 \n\t" // [b02,b02,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm4 \n\t" // [c00,c10,---,---]
	                "vbroadcastsd  9*8(%[b]), %%ymm0 \n\t" // [b12,b12,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm7 \n\t" // [c01,c11,---,---]
	                "vbroadcastsd 10*8(%[b]), %%ymm0 \n\t" // [b22,b22,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm10\n\t" // [c02,c12,---,---]
	                "vbroadcastsd 11*8(%[b]), %%ymm0 \n\t" // [b32,b32,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t" // k=3
	                "vmovapd       6*8(%[a]), %%xmm1 \n\t" // [a03,a13,---,---]
	                "vbroadcastsd 12*8(%[b]), %%ymm0 \n\t" // [b03,b03,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm4 \n\t" // [c00,c10,---,---]
	                "vbroadcastsd 13*8(%[b]), %%ymm0 \n\t" // [b13,b13,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm7 \n\t" // [c01,c11,---,---]
	                "vbroadcastsd 14*8(%[b]), %%ymm0 \n\t" // [b23,b23,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm10\n\t" // [c02,c12,---,---]
	                "vbroadcastsd 15*8(%[b]), %%ymm0 \n\t" // [b33,b33,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                "\n\t" // k=4
	                "prefetcht0   48*8(%[a])\n\t"
	                "prefetcht0   96*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd       0*8(%[a]), %%xmm1 \n\t" // [a04,a14,---,---]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b04,b04,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm4 \n\t" // [c00,c10,---,---]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b14,b14,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm7 \n\t" // [c01,c11,---,---]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b24,b24,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm10\n\t" // [c02,c12,---,---]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b34,b34,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t" // k=5
	                "vmovapd       2*8(%[a]), %%xmm1 \n\t" // [a05,a15,---,---]
	                "vbroadcastsd  4*8(%[b]), %%ymm0 \n\t" // [b05,b05,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm4 \n\t" // [c00,c10,---,---]
	                "vbroadcastsd  5*8(%[b]), %%ymm0 \n\t" // [b15,b15,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm7 \n\t" // [c01,c11,---,---]
	                "vbroadcastsd  6*8(%[b]), %%ymm0 \n\t" // [b25,b25,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm10\n\t" // [c02,c12,---,---]
	                "vbroadcastsd  7*8(%[b]), %%ymm0 \n\t" // [b35,b35,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t" // k=6
	                "prefetcht0   56*8(%[a])\n\t"
	                "prefetcht0  112*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd       4*8(%[a]), %%xmm1 \n\t" // [a06,a16,---,---]
	                "vbroadcastsd  8*8(%[b]), %%ymm0 \n\t" // [b06,b06,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm4 \n\t" // [c00,c10,---,---]
	                "vbroadcastsd  9*8(%[b]), %%ymm0 \n\t" // [b16,b16,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm7 \n\t" // [c01,c11,---,---]
	                "vbroadcastsd 10*8(%[b]), %%ymm0 \n\t" // [b26,b26,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm10\n\t" // [c02,c12,---,---]
	                "vbroadcastsd 11*8(%[b]), %%ymm0 \n\t" // [b36,b36,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t" // k=7
	                "vmovapd       6*8(%[a]), %%xmm1 \n\t" // [a07,a17,---,---]
	                "vbroadcastsd 12*8(%[b]), %%ymm0 \n\t" // [b07,b07,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm4 \n\t" // [c00,c10,---,---]
	                "vbroadcastsd 13*8(%[b]), %%ymm0 \n\t" // [b17,b17,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm7 \n\t" // [c01,c11,---,---]
	                "vbroadcastsd 14*8(%[b]), %%ymm0 \n\t" // [b27,b27,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm10\n\t" // [c02,c12,---,---]
	                "vbroadcastsd 15*8(%[b]), %%ymm0 \n\t" // [b37,b37,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c03,c13,---,---]
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
	                "\n\t" // k=0
	                "vmovapd       0*8(%[a]), %%xmm1 \n\t" // [a00,a10,---,---]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b00,b00,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm4 \n\t" // [c00,c10,---,---]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b10,b10,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm7 \n\t" // [c01,c11,---,---]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b20,b20,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm10\n\t" // [c02,c12,---,---]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b30,b30,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t" // k=1
	                "vmovapd       2*8(%[a]), %%xmm1 \n\t" // [a01,a11,---,---]
	                "vbroadcastsd  4*8(%[b]), %%ymm0 \n\t" // [b01,b01,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm4 \n\t" // [c00,c10,---,---]
	                "vbroadcastsd  5*8(%[b]), %%ymm0 \n\t" // [b11,b11,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm7 \n\t" // [c01,c11,---,---]
	                "vbroadcastsd  6*8(%[b]), %%ymm0 \n\t" // [b21,b21,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm10\n\t" // [c02,c12,---,---]
	                "vbroadcastsd  7*8(%[b]), %%ymm0 \n\t" // [b31,b31,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t" // k=2
	                "vmovapd       4*8(%[a]), %%xmm1 \n\t" // [a02,a12,---,---]
	                "vbroadcastsd  8*8(%[b]), %%ymm0 \n\t" // [b02,b02,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm4 \n\t" // [c00,c10,---,---]
	                "vbroadcastsd  9*8(%[b]), %%ymm0 \n\t" // [b12,b12,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm7 \n\t" // [c01,c11,---,---]
	                "vbroadcastsd 10*8(%[b]), %%ymm0 \n\t" // [b22,b22,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm10\n\t" // [c02,c12,---,---]
	                "vbroadcastsd 11*8(%[b]), %%ymm0 \n\t" // [b32,b32,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t" // k=3
	                "vmovapd       6*8(%[a]), %%xmm1 \n\t" // [a03,a13,---,---]
	                "vbroadcastsd 12*8(%[b]), %%ymm0 \n\t" // [b03,b03,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm4 \n\t" // [c00,c10,---,---]
	                "vbroadcastsd 13*8(%[b]), %%ymm0 \n\t" // [b13,b13,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm7 \n\t" // [c01,c11,---,---]
	                "vbroadcastsd 14*8(%[b]), %%ymm0 \n\t" // [b23,b23,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm10\n\t" // [c02,c12,---,---]
	                "vbroadcastsd 15*8(%[b]), %%ymm0 \n\t" // [b33,b33,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd       0*8(%[a]), %%xmm1 \n\t" // [a00,a10,---,---]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b00,b00,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm4 \n\t" // [c00,c10,---,---]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b10,b10,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm7 \n\t" // [c01,c11,---,---]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b20,b20,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm10\n\t" // [c02,c12,---,---]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b30,b30,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t" // k=1
	                "vmovapd       2*8(%[a]), %%xmm1 \n\t" // [a01,a11,---,---]
	                "vbroadcastsd  4*8(%[b]), %%ymm0 \n\t" // [b01,b01,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm4 \n\t" // [c00,c10,---,---]
	                "vbroadcastsd  5*8(%[b]), %%ymm0 \n\t" // [b11,b11,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm7 \n\t" // [c01,c11,---,---]
	                "vbroadcastsd  6*8(%[b]), %%ymm0 \n\t" // [b21,b21,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm10\n\t" // [c02,c12,---,---]
	                "vbroadcastsd  7*8(%[b]), %%ymm0 \n\t" // [b31,b31,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd       0*8(%[a]), %%xmm1 \n\t" // [a00,a10,---,---]
	                "vbroadcastsd  0*8(%[b]), %%ymm0 \n\t" // [b00,b00,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm4 \n\t" // [c00,c10,---,---]
	                "vbroadcastsd  1*8(%[b]), %%ymm0 \n\t" // [b10,b10,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm7 \n\t" // [c01,c11,---,---]
	                "vbroadcastsd  2*8(%[b]), %%ymm0 \n\t" // [b20,b20,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm10\n\t" // [c02,c12,---,---]
	                "vbroadcastsd  3*8(%[b]), %%ymm0 \n\t" // [b30,b30,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm1 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%xmm0, %%xmm4 \n\t"
	            "vfmadd213pd 0*8(%[c1]          ), %%xmm0, %%xmm7 \n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc2]  ), %%xmm0, %%xmm10\n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2]  ), %%xmm0, %%xmm13\n\t"
	            "\n\t"
	            "vmovupd  %%xmm4 , 0*8(%[c0]          )\n\t"
	            "vmovupd  %%xmm7 , 0*8(%[c1]          )\n\t"
	            "vmovupd  %%xmm10, 0*8(%[c0],%[ldc2]  )\n\t"
	            "vmovupd  %%xmm13, 0*8(%[c1],%[ldc2]  )\n\t"
	            "\n\t"
	            "addq  $2*8, %[c0]\n\t"
	            "addq  $2*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 4*K;

	    }
	    if( M12R & 1 ){

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
	            "prefetcht0   0*8(%[c0]          )\n\t"
	            "prefetcht0   0*8(%[c1]          )\n\t"
	            "prefetcht0   0*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0   0*8(%[c1],%[ldc2]  )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "prefetcht0   24*8(%[a])\n\t"
	                "prefetcht0   96*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastsd  0*8(%[a]), %%ymm1 \n\t" // [a00,a00,a00,a00]
	                "vmovapd       0*8(%[b]), %%ymm0 \n\t" // [b00,b10,b20,b30]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c01,c02,c03]
	                "\n\t" // k=1
	                "vbroadcastsd  1*8(%[a]), %%ymm1 \n\t" // [a01,a01,a01,a01]
	                "vmovapd       4*8(%[b]), %%ymm0 \n\t" // [b01,b11,b21,b31]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c01,c02,c03]
	                "\n\t" // k=2
	                "vbroadcastsd  2*8(%[a]), %%ymm1 \n\t" // [a02,a02,a02,a02]
	                "vmovapd       8*8(%[b]), %%ymm0 \n\t" // [b02,b12,b22,b32]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c01,c02,c03]
	                "\n\t" // k=3
	                "vbroadcastsd  3*8(%[a]), %%ymm1 \n\t" // [a03,a03,a03,a03]
	                "vmovapd      12*8(%[b]), %%ymm0 \n\t" // [b03,b13,b23,b33]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c01,c02,c03]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                "\n\t" // k=4
	                "prefetcht0   40*8(%[a])\n\t"
	                "prefetcht0  112*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastsd  0*8(%[a]), %%ymm1 \n\t" // [a04,a04,a04,a04]
	                "vmovapd       0*8(%[b]), %%ymm0 \n\t" // [b04,b14,b24,b34]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c01,c02,c03]
	                "\n\t" // k=5
	                "vbroadcastsd  1*8(%[a]), %%ymm1 \n\t" // [a05,a05,a05,a05]
	                "vmovapd       4*8(%[b]), %%ymm0 \n\t" // [b05,b15,b25,b35]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c01,c02,c03]
	                "\n\t" // k=6
	                "vbroadcastsd  2*8(%[a]), %%ymm1 \n\t" // [a06,a06,a06,a06]
	                "vmovapd       8*8(%[b]), %%ymm0 \n\t" // [b06,b16,b26,b36]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c01,c02,c03]
	                "\n\t" // k=7
	                "vbroadcastsd  3*8(%[a]), %%ymm1 \n\t" // [a07,a07,a07,a07]
	                "vmovapd      12*8(%[b]), %%ymm0 \n\t" // [b07,b17,b27,b37]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c01,c02,c03]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 4 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vbroadcastsd  0*8(%[a]), %%ymm1 \n\t" // [a00,a00,a00,a00]
	                "vmovapd       0*8(%[b]), %%ymm0 \n\t" // [b00,b10,b20,b30]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c01,c02,c03]
	                "\n\t" // k=1
	                "vbroadcastsd  1*8(%[a]), %%ymm1 \n\t" // [a01,a01,a01,a01]
	                "vmovapd       4*8(%[b]), %%ymm0 \n\t" // [b01,b11,b21,b31]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c01,c02,c03]
	                "\n\t" // k=2
	                "vbroadcastsd  2*8(%[a]), %%ymm1 \n\t" // [a02,a02,a02,a02]
	                "vmovapd       8*8(%[b]), %%ymm0 \n\t" // [b02,b12,b22,b32]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c01,c02,c03]
	                "\n\t" // k=3
	                "vbroadcastsd  3*8(%[a]), %%ymm1 \n\t" // [a03,a03,a03,a03]
	                "vmovapd      12*8(%[b]), %%ymm0 \n\t" // [b03,b13,b23,b33]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c01,c02,c03]
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vbroadcastsd  0*8(%[a]), %%ymm1 \n\t" // [a00,a00,a00,a00]
	                "vmovapd       0*8(%[b]), %%ymm0 \n\t" // [b00,b10,b20,b30]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c01,c02,c03]
	                "\n\t" // k=1
	                "vbroadcastsd  1*8(%[a]), %%ymm1 \n\t" // [a01,a01,a01,a01]
	                "vmovapd       4*8(%[b]), %%ymm0 \n\t" // [b01,b11,b21,b31]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c01,c02,c03]
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vbroadcastsd  0*8(%[a]), %%ymm1 \n\t" // [a00,a00,a00,a00]
	                "vmovapd       0*8(%[b]), %%ymm0 \n\t" // [b00,b10,b20,b30]
	                "vfmadd231pd   %%ymm0 , %%ymm1 , %%ymm4 \n\t" // [c00,c01,c02,c03]
	                "\n\t"
	                "addq  $1*8 , %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vmulpd  %%ymm0 , %%ymm4 , %%ymm4 \n\t" // [c00,c01,c02,c03]
	            "\n\t"
	            "vperm2f128  $0x01, %%ymm4 , %%ymm4 , %%ymm10\n\t" // [c02,c03,c00,c01]
	            "vshufpd     $0x05, %%ymm4 , %%ymm4 , %%ymm7 \n\t" // [c01,c00,c03,c02]
	            "vshufpd     $0x05, %%ymm10, %%ymm10, %%ymm13\n\t" // [c03,c02,c01,c00]
	            "\n\t"
	            "vfmadd213sd 0*8(%[c0]          ), %%xmm0, %%xmm4 \n\t"
	            "vfmadd213sd 0*8(%[c1]          ), %%xmm0, %%xmm7 \n\t"
	            "vfmadd213sd 0*8(%[c0],%[ldc2]  ), %%xmm0, %%xmm10\n\t"
	            "vfmadd213sd 0*8(%[c1],%[ldc2]  ), %%xmm0, %%xmm13\n\t"
	            "\n\t"
	            "vmovsd  %%xmm4 , 0*8(%[c0]          )\n\t"
	            "vmovsd  %%xmm7 , 0*8(%[c1]          )\n\t"
	            "vmovsd  %%xmm10, 0*8(%[c0],%[ldc2]  )\n\t"
	            "vmovsd  %%xmm13, 0*8(%[c1],%[ldc2]  )\n\t"
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
	    c0 = c0 - M + 4*ldc;
	    c1 = c1 - M + 4*ldc;

	  }

	}
/****************************************** Unmodified *****************************************************/
	if( N & 2 ){ 

	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        //c10=0e0;c11=0e0;c12=0e0;c13=0e0;
	        //c20=0e0;c21=0e0;c22=0e0;c23=0e0;
	        //c30=0e0;c31=0e0;c32=0e0;c33=0e0;

	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm12, %%ymm12, %%ymm12\n\t"
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);

	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*4 ); a01 = *(A +  0 + 1*4 ); a02 = *(A + 0 + 2*4 ); a03 = *(A + 0 + 3*4 );
	            //a10 = *(A + 1 + 0*4 ); a11 = *(A +  1 + 1*4 ); a12 = *(A + 1 + 2*4 ); a13 = *(A + 1 + 3*4 );
	            //a20 = *(A + 2 + 0*4 ); a21 = *(A +  2 + 1*4 ); a22 = *(A + 2 + 2*4 ); a23 = *(A + 2 + 3*4 );
	            //a30 = *(A + 3 + 0*4 ); a31 = *(A +  3 + 1*4 ); a32 = *(A + 3 + 2*4 ); a33 = *(A + 3 + 3*4 );
	            //b00 = *(B + 0 + 0*2 ); b01 = *(B +  0 + 1*2 ); b02 = *(B + 0 + 2*2 ); b03 = *(B + 0 + 3*2 );
	            //b10 = *(B + 1 + 0*2 ); b11 = *(B +  1 + 1*2 ); b12 = *(B + 1 + 2*2 ); b13 = *(B + 1 + 3*2 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c11 += a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //c20 += a20 * b00; c20 += a21 * b01; c20 += a22 * b02; c20 += a23 * b03; 
	            //c21 += a20 * b10; c21 += a21 * b11; c21 += a22 * b12; c21 += a23 * b13; 
	            //c30 += a30 * b00; c30 += a31 * b01; c30 += a32 * b02; c30 += a33 * b03; 
	            //c31 += a30 * b10; c31 += a31 * b11; c31 += a32 * b12; c31 += a33 * b13; 
	            //a00 = *(A + 0 + 4*4 ); a01 = *(A +  0 + 5*4 ); a02 = *(A + 0 + 6*4 ); a03 = *(A + 0 + 7*4 );
	            //a10 = *(A + 1 + 4*4 ); a11 = *(A +  1 + 5*4 ); a12 = *(A + 1 + 6*4 ); a13 = *(A + 1 + 7*4 );
	            //a20 = *(A + 2 + 4*4 ); a21 = *(A +  2 + 5*4 ); a22 = *(A + 2 + 6*4 ); a23 = *(A + 2 + 7*4 );
	            //a30 = *(A + 3 + 4*4 ); a31 = *(A +  3 + 5*4 ); a32 = *(A + 3 + 6*4 ); a33 = *(A + 3 + 7*4 );
	            //b00 = *(B + 0 + 4*2 ); b01 = *(B +  0 + 5*2 ); b02 = *(B + 0 + 6*2 ); b03 = *(B + 0 + 7*2 );
	            //b10 = *(B + 1 + 4*2 ); b11 = *(B +  1 + 5*2 ); b12 = *(B + 1 + 6*2 ); b13 = *(B + 1 + 7*2 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c11 += a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //c20 += a20 * b00; c20 += a21 * b01; c20 += a22 * b02; c20 += a23 * b03; 
	            //c21 += a20 * b10; c21 += a21 * b11; c21 += a22 * b12; c21 += a23 * b13; 
	            //c30 += a30 * b00; c30 += a31 * b01; c30 += a32 * b02; c30 += a33 * b03; 
	            //c31 += a30 * b10; c31 += a31 * b11; c31 += a32 * b12; c31 += a33 * b13; 
	            //A+=32;
	            //B+=16;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd        0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd        4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vmovapd        8*8(%[a]), %%ymm2 \n\t" // [a02,a12,a22,a32]
	                "vmovapd       12*8(%[a]), %%ymm3 \n\t" // [a03,a13,a23,a33]
	                "vbroadcastsd   0*8(%[b]), %%ymm4 \n\t" // [b00,b00,b00,b00]
	                "vbroadcastsd   1*8(%[b]), %%ymm5 \n\t" // [b10,b10,b10,b10]
	                "vbroadcastsd   2*8(%[b]), %%ymm6 \n\t" // [b01,b01,b01,b01]
	                "vbroadcastsd   3*8(%[b]), %%ymm7 \n\t" // [b11,b11,b11,b11]
	                "vbroadcastsd   4*8(%[b]), %%ymm8 \n\t" // [b02,b02,b02,b02]
	                "vbroadcastsd   5*8(%[b]), %%ymm9 \n\t" // [b12,b12,b12,b12]
	                "vbroadcastsd   6*8(%[b]), %%ymm10\n\t" // [b03,b03,b03,b03]
	                "vbroadcastsd   7*8(%[b]), %%ymm11\n\t" // [b13,b13,b13,b13]
	                "\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm5 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm6 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm7 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm8 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm9 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm10, %%ymm14\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm11, %%ymm15\n\t"
	                "\n\t"
	                "vmovapd       16*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd       20*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vmovapd       24*8(%[a]), %%ymm2 \n\t" // [a02,a12,a22,a32]
	                "vmovapd       28*8(%[a]), %%ymm3 \n\t" // [a03,a13,a23,a33]
	                "vbroadcastsd   8*8(%[b]), %%ymm4 \n\t" // [b00,b00,b00,b00]
	                "vbroadcastsd   9*8(%[b]), %%ymm5 \n\t" // [b10,b10,b10,b10]
	                "vbroadcastsd  10*8(%[b]), %%ymm6 \n\t" // [b01,b01,b01,b01]
	                "vbroadcastsd  11*8(%[b]), %%ymm7 \n\t" // [b11,b11,b11,b11]
	                "vbroadcastsd  12*8(%[b]), %%ymm8 \n\t" // [b02,b02,b02,b02]
	                "vbroadcastsd  13*8(%[b]), %%ymm9 \n\t" // [b12,b12,b12,b12]
	                "vbroadcastsd  14*8(%[b]), %%ymm10\n\t" // [b03,b03,b03,b03]
	                "vbroadcastsd  15*8(%[b]), %%ymm11\n\t" // [b13,b13,b13,b13]
	                "\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm5 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm6 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm7 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm8 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm9 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm10, %%ymm14\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm11, %%ymm15\n\t"
	                "\n\t"
	                "addq  $32*8, %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	        }
	        if( K & 4 ){
	            //a00 = *(A + 0 + 0*4 ); a01 = *(A +  0 + 1*4 ); a02 = *(A + 0 + 2*4 ); a03 = *(A + 0 + 3*4 );
	            //a10 = *(A + 1 + 0*4 ); a11 = *(A +  1 + 1*4 ); a12 = *(A + 1 + 2*4 ); a13 = *(A + 1 + 3*4 );
	            //a20 = *(A + 2 + 0*4 ); a21 = *(A +  2 + 1*4 ); a22 = *(A + 2 + 2*4 ); a23 = *(A + 2 + 3*4 );
	            //a30 = *(A + 3 + 0*4 ); a31 = *(A +  3 + 1*4 ); a32 = *(A + 3 + 2*4 ); a33 = *(A + 3 + 3*4 );
	            //b00 = *(B + 0 + 0*2 ); b01 = *(B +  0 + 1*2 ); b02 = *(B + 0 + 2*2 ); b03 = *(B + 0 + 3*2 );
	            //b10 = *(B + 1 + 0*2 ); b11 = *(B +  1 + 1*2 ); b12 = *(B + 1 + 2*2 ); b13 = *(B + 1 + 3*2 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c11 += a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //c20 += a20 * b00; c20 += a21 * b01; c20 += a22 * b02; c20 += a23 * b03; 
	            //c21 += a20 * b10; c21 += a21 * b11; c21 += a22 * b12; c21 += a23 * b13; 
	            //c30 += a30 * b00; c30 += a31 * b01; c30 += a32 * b02; c30 += a33 * b03; 
	            //c31 += a30 * b10; c31 += a31 * b11; c31 += a32 * b12; c31 += a33 * b13; 
	            //A+=16;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd        0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd        4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vmovapd        8*8(%[a]), %%ymm2 \n\t" // [a02,a12,a22,a32]
	                "vmovapd       12*8(%[a]), %%ymm3 \n\t" // [a03,a13,a23,a33]
	                "vbroadcastsd   0*8(%[b]), %%ymm4 \n\t" // [b00,b00,b00,b00]
	                "vbroadcastsd   1*8(%[b]), %%ymm5 \n\t" // [b10,b10,b10,b10]
	                "vbroadcastsd   2*8(%[b]), %%ymm6 \n\t" // [b01,b01,b01,b01]
	                "vbroadcastsd   3*8(%[b]), %%ymm7 \n\t" // [b11,b11,b11,b11]
	                "vbroadcastsd   4*8(%[b]), %%ymm8 \n\t" // [b02,b02,b02,b02]
	                "vbroadcastsd   5*8(%[b]), %%ymm9 \n\t" // [b12,b12,b12,b12]
	                "vbroadcastsd   6*8(%[b]), %%ymm10\n\t" // [b03,b03,b03,b03]
	                "vbroadcastsd   7*8(%[b]), %%ymm11\n\t" // [b13,b13,b13,b13]
	                "\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm5 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm6 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm7 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm8 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm9 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm10, %%ymm14\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm11, %%ymm15\n\t"
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 2 ){

	            //a00 = *(A + 0 + 0*4 ); a01 = *(A +  0 + 1*4 );
	            //a10 = *(A + 1 + 0*4 ); a11 = *(A +  1 + 1*4 );
	            //a20 = *(A + 2 + 0*4 ); a21 = *(A +  2 + 1*4 );
	            //a30 = *(A + 3 + 0*4 ); a31 = *(A +  3 + 1*4 );
	            //b00 = *(B + 0 + 0*2 ); b01 = *(B +  0 + 1*2 );
	            //b10 = *(B + 1 + 0*2 ); b11 = *(B +  1 + 1*2 );
	            //c00 += a00 * b00; c00 += a01 * b01;
	            //c10 += a10 * b00; c10 += a11 * b01;
	            //c20 += a20 * b00; c20 += a21 * b01;
	            //c30 += a30 * b00; c30 += a31 * b01;
	            //c01 += a00 * b10; c01 += a01 * b11;
	            //c11 += a10 * b10; c11 += a11 * b11;
	            //c21 += a20 * b10; c21 += a21 * b11;
	            //c31 += a30 * b10; c31 += a31 * b11;
	            //A+=8;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd        0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd        4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vbroadcastsd   0*8(%[b]), %%ymm4 \n\t" // [b00,b00,b00,b00]
	                "vbroadcastsd   1*8(%[b]), %%ymm5 \n\t" // [b10,b10,b10,b10]
	                "vbroadcastsd   2*8(%[b]), %%ymm6 \n\t" // [b01,b01,b01,b01]
	                "vbroadcastsd   3*8(%[b]), %%ymm7 \n\t" // [b11,b11,b11,b11]
	                "\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm5 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm6 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm7 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){
	            //a00 = *(A + 0 + 0*4 );
	            //a10 = *(A + 1 + 0*4 );
	            //a20 = *(A + 2 + 0*4 );
	            //a30 = *(A + 3 + 0*4 );
	            //b00 = *(B + 0 + 0*2 );
	            //b10 = *(B + 1 + 0*2 );
	            //c00 += a00 * b00;
	            //c01 += a00 * b10;
	            //c10 += a10 * b00;
	            //c11 += a10 * b10;
	            //c20 += a20 * b00;
	            //c21 += a20 * b10;
	            //c30 += a30 * b00;
	            //c31 += a30 * b10;
	            //A+=4;
	            //B+=2;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd        0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vbroadcastsd   0*8(%[b]), %%ymm4 \n\t" // [b00,b00,b00,b00]
	                "vbroadcastsd   1*8(%[b]), %%ymm5 \n\t" // [b10,b10,b10,b10]
	                "\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm5 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
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

	        __asm__ __volatile__ (
	            "\n\t"
	            //"vbroadcastsd %[alpha], %%ymm0 \n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]), %%ymm0, %%ymm14\n\t"
	            "vfmadd213pd 0*8(%[c1]), %%ymm0, %%ymm15\n\t"
	            "\n\t"
	            "vmovupd  %%ymm14, 0*8(%[c0])\n\t"
	            "vmovupd  %%ymm15, 0*8(%[c1])\n\t"
	            "\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "addq  $4*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4)
	        );

	        B = B - 2*K;//N*K
	        //C+=4;
	        //c0+=4;
	        //c1+=4;
	      }

	    }
	    if( M & 2 ){

	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        //c10=0e0;c11=0e0;c12=0e0;c13=0e0;
	        //c20=0e0;c21=0e0;c22=0e0;c23=0e0;
	        //c30=0e0;c31=0e0;c32=0e0;c33=0e0;
	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm12, %%ymm12, %%ymm12\n\t"
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*2 ); a01 = *(A +  0 + 1*2 ); a02 = *(A + 0 + 2*2 ); a03 = *(A + 0 + 3*2 );
	            //a10 = *(A + 1 + 0*2 ); a11 = *(A +  1 + 1*2 ); a12 = *(A + 1 + 2*2 ); a13 = *(A + 1 + 3*2 );
	            //b00 = *(B + 0 + 0*2 ); b01 = *(B +  0 + 1*2 ); b02 = *(B + 0 + 2*2 ); b03 = *(B + 0 + 3*2 );
	            //b10 = *(B + 1 + 0*2 ); b11 = *(B +  1 + 1*2 ); b12 = *(B + 1 + 2*2 ); b13 = *(B + 1 + 3*2 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c11 += a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //a00 = *(A + 0 + 4*2 ); a01 = *(A +  0 + 5*2 ); a02 = *(A + 0 + 6*2 ); a03 = *(A + 0 + 7*2 );
	            //a10 = *(A + 1 + 4*2 ); a11 = *(A +  1 + 5*2 ); a12 = *(A + 1 + 6*2 ); a13 = *(A + 1 + 7*2 );
	            //b00 = *(B + 0 + 4*2 ); b01 = *(B +  0 + 5*2 ); b02 = *(B + 0 + 6*2 ); b03 = *(B + 0 + 7*2 );
	            //b10 = *(B + 1 + 4*2 ); b11 = *(B +  1 + 5*2 ); b12 = *(B + 1 + 6*2 ); b13 = *(B + 1 + 7*2 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c11 += a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //A+=16;
	            //B+=16;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10]
	                "movapd   2*8(%[a]), %%xmm1 \n\t" // [a01,a11]
	                "movapd   4*8(%[a]), %%xmm2 \n\t" // [a02,a12]
	                "movapd   6*8(%[a]), %%xmm3 \n\t" // [a03,a13]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "movapd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11]
	                "movapd   4*8(%[b]), %%xmm6 \n\t" // [b02,b12]
	                "movapd   6*8(%[b]), %%xmm7 \n\t" // [b03,b13]
	                "\n\t"
	                "vshufpd   $0x00, %%xmm4 , %%xmm4 , %%xmm8 \n\t" // [b00,b00]
	                "vshufpd   $0x03, %%xmm4 , %%xmm4 , %%xmm4 \n\t" // [b10,b10]
	                "vshufpd   $0x00, %%xmm5 , %%xmm5 , %%xmm9 \n\t" // [b01,b01]
	                "vshufpd   $0x03, %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b11,b11]
	                "vfmadd231pd %%xmm0 , %%xmm8 , %%xmm14\n\t"
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm9 , %%xmm14\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm5 , %%xmm15\n\t"
	                "vshufpd   $0x00, %%xmm6 , %%xmm6 , %%xmm10\n\t" // [b02,b02]
	                "vshufpd   $0x03, %%xmm6 , %%xmm6 , %%xmm6 \n\t" // [b12,b12]
	                "vshufpd   $0x00, %%xmm7 , %%xmm7 , %%xmm11\n\t" // [b03,b03]
	                "vshufpd   $0x03, %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b13,b13]
	                "vfmadd231pd %%xmm2 , %%xmm10, %%xmm14\n\t"
	                "vfmadd231pd %%xmm2 , %%xmm6 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm3 , %%xmm11, %%xmm14\n\t"
	                "vfmadd231pd %%xmm3 , %%xmm7 , %%xmm15\n\t"
	                "\n\t"
	                "movapd   8*8(%[a]), %%xmm0 \n\t" // [a04,a14]
	                "movapd  10*8(%[a]), %%xmm1 \n\t" // [a05,a15]
	                "movapd  12*8(%[a]), %%xmm2 \n\t" // [a06,a16]
	                "movapd  14*8(%[a]), %%xmm3 \n\t" // [a07,a17]
	                "movapd   8*8(%[b]), %%xmm4 \n\t" // [b04,b14]
	                "movapd  10*8(%[b]), %%xmm5 \n\t" // [b05,b15]
	                "movapd  12*8(%[b]), %%xmm6 \n\t" // [b06,b16]
	                "movapd  14*8(%[b]), %%xmm7 \n\t" // [b07,b17]
	                "\n\t"
	                "vshufpd   $0x00, %%xmm4 , %%xmm4 , %%xmm8 \n\t" // [b04,b04]
	                "vshufpd   $0x03, %%xmm4 , %%xmm4 , %%xmm4 \n\t" // [b14,b14]
	                "vshufpd   $0x00, %%xmm5 , %%xmm5 , %%xmm9 \n\t" // [b05,b05]
	                "vshufpd   $0x03, %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b15,b15]
	                "vfmadd231pd %%xmm0 , %%xmm8 , %%xmm14\n\t"
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm9 , %%xmm14\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm5 , %%xmm15\n\t"
	                "vshufpd   $0x00, %%xmm6 , %%xmm6 , %%xmm10\n\t" // [b06,b06]
	                "vshufpd   $0x03, %%xmm6 , %%xmm6 , %%xmm6 \n\t" // [b16,b16]
	                "vshufpd   $0x00, %%xmm7 , %%xmm7 , %%xmm11\n\t" // [b07,b07]
	                "vshufpd   $0x03, %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b17,b17]
	                "vfmadd231pd %%xmm2 , %%xmm10, %%xmm14\n\t"
	                "vfmadd231pd %%xmm2 , %%xmm6 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm3 , %%xmm11, %%xmm14\n\t"
	                "vfmadd231pd %%xmm3 , %%xmm7 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	        }
	        if( K & 4 ){
	            //a00 = *(A + 0 + 0*2 ); a01 = *(A +  0 + 1*2 ); a02 = *(A + 0 + 2*2 ); a03 = *(A + 0 + 3*2 );
	            //a10 = *(A + 1 + 0*2 ); a11 = *(A +  1 + 1*2 ); a12 = *(A + 1 + 2*2 ); a13 = *(A + 1 + 3*2 );
	            //b00 = *(B + 0 + 0*2 ); b01 = *(B +  0 + 1*2 ); b02 = *(B + 0 + 2*2 ); b03 = *(B + 0 + 3*2 );
	            //b10 = *(B + 1 + 0*2 ); b11 = *(B +  1 + 1*2 ); b12 = *(B + 1 + 2*2 ); b13 = *(B + 1 + 3*2 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c11 += a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //A+=8;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10]
	                "movapd   2*8(%[a]), %%xmm1 \n\t" // [a01,a11]
	                "movapd   4*8(%[a]), %%xmm2 \n\t" // [a02,a12]
	                "movapd   6*8(%[a]), %%xmm3 \n\t" // [a03,a13]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "movapd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11]
	                "movapd   4*8(%[b]), %%xmm6 \n\t" // [b02,b12]
	                "movapd   6*8(%[b]), %%xmm7 \n\t" // [b03,b13]
	                "\n\t"
	                "vshufpd   $0x00, %%xmm4 , %%xmm4 , %%xmm8 \n\t" // [b00,b00]
	                "vshufpd   $0x03, %%xmm4 , %%xmm4 , %%xmm4 \n\t" // [b10,b10]
	                "vshufpd   $0x00, %%xmm5 , %%xmm5 , %%xmm9 \n\t" // [b01,b01]
	                "vshufpd   $0x03, %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b11,b11]
	                "vfmadd231pd %%xmm0 , %%xmm8 , %%xmm14\n\t"
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm9 , %%xmm14\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm5 , %%xmm15\n\t"
	                "vshufpd   $0x00, %%xmm6 , %%xmm6 , %%xmm10\n\t" // [b02,b02]
	                "vshufpd   $0x03, %%xmm6 , %%xmm6 , %%xmm6 \n\t" // [b12,b12]
	                "vshufpd   $0x00, %%xmm7 , %%xmm7 , %%xmm11\n\t" // [b03,b03]
	                "vshufpd   $0x03, %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b13,b13]
	                "vfmadd231pd %%xmm2 , %%xmm10, %%xmm14\n\t"
	                "vfmadd231pd %%xmm2 , %%xmm6 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm3 , %%xmm11, %%xmm14\n\t"
	                "vfmadd231pd %%xmm3 , %%xmm7 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $8*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 2 ){
	            //a00 = *(A + 0 + 0*2 ); a01 = *(A +  0 + 1*2 );
	            //a10 = *(A + 1 + 0*2 ); a11 = *(A +  1 + 1*2 );
	            //b00 = *(B + 0 + 0*2 ); b01 = *(B +  0 + 1*2 );
	            //b10 = *(B + 1 + 0*2 ); b11 = *(B +  1 + 1*2 );
	            //c00 += a00 * b00; c00 += a01 * b01;
	            //c01 += a00 * b10; c01 += a01 * b11;
	            //c10 += a10 * b00; c10 += a11 * b01;
	            //c11 += a10 * b10; c11 += a11 * b11;
	            //A+=4;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10]
	                "movapd   2*8(%[a]), %%xmm1 \n\t" // [a01,a11]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "movapd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11]
	                "\n\t"
	                "vshufpd   $0x00, %%xmm4 , %%xmm4 , %%xmm8 \n\t" // [b00,b00]
	                "vshufpd   $0x03, %%xmm4 , %%xmm4 , %%xmm4 \n\t" // [b10,b10]
	                "vshufpd   $0x00, %%xmm5 , %%xmm5 , %%xmm9 \n\t" // [b01,b01]
	                "vshufpd   $0x03, %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b11,b11]
	                "vfmadd231pd %%xmm0 , %%xmm8 , %%xmm14\n\t"
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm9 , %%xmm14\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm5 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){
	            //a00 = *(A + 0 + 0*2 );
	            //a10 = *(A + 1 + 0*2 );
	            //b00 = *(B + 0 + 0*2 );
	            //b10 = *(B + 1 + 0*2 );
	            //c00 += a00 * b00;
	            //c01 += a00 * b10;
	            //c10 += a10 * b00;
	            //c11 += a10 * b10;
	            //A+=2;
	            //B+=2;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "\n\t"
	                "vshufpd   $0x00, %%xmm4 , %%xmm4 , %%xmm8 \n\t" // [b00,b00]
	                "vshufpd   $0x03, %%xmm4 , %%xmm4 , %%xmm4 \n\t" // [b10,b10]
	                "vfmadd231pd %%xmm0 , %%xmm8 , %%xmm14\n\t"
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $2*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+0+1*ldc) += alpha*c01;
	        //*(C+1+0*ldc) += alpha*c10;
	        //*(C+1+1*ldc) += alpha*c11;

	        __asm__ __volatile__ (
	            "\n\t"
	            //"vbroadcastsd %[alpha], %%ymm0 \n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]), %%xmm0, %%xmm14\n\t"
	            "vfmadd213pd 0*8(%[c1]), %%xmm0, %%xmm15\n\t"
	            "\n\t"
	            "vmovupd  %%xmm14, 0*8(%[c0])\n\t"
	            "vmovupd  %%xmm15, 0*8(%[c1])\n\t"
	            "\n\t"
	            "addq  $2*8, %[c0]\n\t"
	            "addq  $2*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4)
	        );


	        B = B - 2*K;//N*K
	        //C+=2;
	        //c0+=2;
	        //c1+=2;

	    }
	    if( M & 1 ){

	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        //c10=0e0;c11=0e0;c12=0e0;c13=0e0;
	        //c20=0e0;c21=0e0;c22=0e0;c23=0e0;
	        //c30=0e0;c31=0e0;c32=0e0;c33=0e0;
	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm12, %%ymm12, %%ymm12\n\t"
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*1 ); a01 = *(A +  0 + 1*1 ); a02 = *(A + 0 + 2*1 ); a03 = *(A + 0 + 3*1 );
	            //b00 = *(B + 0 + 0*2 ); b01 = *(B +  0 + 1*2 ); b02 = *(B + 0 + 2*2 ); b03 = *(B + 0 + 3*2 );
	            //b10 = *(B + 1 + 0*2 ); b11 = *(B +  1 + 1*2 ); b12 = *(B + 1 + 2*2 ); b13 = *(B + 1 + 3*2 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //a00 = *(A + 0 + 4*1 ); a01 = *(A +  0 + 5*1 ); a02 = *(A + 0 + 6*1 ); a03 = *(A + 0 + 7*1 );
	            //b00 = *(B + 0 + 4*2 ); b01 = *(B +  0 + 5*2 ); b02 = *(B + 0 + 6*2 ); b03 = *(B + 0 + 7*2 );
	            //b10 = *(B + 1 + 4*2 ); b11 = *(B +  1 + 5*2 ); b12 = *(B + 1 + 6*2 ); b13 = *(B + 1 + 7*2 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //A+=8;
	            //B+=16;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a01]
	                "movapd   2*8(%[a]), %%xmm2 \n\t" // [a02,a03]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "movapd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11]
	                "movapd   4*8(%[b]), %%xmm6 \n\t" // [b02,b12]
	                "movapd   6*8(%[b]), %%xmm7 \n\t" // [b03,b13]
	                "\n\t"
	                "vshufpd   $0x00, %%xmm0 , %%xmm0 , %%xmm8 \n\t" // [a00,a00]
	                "vshufpd   $0x03, %%xmm0 , %%xmm0 , %%xmm9 \n\t" // [a01,a01]
	                "vshufpd   $0x00, %%xmm2 , %%xmm2 , %%xmm10\n\t" // [a02,a02]
	                "vshufpd   $0x03, %%xmm2 , %%xmm2 , %%xmm11\n\t" // [a03,a03]
	                "vfmadd231pd %%xmm8  , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm9  , %%xmm5 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm10 , %%xmm6 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm11 , %%xmm7 , %%xmm15\n\t"
	                "\n\t"
	                "movapd   4*8(%[a]), %%xmm0 \n\t" // [a00,a01]
	                "movapd   6*8(%[a]), %%xmm2 \n\t" // [a02,a03]
	                "movapd   8*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "movapd  10*8(%[b]), %%xmm5 \n\t" // [b01,b11]
	                "movapd  12*8(%[b]), %%xmm6 \n\t" // [b02,b12]
	                "movapd  14*8(%[b]), %%xmm7 \n\t" // [b03,b13]
	                "\n\t"
	                "vshufpd   $0x00, %%xmm0 , %%xmm0 , %%xmm8 \n\t" // [a00,a00]
	                "vshufpd   $0x03, %%xmm0 , %%xmm0 , %%xmm9 \n\t" // [a01,a01]
	                "vshufpd   $0x00, %%xmm2 , %%xmm2 , %%xmm10\n\t" // [a02,a02]
	                "vshufpd   $0x03, %%xmm2 , %%xmm2 , %%xmm11\n\t" // [a03,a03]
	                "vfmadd231pd %%xmm8  , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm9  , %%xmm5 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm10 , %%xmm6 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm11 , %%xmm7 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	        }
	        if( K & 4 ){
	            //a00 = *(A + 0 + 0*1 ); a01 = *(A +  0 + 1*1 ); a02 = *(A + 0 + 2*1 ); a03 = *(A + 0 + 3*1 );
	            //b00 = *(B + 0 + 0*2 ); b01 = *(B +  0 + 1*2 ); b02 = *(B + 0 + 2*2 ); b03 = *(B + 0 + 3*2 );
	            //b10 = *(B + 1 + 0*2 ); b11 = *(B +  1 + 1*2 ); b12 = *(B + 1 + 2*2 ); b13 = *(B + 1 + 3*2 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //A+=4;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a01]
	                "movapd   2*8(%[a]), %%xmm2 \n\t" // [a02,a03]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "movapd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11]
	                "movapd   4*8(%[b]), %%xmm6 \n\t" // [b02,b12]
	                "movapd   6*8(%[b]), %%xmm7 \n\t" // [b03,b13]
	                "\n\t"
	                "vshufpd   $0x03, %%xmm0 , %%xmm0 , %%xmm1 \n\t" // [a01,a01]
	                "vshufpd   $0x00, %%xmm0 , %%xmm0 , %%xmm0 \n\t" // [a00,a00]
	                "vshufpd   $0x03, %%xmm2 , %%xmm2 , %%xmm3 \n\t" // [a03,a03]
	                "vshufpd   $0x00, %%xmm2 , %%xmm2 , %%xmm2 \n\t" // [a02,a02]
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm5 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm2 , %%xmm6 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm3 , %%xmm7 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $8*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 2 ){
	            //a00 = *(A + 0 + 0*1 ); a01 = *(A +  0 + 1*1 );
	            //b00 = *(B + 0 + 0*2 ); b01 = *(B +  0 + 1*2 );
	            //b10 = *(B + 1 + 0*2 ); b11 = *(B +  1 + 1*2 );
	            //c00 += a00 * b00; c00 += a01 * b01;
	            //c01 += a00 * b10; c01 += a01 * b11;
	            //A+=2;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a01]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "movapd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11]
	                "\n\t"
	                "vshufpd   $0x03, %%xmm0 , %%xmm0 , %%xmm1 \n\t" // [a01,a01]
	                "vshufpd   $0x00, %%xmm0 , %%xmm0 , %%xmm0 \n\t" // [a00,a00]
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm5 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $2*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){
	            //a00 = *(A + 0 + 0*1 );
	            //b00 = *(B + 0 + 0*2 );
	            //b10 = *(B + 1 + 0*2 );
	            //c00 += a00 * b00;
	            //c01 += a00 * b10;
	            //A+=1;
	            //B+=2;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movsd    0*8(%[a]), %%xmm0 \n\t" // [a00]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "\n\t"
	                "vshufpd   $0x00, %%xmm0 , %%xmm0 , %%xmm0 \n\t" // [a00,a00]
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $1*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+0+1*ldc) += alpha*c01;

	        __asm__ __volatile__ (
	            "\n\t"
	            "movupd %[alpha], %%xmm0\n\t"
	            "\n\t"
	            "mulpd   %%xmm0 , %%xmm15\n\t"
	            "movapd  %%xmm15, %%xmm14\n\t"
	            "shufpd  $0x01, %%xmm15, %%xmm14\n\t"
	            "addsd   0*8(%[c0]), %%xmm15\n\t"
	            "addsd   0*8(%[c1]), %%xmm14\n\t"
	            "movlpd  %%xmm15, 0*8(%[c0])\n\t"
	            "movlpd  %%xmm14, 0*8(%[c1])\n\t"
	            "\n\t"
	            "addq  $1*8, %[c0]\n\t"
	            "addq  $1*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 2*K;//N*K
	        //C+=1;
	        //c0+=1;
	        //c1+=1;

	    }
	    A = A - M*K;
	    B = B + 2*K;
	    //C  = C - M + 2*ldc;
	    c0 = c0- M + 2*ldc;
	    c1 = c1- M + 2*ldc;

	}
	if( N & 1 ){

	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        //c10=0e0;c11=0e0;c12=0e0;c13=0e0;
	        //c20=0e0;c21=0e0;c22=0e0;c23=0e0;
	        //c30=0e0;c31=0e0;c32=0e0;c33=0e0;
	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*4 ); a01 = *(A +  0 + 1*4 ); a02 = *(A + 0 + 2*4 ); a03 = *(A + 0 + 3*4 );
	            //a10 = *(A + 1 + 0*4 ); a11 = *(A +  1 + 1*4 ); a12 = *(A + 1 + 2*4 ); a13 = *(A + 1 + 3*4 );
	            //a20 = *(A + 2 + 0*4 ); a21 = *(A +  2 + 1*4 ); a22 = *(A + 2 + 2*4 ); a23 = *(A + 2 + 3*4 );
	            //a30 = *(A + 3 + 0*4 ); a31 = *(A +  3 + 1*4 ); a32 = *(A + 3 + 2*4 ); a33 = *(A + 3 + 3*4 );
	            //b00 = *(B + 0 + 0*1 ); b01 = *(B +  0 + 1*1 ); b02 = *(B + 0 + 2*1 ); b03 = *(B + 0 + 3*1 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c20 += a20 * b00; c20 += a21 * b01; c20 += a22 * b02; c20 += a23 * b03; 
	            //c30 += a30 * b00; c30 += a31 * b01; c30 += a32 * b02; c30 += a33 * b03; 
	            //a00 = *(A + 0 + 4*4 ); a01 = *(A +  0 + 5*4 ); a02 = *(A + 0 + 6*4 ); a03 = *(A + 0 + 7*4 );
	            //a10 = *(A + 1 + 4*4 ); a11 = *(A +  1 + 5*4 ); a12 = *(A + 1 + 6*4 ); a13 = *(A + 1 + 7*4 );
	            //a20 = *(A + 2 + 4*4 ); a21 = *(A +  2 + 5*4 ); a22 = *(A + 2 + 6*4 ); a23 = *(A + 2 + 7*4 );
	            //a30 = *(A + 3 + 4*4 ); a31 = *(A +  3 + 5*4 ); a32 = *(A + 3 + 6*4 ); a33 = *(A + 3 + 7*4 );
	            //b00 = *(B + 0 + 4*1 ); b01 = *(B +  0 + 5*1 ); b02 = *(B + 0 + 6*1 ); b03 = *(B + 0 + 7*1 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c20 += a20 * b00; c20 += a21 * b01; c20 += a22 * b02; c20 += a23 * b03; 
	            //c30 += a30 * b00; c30 += a31 * b01; c30 += a32 * b02; c30 += a33 * b03; 
	            //A+=32;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd        0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd        4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vmovapd        8*8(%[a]), %%ymm2 \n\t" // [a02,a12,a22,a32]
	                "vmovapd       12*8(%[a]), %%ymm3 \n\t" // [a03,a13,a23,a33]
	                "vbroadcastsd   0*8(%[b]), %%ymm4 \n\t" // [b00,b00,b00,b00]
	                "vbroadcastsd   1*8(%[b]), %%ymm5 \n\t" // [b01,b01,b01,b01]
	                "vbroadcastsd   2*8(%[b]), %%ymm6 \n\t" // [b02,b02,b02,b02]
	                "vbroadcastsd   3*8(%[b]), %%ymm7 \n\t" // [b03,b03,b03,b03]
	                "\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm5 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm6 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm7 , %%ymm15\n\t"
	                "\n\t"
	                "vmovapd       16*8(%[a]), %%ymm8 \n\t" // [a00,a10,a20,a30]
	                "vmovapd       20*8(%[a]), %%ymm9 \n\t" // [a01,a11,a21,a31]
	                "vmovapd       24*8(%[a]), %%ymm2 \n\t" // [a02,a12,a22,a32]
	                "vmovapd       28*8(%[a]), %%ymm3 \n\t" // [a03,a13,a23,a33]
	                "vbroadcastsd   4*8(%[b]), %%ymm10\n\t" // [b00,b00,b00,b00]
	                "vbroadcastsd   5*8(%[b]), %%ymm11\n\t" // [b01,b01,b01,b01]
	                "vbroadcastsd   6*8(%[b]), %%ymm12\n\t" // [b02,b02,b02,b02]
	                "vbroadcastsd   7*8(%[b]), %%ymm13\n\t" // [b03,b03,b03,b03]
	                "\n\t"
	                "vfmadd231pd %%ymm8 , %%ymm10, %%ymm15\n\t"
	                "vfmadd231pd %%ymm9 , %%ymm11, %%ymm15\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm12, %%ymm15\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm13, %%ymm15\n\t"
	                "\n\t"
	                "addq  $32*8, %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          }
	        }
	        if( K & 4 ){
	            //a00 = *(A + 0 + 0*4 ); a01 = *(A +  0 + 1*4 ); a02 = *(A + 0 + 2*4 ); a03 = *(A + 0 + 3*4 );
	            //a10 = *(A + 1 + 0*4 ); a11 = *(A +  1 + 1*4 ); a12 = *(A + 1 + 2*4 ); a13 = *(A + 1 + 3*4 );
	            //a20 = *(A + 2 + 0*4 ); a21 = *(A +  2 + 1*4 ); a22 = *(A + 2 + 2*4 ); a23 = *(A + 2 + 3*4 );
	            //a30 = *(A + 3 + 0*4 ); a31 = *(A +  3 + 1*4 ); a32 = *(A + 3 + 2*4 ); a33 = *(A + 3 + 3*4 );
	            //b00 = *(B + 0 + 0*1 ); b01 = *(B +  0 + 1*1 ); b02 = *(B + 0 + 2*1 ); b03 = *(B + 0 + 3*1 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c20 += a20 * b00; c20 += a21 * b01; c20 += a22 * b02; c20 += a23 * b03; 
	            //c30 += a30 * b00; c30 += a31 * b01; c30 += a32 * b02; c30 += a33 * b03; 
	            //A+=16;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd        0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd        4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vmovapd        8*8(%[a]), %%ymm2 \n\t" // [a02,a12,a22,a32]
	                "vmovapd       12*8(%[a]), %%ymm3 \n\t" // [a03,a13,a23,a33]
	                "vbroadcastsd   0*8(%[b]), %%ymm4 \n\t" // [b00,b00,b00,b00]
	                "vbroadcastsd   1*8(%[b]), %%ymm5 \n\t" // [b01,b01,b01,b01]
	                "vbroadcastsd   2*8(%[b]), %%ymm6 \n\t" // [b02,b02,b02,b02]
	                "vbroadcastsd   3*8(%[b]), %%ymm7 \n\t" // [b03,b03,b03,b03]
	                "\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm5 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm6 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm7 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 2 ){
	            //a00 = *(A + 0 + 0*4 ); a01 = *(A +  0 + 1*4 );
	            //a10 = *(A + 1 + 0*4 ); a11 = *(A +  1 + 1*4 );
	            //a20 = *(A + 2 + 0*4 ); a21 = *(A +  2 + 1*4 );
	            //a30 = *(A + 3 + 0*4 ); a31 = *(A +  3 + 1*4 );
	            //b00 = *(B + 0 + 0*1 ); b01 = *(B +  0 + 1*1 );
	            //c00 += a00 * b00; c00 += a01 * b01;
	            //c10 += a10 * b00; c10 += a11 * b01;
	            //c20 += a20 * b00; c20 += a21 * b01;
	            //c30 += a30 * b00; c30 += a31 * b01;
	            //A+=8;
	            //B+=2;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd        0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd        4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vbroadcastsd   0*8(%[b]), %%ymm4 \n\t" // [b00,b00,b00,b00]
	                "vbroadcastsd   1*8(%[b]), %%ymm5 \n\t" // [b01,b01,b01,b01]
	                "\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm5 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){
	            //a00 = *(A + 0 + 0*4 );
	            //a10 = *(A + 1 + 0*4 );
	            //a20 = *(A + 2 + 0*4 );
	            //a30 = *(A + 3 + 0*4 );
	            //b00 = *(B + 0 + 0*1 );
	            //c00 += a00 * b00;
	            //c10 += a10 * b00;
	            //c20 += a20 * b00;
	            //c30 += a30 * b00;
	            //A+=4;
	            //B+=1;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd        0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vbroadcastsd   0*8(%[b]), %%ymm4 \n\t" // [b00,b00,b00,b00]
	                "\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $1*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+1+0*ldc) += alpha*c10;
	        //*(C+2+0*ldc) += alpha*c20;
	        //*(C+3+0*ldc) += alpha*c30;

	        __asm__ __volatile__ (
	            "\n\t"
	            //"vbroadcastsd %[alpha], %%ymm0 \n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "vfmadd213pd 0*8(%[c0]), %%ymm0, %%ymm15\n\t"
	            "vmovupd  %%ymm15, 0*8(%[c0])\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "addq  $4*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4)
	        );

	        B = B - 1*K;//N*K
	        //C+=4;
	        //c0+=4;
	        //c1+=4;
	      }

	    }
	    if( M & 2 ){

	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        //c10=0e0;c11=0e0;c12=0e0;c13=0e0;
	        //c20=0e0;c21=0e0;c22=0e0;c23=0e0;
	        //c30=0e0;c31=0e0;c32=0e0;c33=0e0;
	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*2 ); a01 = *(A +  0 + 1*2 ); a02 = *(A + 0 + 2*2 ); a03 = *(A + 0 + 3*2 );
	            //a10 = *(A + 1 + 0*2 ); a11 = *(A +  1 + 1*2 ); a12 = *(A + 1 + 2*2 ); a13 = *(A + 1 + 3*2 );
	            //b00 = *(B + 0 + 0*1 ); b01 = *(B +  0 + 1*1 ); b02 = *(B + 0 + 2*1 ); b03 = *(B + 0 + 3*1 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //a00 = *(A + 0 + 4*2 ); a01 = *(A +  0 + 5*2 ); a02 = *(A + 0 + 6*2 ); a03 = *(A + 0 + 7*2 );
	            //a10 = *(A + 1 + 4*2 ); a11 = *(A +  1 + 5*2 ); a12 = *(A + 1 + 6*2 ); a13 = *(A + 1 + 7*2 );
	            //b00 = *(B + 0 + 4*1 ); b01 = *(B +  0 + 5*1 ); b02 = *(B + 0 + 6*1 ); b03 = *(B + 0 + 7*1 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //A+=16;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10]
	                "movapd   2*8(%[a]), %%xmm1 \n\t" // [a01,a11]
	                "movapd   4*8(%[a]), %%xmm2 \n\t" // [a02,a12]
	                "movapd   6*8(%[a]), %%xmm3 \n\t" // [a03,a13]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b01]
	                "movapd   2*8(%[b]), %%xmm6 \n\t" // [b02,b03]
	                "\n\t"
	                "vshufpd   $0x03, %%xmm4 , %%xmm4 , %%xmm5 \n\t" // [b01,b01]
	                "vshufpd   $0x00, %%xmm4 , %%xmm4 , %%xmm4 \n\t" // [b00,b00]
	                "vshufpd   $0x03, %%xmm6 , %%xmm6 , %%xmm7 \n\t" // [b03,b03]
	                "vshufpd   $0x00, %%xmm6 , %%xmm6 , %%xmm6 \n\t" // [b02,b02]
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm5 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm2 , %%xmm6 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm3 , %%xmm7 , %%xmm15\n\t"
	                "\n\t"
	                "movapd   8*8(%[a]), %%xmm8 \n\t" // [a00,a10]
	                "movapd  10*8(%[a]), %%xmm9 \n\t" // [a01,a11]
	                "movapd  12*8(%[a]), %%xmm10\n\t" // [a02,a12]
	                "movapd  14*8(%[a]), %%xmm11\n\t" // [a03,a13]
	                "movapd   4*8(%[b]), %%xmm12\n\t" // [b00,b01]
	                "movapd   6*8(%[b]), %%xmm13\n\t" // [b02,b03]
	                "\n\t"
	                "vshufpd   $0x03, %%xmm12, %%xmm12, %%xmm5 \n\t" // [b01,b01]
	                "vshufpd   $0x00, %%xmm12, %%xmm12, %%xmm12\n\t" // [b00,b00]
	                "vshufpd   $0x03, %%xmm13, %%xmm13, %%xmm7 \n\t" // [b03,b03]
	                "vshufpd   $0x00, %%xmm13, %%xmm13, %%xmm13\n\t" // [b02,b02]
	                "vfmadd231pd %%xmm8 , %%xmm12, %%xmm15\n\t"
	                "vfmadd231pd %%xmm9 , %%xmm5 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm10, %%xmm13, %%xmm15\n\t"
	                "vfmadd231pd %%xmm11, %%xmm7 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	        }
	        if( K & 4 ){
	            //a00 = *(A + 0 + 0*2 ); a01 = *(A +  0 + 1*2 ); a02 = *(A + 0 + 2*2 ); a03 = *(A + 0 + 3*2 );
	            //a10 = *(A + 1 + 0*2 ); a11 = *(A +  1 + 1*2 ); a12 = *(A + 1 + 2*2 ); a13 = *(A + 1 + 3*2 );
	            //b00 = *(B + 0 + 0*1 ); b01 = *(B +  0 + 1*1 ); b02 = *(B + 0 + 2*1 ); b03 = *(B + 0 + 3*1 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //A+=8;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10]
	                "movapd   2*8(%[a]), %%xmm1 \n\t" // [a01,a11]
	                "movapd   4*8(%[a]), %%xmm2 \n\t" // [a02,a12]
	                "movapd   6*8(%[a]), %%xmm3 \n\t" // [a03,a13]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b01]
	                "movapd   2*8(%[b]), %%xmm6 \n\t" // [b02,b03]
	                "\n\t"
	                "vshufpd   $0x03, %%xmm4 , %%xmm4 , %%xmm5 \n\t" // [b01,b01]
	                "vshufpd   $0x00, %%xmm4 , %%xmm4 , %%xmm4 \n\t" // [b00,b00]
	                "vshufpd   $0x03, %%xmm6 , %%xmm6 , %%xmm7 \n\t" // [b03,b03]
	                "vshufpd   $0x00, %%xmm6 , %%xmm6 , %%xmm6 \n\t" // [b02,b02]
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm5 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm2 , %%xmm6 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm3 , %%xmm7 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 2 ){
	            //a00 = *(A + 0 + 0*2 ); a01 = *(A +  0 + 1*2 );
	            //a10 = *(A + 1 + 0*2 ); a11 = *(A +  1 + 1*2 );
	            //b00 = *(B + 0 + 0*1 ); b01 = *(B +  0 + 1*1 );
	            //c00 += a00 * b00; c00 += a01 * b01;
	            //c10 += a10 * b00; c10 += a11 * b01;
	            //A+=4;
	            //B+=2;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10]
	                "movapd   2*8(%[a]), %%xmm1 \n\t" // [a01,a11]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b01]
	                "\n\t"
	                "vshufpd   $0x03, %%xmm4 , %%xmm4 , %%xmm5 \n\t" // [b01,b01]
	                "vshufpd   $0x00, %%xmm4 , %%xmm4 , %%xmm4 \n\t" // [b00,b00]
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm5 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){
	            //a00 = *(A + 0 + 0*2 );
	            //a10 = *(A + 1 + 0*2 );
	            //b00 = *(B + 0 + 0*1 );
	            //c00 += a00 * b00;
	            //c10 += a10 * b00;
	            //A+=2;
	            //B+=1;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10]
	                "movsd    0*8(%[b]), %%xmm4 \n\t" // [b00]
	                "\n\t"
	                "vshufpd   $0x00, %%xmm4 , %%xmm4 , %%xmm4 \n\t" // [b00,b00]
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $2*8, %[a]\n\t"
	                "addq  $1*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+1+0*ldc) += alpha*c10;

	        __asm__ __volatile__ (
	            "\n\t"
	            "movupd %[alpha], %%xmm0\n\t"
	            "vfmadd213pd 0*8(%[c0]), %%xmm0, %%xmm15\n\t"
	            "movupd  %%xmm15, 0*8(%[c0])\n\t"
	            "addq  $2*8, %[c0]\n\t"
	            "addq  $2*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4)
	        );


	        B = B - 1*K;//N*K
	        //C+=2;
	        //c0+=2;
	        //c1+=2;

	    }
	    if( M & 1 ){

	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        //c10=0e0;c11=0e0;c12=0e0;c13=0e0;
	        //c20=0e0;c21=0e0;c22=0e0;c23=0e0;
	        //c30=0e0;c31=0e0;c32=0e0;c33=0e0;
	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*1 ); a01 = *(A +  0 + 1*1 ); a02 = *(A + 0 + 2*1 ); a03 = *(A + 0 + 3*1 );
	            //b00 = *(B + 0 + 0*1 ); b01 = *(B +  0 + 1*1 ); b02 = *(B + 0 + 2*1 ); b03 = *(B + 0 + 3*1 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //a00 = *(A + 0 + 4*1 ); a01 = *(A +  0 + 5*1 ); a02 = *(A + 0 + 6*1 ); a03 = *(A + 0 + 7*1 );
	            //b00 = *(B + 0 + 4*1 ); b01 = *(B +  0 + 5*1 ); b02 = *(B + 0 + 6*1 ); b03 = *(B + 0 + 7*1 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //A+=8;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovupd   0*8(%[a]), %%ymm0 \n\t" // [a00,a01,a02,a03]
	                "vmovupd   4*8(%[a]), %%ymm1 \n\t" // [a04,a05,a06,a07]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b01,b02,b03]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b04,b05,b06,b07]
	                "\n\t"
	                "vfmadd231pd %%ymm0  , %%ymm4 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1  , %%ymm5 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $8*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	        }
	        if( K & 4 ){
	            //a00 = *(A + 0 + 0*1 ); a01 = *(A +  0 + 1*1 ); a02 = *(A + 0 + 2*1 ); a03 = *(A + 0 + 3*1 );
	            //b00 = *(B + 0 + 0*1 ); b01 = *(B +  0 + 1*1 ); b02 = *(B + 0 + 2*1 ); b03 = *(B + 0 + 3*1 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //A+=4;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovupd   0*8(%[a]), %%ymm0 \n\t" // [a00,a01,a02,a03]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b01,b02,b03]
	                "\n\t"
	                "vfmadd231pd %%ymm0  , %%ymm4 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 2 ){
	            //a00 = *(A + 0 + 0*1 ); a01 = *(A +  0 + 1*1 );
	            //b00 = *(B + 0 + 0*1 ); b01 = *(B +  0 + 1*1 );
	            //c00 += a00 * b00; c00 += a01 * b01;
	            //A+=2;
	            //B+=2;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movupd   0*8(%[a]), %%xmm0 \n\t" // [a00,a01]
	                "movupd   0*8(%[b]), %%xmm4 \n\t" // [b00,b01]
	                "\n\t"
	                "mulpd  %%xmm0, %%xmm4 \n\t"
	                "addpd  %%xmm4, %%xmm15\n\t"
	                "\n\t"
	                "addq  $2*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
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

	            __asm__ __volatile__ (
	                "\n\t"
	                "movsd   0*8(%[a]), %%xmm0 \n\t" // [a00,0]
	                "movsd   0*8(%[b]), %%xmm4 \n\t" // [b00,0]
	                "\n\t"
	                "mulsd  %%xmm0, %%xmm4 \n\t" // [a00*b00,0]
	                "addsd  %%xmm4, %%xmm15\n\t"
	                "\n\t"
	                "addq  $1*8, %[a]\n\t"
	                "addq  $1*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        //*(C+0+0*ldc) += alpha*c00;

	        __asm__ __volatile__ (
	            "\n\t"
	            "movupd           %[alpha], %%xmm0 \n\t"
	            "movsd          0*8(%[c0]), %%xmm1 \n\t"
	            "vperm2f128 $0x01, %%ymm15, %%ymm15, %%ymm14\n\t"
	            "vhaddpd           %%ymm14, %%ymm15, %%ymm13\n\t"
	            "vdppd      $0x31, %%xmm13, %%xmm0 , %%xmm12\n\t"
	            "addsd             %%xmm1 , %%xmm12\n\t"
	            "movlpd            %%xmm12, 0*8(%[c0])\n\t"
	            "\n\t"
	            "addq  $1*8, %[c0]\n\t"
	            "addq  $1*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4)
	        );


	        B = B - 1*K;//N*K
	        //C+=1;
	        //c0+=1;
	        //c1+=1;

	    }
	    A = A - M*K;
	    B = B + 1*K;
	    //C  = C - M + 1*ldc;
	    c0 = c0- M + 1*ldc;
	    c1 = c1- M + 1*ldc;
	}

	A = A + M*K;
	B = B - K*N;
	//C  = C - ldc*N + M;
	c0 = c0- ldc*N + M;
	c1 = c1- ldc*N + M;
	// ---- Kernel

	//free(cc);
}

