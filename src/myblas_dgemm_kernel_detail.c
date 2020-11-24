#include "myblas_internal.h"
#include <stdio.h>
#include <stdlib.h>

/*  case 8x6x4x1

for( size_t k=0; k<4; k++ ){
 for( size_t j=0; j<6; j++ ){
   for( size_t i=0; i<8; i++ ){
      for( size_t l=0; l<1; l++ ){
        c[i+8*j] += (*(A2+l+i*1+k*1*8))*(*(B2+l+j*1+k*1*6));
      }
   }
 }
}
A2+=8*4*1;
B2+=4*1*6;

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

	size_t N6Q = N / 6;
	size_t N6R = N % 6;

	// ---- Kernel
	if( N6Q  ){
	  size_t n = ( N6Q );
	  while( n-- ){

	    if( M >> 3 ){
	      size_t m = ( M >> 3 );
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
	            "prefetcht0  0*8(%[c0]          )\n\t"
	            "prefetcht0  4*8(%[c0]          )\n\t"
	            "prefetcht0  0*8(%[c1]          )\n\t"
	            "prefetcht0  4*8(%[c1]          )\n\t"
	            "prefetcht0  0*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0  4*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0  0*8(%[c1],%[ldc2]  )\n\t"
	            "prefetcht0  4*8(%[c1],%[ldc2]  )\n\t"
	            "prefetcht0  0*8(%[c0],%[ldc2],2)\n\t"
	            "prefetcht0  4*8(%[c0],%[ldc2],2)\n\t"
	            "prefetcht0  0*8(%[c1],%[ldc2],2)\n\t"
	            "prefetcht0  4*8(%[c1],%[ldc2],2)\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );

	        if( K >> 2 ){
	          size_t k = ( K >> 2 );
	          k--;

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "prefetcht0    96*8(%[a])\n\t"
	                "prefetcht0    24*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128       0*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          while( k-- ){

	           // 
	           // c[i+8*j] += (*(A+l+i*1+k*1*8))*(*(B+l+j*1+k*1*6));
	           // 
	           // a[i,k]
	           // 
	           // a00 = *(A+0+0*1+0*1*8)
	           // a10 = *(A+0+1*1+0*1*8)
	           // a20 = *(A+0+2*1+0*1*8)
	           // a30 = *(A+0+3*1+0*1*8)
	           // ...
	           // a70 = *(A+0+7*1+0*1*8)
	           // a01 = *(A+0+0*1+1*1*8)
	           // a11 = *(A+0+1*1+1*1*8)
	           // ...
	           // a73 = *(A+0+7*1+3*1*8)
	           // 
	           // b[j,k]
	           // 
	           // b00 = *(B+0+0*1+0*1*6) 
	           // b10 = *(B+0+1*1+0*1*6) 
	           // b20 = *(B+0+2*1+0*1*6) 
	           // b30 = *(B+0+3*1+0*1*6) 
	           // ...
	           // b53 = *(B+0+4*1+3*1*6) 
	           // 
	           // c[i,j]
	           // 
	           // k=0 block
	           // c00 += a00*b00; c10 += a10*b00; c20 += a20*b00; c30 += a30*b00; c40 += a40*b00; c50 += a50*b00; c60 += a60*b00; c70 += a70*b00;
	           // c01 += a00*b10; c11 += a10*b10; c21 += a20*b10; c31 += a30*b10; c41 += a40*b10; c51 += a50*b10; c61 += a60*b10; c71 += a70*b10;
	           // c02 += a00*b20; c12 += a10*b20; c22 += a20*b20; c32 += a30*b20; c42 += a40*b20; c52 += a50*b20; c62 += a60*b20; c72 += a70*b20;
	           // c03 += a00*b30; c13 += a10*b30; c23 += a20*b30; c33 += a30*b30; c43 += a40*b30; c53 += a50*b30; c63 += a60*b30; c73 += a70*b30;
	           // c04 += a00*b40; c13 += a10*b40; c23 += a20*b40; c33 += a30*b40; c44 += a40*b40; c53 += a50*b40; c63 += a60*b40; c73 += a70*b40;
	           // c05 += a00*b50; c13 += a10*b50; c23 += a20*b50; c33 += a30*b50; c45 += a40*b50; c53 += a50*b50; c63 += a60*b50; c73 += a70*b50;
	           // 
	           // 
	           //__asm__ (
	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                //"prefetcht0    96*8(%[a])\n\t"
	                //"prefetcht0    24*8(%[b])\n\t"
	                "\n\t"
	                //"vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                //"vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                //"vbroadcastf128       0*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastf128       2*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b2k,b2k,b2k,b2k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm9 \n\t" // [c42,c52,c62,c72]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm10\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c43,c53,c63,c73]
	                "vbroadcastf128       4*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b4k,b4k,b4k,b4k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c14,c24,c34]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm13\n\t" // [c44,c54,c64,c74]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b5k,b5k,b5k,b5k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm14\n\t" // [c05,c15,c25,c35]
	                "vmovapd              8*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c45,c55,c65,c75]
	                "vbroadcastf128       6*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vmovapd             12*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "\n\t" // k=1
	                "prefetcht0   104*8(%[a])\n\t"
	                "prefetcht0    32*8(%[b])\n\t"
	                "\n\t"
	                //"vmovapd              8*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                //"vmovapd             12*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                //"vbroadcastf128       6*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastf128       8*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b2k,b2k,b2k,b2k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm9 \n\t" // [c42,c52,c62,c72]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm10\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c43,c53,c63,c73]
	                "vbroadcastf128      10*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b4k,b4k,b4k,b4k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c14,c24,c34]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm13\n\t" // [c44,c54,c64,c74]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b5k,b5k,b5k,b5k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm14\n\t" // [c05,c15,c25,c35]
	                "vmovapd             16*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c45,c55,c65,c75]
	                "vbroadcastf128      12*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vmovapd             20*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "\n\t" // k=2
	                "prefetcht0   112*8(%[a])\n\t"
	                "prefetcht0    40*8(%[b])\n\t"
	                "\n\t"
	                //"vmovapd             16*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                //"vmovapd             20*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                //"vbroadcastf128      12*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastf128      14*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b2k,b2k,b2k,b2k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm9 \n\t" // [c42,c52,c62,c72]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm10\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c43,c53,c63,c73]
	                "vbroadcastf128      16*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b4k,b4k,b4k,b4k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c14,c24,c34]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm13\n\t" // [c44,c54,c64,c74]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b5k,b5k,b5k,b5k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm14\n\t" // [c05,c15,c25,c35]
	                "vmovapd             24*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c45,c55,c65,c75]
	                "vbroadcastf128      18*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vmovapd             28*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "\n\t" // k=3
	                "prefetcht0   120*8(%[a])\n\t"
	                "prefetcht0    48*8(%[b])\n\t"
	                "\n\t"
	                //"vmovapd             24*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                //"vmovapd             28*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                //"vbroadcastf128      18*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastf128      20*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b2k,b2k,b2k,b2k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm9 \n\t" // [c42,c52,c62,c72]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm10\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c43,c53,c63,c73]
	                "vbroadcastf128      22*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b4k,b4k,b4k,b4k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c14,c24,c34]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm13\n\t" // [c44,c54,c64,c74]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b5k,b5k,b5k,b5k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm14\n\t" // [c05,c15,c25,c35]
	                "vmovapd             32*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c45,c55,c65,c75]
	                "vbroadcastf128      24*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vmovapd             24*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "\n\t"
	                "prefetcht0   128*8(%[a])\n\t"
	                "prefetcht0    56*8(%[b])\n\t"
	                "\n\t"
	                "addq  $32*8, %[a]\n\t"
	                "addq  $24*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }

 	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                //"prefetcht0    96*8(%[a])\n\t"
	                //"prefetcht0    24*8(%[b])\n\t"
	                //"\n\t"
	                //"vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                //"vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                //"vbroadcastf128       0*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastf128       2*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b2k,b2k,b2k,b2k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm9 \n\t" // [c42,c52,c62,c72]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm10\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c43,c53,c63,c73]
	                "vbroadcastf128       4*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b4k,b4k,b4k,b4k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c14,c24,c34]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm13\n\t" // [c44,c54,c64,c74]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b5k,b5k,b5k,b5k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm14\n\t" // [c05,c15,c25,c35]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c45,c55,c65,c75]
	                "\n\t" // k=1
	                "prefetcht0   104*8(%[a])\n\t"
	                "prefetcht0    32*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd              8*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd             12*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128       6*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastf128       8*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b2k,b2k,b2k,b2k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm9 \n\t" // [c42,c52,c62,c72]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm10\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c43,c53,c63,c73]
	                "vbroadcastf128      10*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b4k,b4k,b4k,b4k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c14,c24,c34]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm13\n\t" // [c44,c54,c64,c74]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b5k,b5k,b5k,b5k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm14\n\t" // [c05,c15,c25,c35]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c45,c55,c65,c75]
	                "\n\t" // k=2
	                "prefetcht0   112*8(%[a])\n\t"
	                "prefetcht0    40*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd             16*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd             20*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128      12*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastf128      14*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b2k,b2k,b2k,b2k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm9 \n\t" // [c42,c52,c62,c72]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm10\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c43,c53,c63,c73]
	                "vbroadcastf128      16*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b4k,b4k,b4k,b4k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c14,c24,c34]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm13\n\t" // [c44,c54,c64,c74]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b5k,b5k,b5k,b5k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm14\n\t" // [c05,c15,c25,c35]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c45,c55,c65,c75]
	                "\n\t" // k=3
	                "prefetcht0   120*8(%[a])\n\t"
	                "prefetcht0    48*8(%[b])\n\t"
	                "\n\t"
	                "vmovapd             24*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd             28*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128      18*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastf128      20*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b2k,b2k,b2k,b2k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm9 \n\t" // [c42,c52,c62,c72]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm10\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c43,c53,c63,c73]
	                "vbroadcastf128      22*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b4k,b4k,b4k,b4k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c14,c24,c34]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm13\n\t" // [c44,c54,c64,c74]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b5k,b5k,b5k,b5k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm14\n\t" // [c05,c15,c25,c35]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c45,c55,c65,c75]
	                "\n\t"
	                "addq  $32*8, %[a]\n\t"
	                "addq  $24*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128       0*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastf128       2*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b2k,b2k,b2k,b2k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm9 \n\t" // [c42,c52,c62,c72]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm10\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c43,c53,c63,c73]
	                "vbroadcastf128       4*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b4k,b4k,b4k,b4k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c14,c24,c34]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm13\n\t" // [c44,c54,c64,c74]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b5k,b5k,b5k,b5k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm14\n\t" // [c05,c15,c25,c35]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c45,c55,c65,c75]
	                "\n\t" // k=1
	                "vmovapd              8*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd             12*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128       6*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastf128       8*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b2k,b2k,b2k,b2k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm9 \n\t" // [c42,c52,c62,c72]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm10\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c43,c53,c63,c73]
	                "vbroadcastf128      10*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b4k,b4k,b4k,b4k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c14,c24,c34]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm13\n\t" // [c44,c54,c64,c74]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b5k,b5k,b5k,b5k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm14\n\t" // [c05,c15,c25,c35]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c45,c55,c65,c75]
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $12*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128       0*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastf128       2*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b2k,b2k,b2k,b2k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm9 \n\t" // [c42,c52,c62,c72]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm10\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c43,c53,c63,c73]
	                "vbroadcastf128       4*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b4k,b4k,b4k,b4k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm12\n\t" // [c04,c14,c24,c34]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm13\n\t" // [c44,c54,c64,c74]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b5k,b5k,b5k,b5k]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm14\n\t" // [c05,c15,c25,c35]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm15\n\t" // [c45,c55,c65,c75]
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $6*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        //  %%ymm4  [c00,c10,c20,c30]
	        //  %%ymm5  [c40,c50,c60,c70]
	        //  %%ymm6  [c01,c11,c21,c31]
	        //  %%ymm7  [c41,c51,c61,c71]
	        //  %%ymm8  [c02,c12,c22,c32]
	        //  %%ymm9  [c42,c52,c62,c72]
	        //  %%ymm10 [c03,c13,c23,c33]
	        //  %%ymm11 [c43,c53,c63,c73]
	        //  %%ymm12 [c04,c14,c24,c34]
	        //  %%ymm13 [c44,c54,c64,c74]
	        //  %%ymm14 [c05,c15,c25,c35]
	        //  %%ymm15 [c45,c55,c65,c75]

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%ymm0, %%ymm4 \n\t"
	            "vfmadd213pd 4*8(%[c0]          ), %%ymm0, %%ymm5 \n\t"
	            "vfmadd213pd 0*8(%[c1]          ), %%ymm0, %%ymm6 \n\t"
	            "vfmadd213pd 4*8(%[c1]          ), %%ymm0, %%ymm7 \n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc2]  ), %%ymm0, %%ymm8 \n\t"
	            "vfmadd213pd 4*8(%[c0],%[ldc2]  ), %%ymm0, %%ymm9 \n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2]  ), %%ymm0, %%ymm10\n\t"
	            "vfmadd213pd 4*8(%[c1],%[ldc2]  ), %%ymm0, %%ymm11\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc2],2), %%ymm0, %%ymm12\n\t"
	            "vfmadd213pd 4*8(%[c0],%[ldc2],2), %%ymm0, %%ymm13\n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2],2), %%ymm0, %%ymm14\n\t"
	            "vfmadd213pd 4*8(%[c1],%[ldc2],2), %%ymm0, %%ymm15\n\t"
	            "\n\t"
	            "vmovupd  %%ymm4 , 0*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm5 , 4*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm6 , 0*8(%[c1]          )\n\t"
	            "vmovupd  %%ymm7 , 4*8(%[c1]          )\n\t"
	            "vmovupd  %%ymm8 , 0*8(%[c0],%[ldc2]  )\n\t"
	            "vmovupd  %%ymm9 , 4*8(%[c0],%[ldc2]  )\n\t"
	            "vmovupd  %%ymm10, 0*8(%[c1],%[ldc2]  )\n\t"
	            "vmovupd  %%ymm11, 4*8(%[c1],%[ldc2]  )\n\t"
	            "vmovupd  %%ymm12, 0*8(%[c0],%[ldc2],2)\n\t"
	            "vmovupd  %%ymm13, 4*8(%[c0],%[ldc2],2)\n\t"
	            "vmovupd  %%ymm14, 0*8(%[c1],%[ldc2],2)\n\t"
	            "vmovupd  %%ymm15, 4*8(%[c1],%[ldc2],2)\n\t"
	            "\n\t"
	            "addq  $8*8, %[c0]\n\t"
	            "addq  $8*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 6*K;

	      }
	    }
	    if( M & 4 ){

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
	            "prefetcht0  0*8(%[c0]          )\n\t"
	            "prefetcht0  0*8(%[c1]          )\n\t"
	            "prefetcht0  0*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0  0*8(%[c1],%[ldc2]  )\n\t"
	            "prefetcht0  0*8(%[c0],%[ldc2],2)\n\t"
	            "prefetcht0  0*8(%[c1],%[ldc2],2)\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );

	        if( K >> 2 ){
	          size_t k = ( K >> 2 );
	          while( k-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128       0*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "vbroadcastf128       2*8(%[b]), %%ymm7 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm7 , %%ymm7 , %%ymm6 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm7 , %%ymm7 , %%ymm7 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm6 , %%ymm12\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm7 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vbroadcastf128       4*8(%[b]), %%ymm9 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm9 , %%ymm9 , %%ymm8 \n\t" // [b4k,b4k,b4k,b4k]
	                "vshufpd       $0x0f  , %%ymm9 , %%ymm9 , %%ymm9 \n\t" // [b5k,b5k,b5k,b5k]
	                "vfmadd231pd   %%ymm0 , %%ymm8 , %%ymm14\n\t" // [c04,c14,c24,c34]
	                "vfmadd231pd   %%ymm0 , %%ymm9 , %%ymm15\n\t" // [c05,c15,c25,c35]
	                "\n\t" // k=1
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128       6*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm1 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "vbroadcastf128       8*8(%[b]), %%ymm7 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm7 , %%ymm7 , %%ymm6 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm7 , %%ymm7 , %%ymm7 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm1 , %%ymm6 , %%ymm12\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm7 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vbroadcastf128      10*8(%[b]), %%ymm9 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm9 , %%ymm9 , %%ymm8 \n\t" // [b4k,b4k,b4k,b4k]
	                "vshufpd       $0x0f  , %%ymm9 , %%ymm9 , %%ymm9 \n\t" // [b5k,b5k,b5k,b5k]
	                "vfmadd231pd   %%ymm1 , %%ymm8 , %%ymm14\n\t" // [c04,c14,c24,c34]
	                "vfmadd231pd   %%ymm1 , %%ymm9 , %%ymm15\n\t" // [c05,c15,c25,c35]
	                "\n\t" // k=2
	                "vmovapd              8*8(%[a]), %%ymm2 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128      12*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm2 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm2 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "vbroadcastf128      14*8(%[b]), %%ymm7 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm7 , %%ymm7 , %%ymm6 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm7 , %%ymm7 , %%ymm7 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm2 , %%ymm6 , %%ymm12\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm2 , %%ymm7 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vbroadcastf128      16*8(%[b]), %%ymm9 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm9 , %%ymm9 , %%ymm8 \n\t" // [b4k,b4k,b4k,b4k]
	                "vshufpd       $0x0f  , %%ymm9 , %%ymm9 , %%ymm9 \n\t" // [b5k,b5k,b5k,b5k]
	                "vfmadd231pd   %%ymm2 , %%ymm8 , %%ymm14\n\t" // [c04,c14,c24,c34]
	                "vfmadd231pd   %%ymm2 , %%ymm9 , %%ymm15\n\t" // [c05,c15,c25,c35]
	                "\n\t" // k=3
	                "vmovapd             12*8(%[a]), %%ymm3 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128      18*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm3 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm3 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "vbroadcastf128      20*8(%[b]), %%ymm7 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm7 , %%ymm7 , %%ymm6 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm7 , %%ymm7 , %%ymm7 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm3 , %%ymm6 , %%ymm12\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm3 , %%ymm7 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vbroadcastf128      22*8(%[b]), %%ymm9 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm9 , %%ymm9 , %%ymm8 \n\t" // [b4k,b4k,b4k,b4k]
	                "vshufpd       $0x0f  , %%ymm9 , %%ymm9 , %%ymm9 \n\t" // [b5k,b5k,b5k,b5k]
	                "vfmadd231pd   %%ymm3 , %%ymm8 , %%ymm14\n\t" // [c04,c14,c24,c34]
	                "vfmadd231pd   %%ymm3 , %%ymm9 , %%ymm15\n\t" // [c05,c15,c25,c35]
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $24*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128       0*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "vbroadcastf128       2*8(%[b]), %%ymm7 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm7 , %%ymm7 , %%ymm6 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm7 , %%ymm7 , %%ymm7 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm6 , %%ymm12\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm7 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vbroadcastf128       4*8(%[b]), %%ymm9 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm9 , %%ymm9 , %%ymm8 \n\t" // [b4k,b4k,b4k,b4k]
	                "vshufpd       $0x0f  , %%ymm9 , %%ymm9 , %%ymm9 \n\t" // [b5k,b5k,b5k,b5k]
	                "vfmadd231pd   %%ymm0 , %%ymm8 , %%ymm14\n\t" // [c04,c14,c24,c34]
	                "vfmadd231pd   %%ymm0 , %%ymm9 , %%ymm15\n\t" // [c05,c15,c25,c35]
	                "\n\t" // k=1
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128       6*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm1 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "vbroadcastf128       8*8(%[b]), %%ymm7 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm7 , %%ymm7 , %%ymm6 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm7 , %%ymm7 , %%ymm7 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm1 , %%ymm6 , %%ymm12\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm7 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vbroadcastf128      10*8(%[b]), %%ymm9 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm9 , %%ymm9 , %%ymm8 \n\t" // [b4k,b4k,b4k,b4k]
	                "vshufpd       $0x0f  , %%ymm9 , %%ymm9 , %%ymm9 \n\t" // [b5k,b5k,b5k,b5k]
	                "vfmadd231pd   %%ymm1 , %%ymm8 , %%ymm14\n\t" // [c04,c14,c24,c34]
	                "vfmadd231pd   %%ymm1 , %%ymm9 , %%ymm15\n\t" // [c05,c15,c25,c35]
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $12*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128       0*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "vbroadcastf128       2*8(%[b]), %%ymm7 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm7 , %%ymm7 , %%ymm6 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm7 , %%ymm7 , %%ymm7 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm6 , %%ymm12\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm7 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "vbroadcastf128       4*8(%[b]), %%ymm9 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm9 , %%ymm9 , %%ymm8 \n\t" // [b4k,b4k,b4k,b4k]
	                "vshufpd       $0x0f  , %%ymm9 , %%ymm9 , %%ymm9 \n\t" // [b5k,b5k,b5k,b5k]
	                "vfmadd231pd   %%ymm0 , %%ymm8 , %%ymm14\n\t" // [c04,c14,c24,c34]
	                "vfmadd231pd   %%ymm0 , %%ymm9 , %%ymm15\n\t" // [c05,c15,c25,c35]
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $6*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        //  %%ymm10 [c00,c10,c20,c30]
	        //  %%ymm11 [c01,c11,c21,c31]
	        //  %%ymm12 [c02,c12,c22,c32]
	        //  %%ymm13 [c03,c13,c23,c33]
	        //  %%ymm14 [c04,c14,c24,c34]
	        //  %%ymm15 [c05,c15,c25,c35]

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%ymm0, %%ymm10\n\t"
	            "vfmadd213pd 0*8(%[c1]          ), %%ymm0, %%ymm11\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc2]  ), %%ymm0, %%ymm12\n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2]  ), %%ymm0, %%ymm13\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc2],2), %%ymm0, %%ymm14\n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2],2), %%ymm0, %%ymm15\n\t"
	            "\n\t"
	            "vmovupd  %%ymm10, 0*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm11, 0*8(%[c1]          )\n\t"
	            "vmovupd  %%ymm12, 0*8(%[c0],%[ldc2]  )\n\t"
	            "vmovupd  %%ymm13, 0*8(%[c1],%[ldc2]  )\n\t"
	            "vmovupd  %%ymm14, 0*8(%[c0],%[ldc2],2)\n\t"
	            "vmovupd  %%ymm15, 0*8(%[c1],%[ldc2],2)\n\t"
	            "\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "addq  $4*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 6*K;

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
	            "prefetcht0  0*8(%[c0]          )\n\t"
	            "prefetcht0  0*8(%[c1]          )\n\t"
	            "prefetcht0  0*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0  0*8(%[c1],%[ldc2]  )\n\t"
	            "prefetcht0  0*8(%[c0],%[ldc2],2)\n\t"
	            "prefetcht0  0*8(%[c1],%[ldc2],2)\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );

	        if( K >> 2 ){
	          size_t k = ( K >> 2 );
	          while( k-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%xmm0 \n\t" // [a0k,a1k,---,---]
	                "vmovupd              0*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "vmovupd              2*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm7 , %%xmm7 , %%xmm6 \n\t" // [b2k,b2k,---,---]
	                "vshufpd       $0x0f  , %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b3k,b3k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm6 , %%xmm12\n\t" // [c02,c12,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm7 , %%xmm13\n\t" // [c03,c13,---,---]
	                "vmovupd              4*8(%[b]), %%xmm9 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm9 , %%xmm9 , %%xmm8 \n\t" // [b4k,b4k,---,---]
	                "vshufpd       $0x0f  , %%xmm9 , %%xmm9 , %%xmm9 \n\t" // [b5k,b5k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm8 , %%xmm14\n\t" // [c04,c14,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm9 , %%xmm15\n\t" // [c05,c15,---,---]
	                "\n\t" // k=1
	                "vmovapd              2*8(%[a]), %%xmm1 \n\t" // [a0k,a1k,---,---]
	                "vmovupd              6*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "vmovupd              8*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm7 , %%xmm7 , %%xmm6 \n\t" // [b2k,b2k,---,---]
	                "vshufpd       $0x0f  , %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b3k,b3k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm6 , %%xmm12\n\t" // [c02,c12,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm7 , %%xmm13\n\t" // [c03,c13,---,---]
	                "vmovupd             10*8(%[b]), %%xmm9 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm9 , %%xmm9 , %%xmm8 \n\t" // [b4k,b4k,---,---]
	                "vshufpd       $0x0f  , %%xmm9 , %%xmm9 , %%xmm9 \n\t" // [b5k,b5k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm8 , %%xmm14\n\t" // [c04,c14,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm9 , %%xmm15\n\t" // [c05,c15,---,---]
	                "\n\t" // k=2
	                "vmovapd              4*8(%[a]), %%xmm2 \n\t" // [a0k,a1k,---,---]
	                "vmovupd             12*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "vmovupd             14*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm7 , %%xmm7 , %%xmm6 \n\t" // [b2k,b2k,---,---]
	                "vshufpd       $0x0f  , %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b3k,b3k,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm6 , %%xmm12\n\t" // [c02,c12,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm7 , %%xmm13\n\t" // [c03,c13,---,---]
	                "vmovupd             16*8(%[b]), %%xmm9 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm9 , %%xmm9 , %%xmm8 \n\t" // [b4k,b4k,---,---]
	                "vshufpd       $0x0f  , %%xmm9 , %%xmm9 , %%xmm9 \n\t" // [b5k,b5k,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm8 , %%xmm14\n\t" // [c04,c14,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm9 , %%xmm15\n\t" // [c05,c15,---,---]
	                "\n\t" // k=3
	                "vmovapd              6*8(%[a]), %%xmm3 \n\t" // [a0k,a1k,---,---]
	                "vmovupd             18*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "vmovupd             20*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm7 , %%xmm7 , %%xmm6 \n\t" // [b2k,b2k,---,---]
	                "vshufpd       $0x0f  , %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b3k,b3k,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm6 , %%xmm12\n\t" // [c02,c12,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm7 , %%xmm13\n\t" // [c03,c13,---,---]
	                "vmovupd             22*8(%[b]), %%xmm9 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm9 , %%xmm9 , %%xmm8 \n\t" // [b4k,b4k,---,---]
	                "vshufpd       $0x0f  , %%xmm9 , %%xmm9 , %%xmm9 \n\t" // [b5k,b5k,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm8 , %%xmm14\n\t" // [c04,c14,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm9 , %%xmm15\n\t" // [c05,c15,---,---]
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $24*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%xmm0 \n\t" // [a0k,a1k,---,---]
	                "vmovupd              0*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "vmovupd              2*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm7 , %%xmm7 , %%xmm6 \n\t" // [b2k,b2k,---,---]
	                "vshufpd       $0x0f  , %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b3k,b3k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm6 , %%xmm12\n\t" // [c02,c12,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm7 , %%xmm13\n\t" // [c03,c13,---,---]
	                "vmovupd              4*8(%[b]), %%xmm9 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm9 , %%xmm9 , %%xmm8 \n\t" // [b4k,b4k,---,---]
	                "vshufpd       $0x0f  , %%xmm9 , %%xmm9 , %%xmm9 \n\t" // [b5k,b5k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm8 , %%xmm14\n\t" // [c04,c14,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm9 , %%xmm15\n\t" // [c05,c15,---,---]
	                "\n\t" // k=1
	                "vmovapd              2*8(%[a]), %%xmm1 \n\t" // [a0k,a1k,---,---]
	                "vmovupd              6*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "vmovupd              8*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm7 , %%xmm7 , %%xmm6 \n\t" // [b2k,b2k,---,---]
	                "vshufpd       $0x0f  , %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b3k,b3k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm6 , %%xmm12\n\t" // [c02,c12,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm7 , %%xmm13\n\t" // [c03,c13,---,---]
	                "vmovupd             10*8(%[b]), %%xmm9 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm9 , %%xmm9 , %%xmm8 \n\t" // [b4k,b4k,---,---]
	                "vshufpd       $0x0f  , %%xmm9 , %%xmm9 , %%xmm9 \n\t" // [b5k,b5k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm8 , %%xmm14\n\t" // [c04,c14,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm9 , %%xmm15\n\t" // [c05,c15,---,---]
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $12*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%xmm0 \n\t" // [a0k,a1k,---,---]
	                "vmovupd              0*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "vmovupd              2*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm7 , %%xmm7 , %%xmm6 \n\t" // [b2k,b2k,---,---]
	                "vshufpd       $0x0f  , %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b3k,b3k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm6 , %%xmm12\n\t" // [c02,c12,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm7 , %%xmm13\n\t" // [c03,c13,---,---]
	                "vmovupd              4*8(%[b]), %%xmm9 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm9 , %%xmm9 , %%xmm8 \n\t" // [b4k,b4k,---,---]
	                "vshufpd       $0x0f  , %%xmm9 , %%xmm9 , %%xmm9 \n\t" // [b5k,b5k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm8 , %%xmm14\n\t" // [c04,c14,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm9 , %%xmm15\n\t" // [c05,c15,---,---]
	                "\n\t"
	                "addq  $2*8, %[a]\n\t"
	                "addq  $6*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        //  %%xmm10 [c00,c10,---,---]
	        //  %%xmm11 [c01,c11,---,---]
	        //  %%xmm12 [c02,c12,---,---]
	        //  %%xmm13 [c03,c13,---,---]
	        //  %%xmm14 [c04,c14,---,---]
	        //  %%xmm15 [c05,c15,---,---]

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%xmm0, %%xmm10\n\t"
	            "vfmadd213pd 0*8(%[c1]          ), %%xmm0, %%xmm11\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc2]  ), %%xmm0, %%xmm12\n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2]  ), %%xmm0, %%xmm13\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc2],2), %%xmm0, %%xmm14\n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2],2), %%xmm0, %%xmm15\n\t"
	            "\n\t"
	            "vmovupd  %%xmm10, 0*8(%[c0]          )\n\t"
	            "vmovupd  %%xmm11, 0*8(%[c1]          )\n\t"
	            "vmovupd  %%xmm12, 0*8(%[c0],%[ldc2]  )\n\t"
	            "vmovupd  %%xmm13, 0*8(%[c1],%[ldc2]  )\n\t"
	            "vmovupd  %%xmm14, 0*8(%[c0],%[ldc2],2)\n\t"
	            "vmovupd  %%xmm15, 0*8(%[c1],%[ldc2],2)\n\t"
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
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);

	        __asm__ __volatile__ (
	            "\n\t"
	            "prefetcht0  0*8(%[c0]          )\n\t"
	            "prefetcht0  0*8(%[c1]          )\n\t"
	            "prefetcht0  0*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0  0*8(%[c1],%[ldc2]  )\n\t"
	            "prefetcht0  0*8(%[c0],%[ldc2],2)\n\t"
	            "prefetcht0  0*8(%[c1],%[ldc2],2)\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );

	        if( K >> 2 ){
	          size_t k = ( K >> 2 );
	          while( k-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovsd               0*8(%[a]), %%xmm0 \n\t" // [a0k,---,---,---]
	                "vmovupd              0*8(%[b]), %%xmm4 \n\t" // [b0k,b1k,---,---]
	                "vmovupd              2*8(%[b]), %%xmm6 \n\t" // [b2k,b3k,---,---]
	                "vmovupd              4*8(%[b]), %%xmm8 \n\t" // [b4k,b5k,---,---]
	                "vshufpd       $0x00  , %%xmm0 , %%xmm0 , %%xmm0 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm13\n\t" // [c00,c01,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm6 , %%xmm14\n\t" // [c02,c03,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm8 , %%xmm15\n\t" // [c04,c05,---,---]
	                "\n\t" // k=1
	                "vmovsd               1*8(%[a]), %%xmm1 \n\t" // [a0k,---,---,---]
	                "vmovupd              6*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vmovupd              8*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vmovupd             10*8(%[b]), %%xmm9 \n\t" // [b4k,b5k,---,---]
	                "vshufpd       $0x00  , %%xmm1 , %%xmm1 , %%xmm1 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm5 , %%xmm13\n\t" // [c00,c01,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm7 , %%xmm14\n\t" // [c02,c03,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm9 , %%xmm15\n\t" // [c04,c05,---,---]
	                "\n\t" // k=2
	                "vmovsd               2*8(%[a]), %%xmm2 \n\t" // [a0k,---,---,---]
	                "vmovupd             12*8(%[b]), %%xmm4 \n\t" // [b0k,b1k,---,---]
	                "vmovupd             14*8(%[b]), %%xmm6 \n\t" // [b2k,b3k,---,---]
	                "vmovupd             16*8(%[b]), %%xmm8 \n\t" // [b4k,b5k,---,---]
	                "vshufpd       $0x00  , %%xmm2 , %%xmm2 , %%xmm2 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm4 , %%xmm13\n\t" // [c00,c01,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm6 , %%xmm14\n\t" // [c02,c03,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm8 , %%xmm15\n\t" // [c04,c05,---,---]
	                "\n\t" // k=3
	                "vmovsd               3*8(%[a]), %%xmm3 \n\t" // [a0k,---,---,---]
	                "vmovupd             18*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vmovupd             20*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vmovupd             22*8(%[b]), %%xmm9 \n\t" // [b4k,b5k,---,---]
	                "vshufpd       $0x00  , %%xmm3 , %%xmm3 , %%xmm3 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm5 , %%xmm13\n\t" // [c00,c01,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm7 , %%xmm14\n\t" // [c02,c03,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm9 , %%xmm15\n\t" // [c04,c05,---,---]
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $24*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovsd               0*8(%[a]), %%xmm0 \n\t" // [a0k,---,---,---]
	                "vmovupd              0*8(%[b]), %%xmm4 \n\t" // [b0k,b1k,---,---]
	                "vmovupd              2*8(%[b]), %%xmm6 \n\t" // [b2k,b3k,---,---]
	                "vmovupd              4*8(%[b]), %%xmm8 \n\t" // [b4k,b5k,---,---]
	                "vshufpd       $0x00  , %%xmm0 , %%xmm0 , %%xmm0 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm13\n\t" // [c00,c01,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm6 , %%xmm14\n\t" // [c02,c03,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm8 , %%xmm15\n\t" // [c04,c05,---,---]
	                "\n\t" // k=1
	                "vmovsd               1*8(%[a]), %%xmm1 \n\t" // [a0k,---,---,---]
	                "vmovupd              6*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vmovupd              8*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vmovupd             10*8(%[b]), %%xmm9 \n\t" // [b4k,b5k,---,---]
	                "vshufpd       $0x00  , %%xmm1 , %%xmm1 , %%xmm1 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm5 , %%xmm13\n\t" // [c00,c01,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm7 , %%xmm14\n\t" // [c02,c03,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm9 , %%xmm15\n\t" // [c04,c05,---,---]
	                "\n\t"
	                "addq  $2*8, %[a]\n\t"
	                "addq  $12*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovsd               0*8(%[a]), %%xmm0 \n\t" // [a0k,---,---,---]
	                "vmovupd              0*8(%[b]), %%xmm4 \n\t" // [b0k,b1k,---,---]
	                "vmovupd              2*8(%[b]), %%xmm6 \n\t" // [b2k,b3k,---,---]
	                "vmovupd              4*8(%[b]), %%xmm8 \n\t" // [b4k,b5k,---,---]
	                "vshufpd       $0x00  , %%xmm0 , %%xmm0 , %%xmm0 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm13\n\t" // [c00,c01,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm6 , %%xmm14\n\t" // [c02,c03,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm8 , %%xmm15\n\t" // [c04,c05,---,---]
	                "\n\t"
	                "addq  $1*8, %[a]\n\t"
	                "addq  $6*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        //  %%xmm13 [c00,c01,---,---]
	        //  %%xmm14 [c02,c03,---,---]
	        //  %%xmm15 [c04,c05,---,---]

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vmovlpd  0*8(%[c0]          ), %%xmm10, %%xmm10\n\t"
	            "vmovhpd  0*8(%[c1]          ), %%xmm10, %%xmm10\n\t"
	            "vmovlpd  0*8(%[c0],%[ldc2]  ), %%xmm11, %%xmm11\n\t"
	            "vmovhpd  0*8(%[c1],%[ldc2]  ), %%xmm11, %%xmm11\n\t"
	            "vmovlpd  0*8(%[c0],%[ldc2],2), %%xmm12, %%xmm12\n\t"
	            "vmovhpd  0*8(%[c1],%[ldc2],2), %%xmm12, %%xmm12\n\t"
	            "\n\t"
	            "vfmadd213pd  %%xmm10, %%xmm0, %%xmm13\n\t"
	            "vfmadd213pd  %%xmm11, %%xmm0, %%xmm14\n\t"
	            "vfmadd213pd  %%xmm12, %%xmm0, %%xmm15\n\t"
	            "\n\t"
	            "vmovlpd  %%xmm13, 0*8(%[c0]          )\n\t"
	            "vmovhpd  %%xmm13, 0*8(%[c1]          )\n\t"
	            "vmovlpd  %%xmm14, 0*8(%[c0],%[ldc2]  )\n\t"
	            "vmovhpd  %%xmm14, 0*8(%[c1],%[ldc2]  )\n\t"
	            "vmovlpd  %%xmm15, 0*8(%[c0],%[ldc2],2)\n\t"
	            "vmovhpd  %%xmm15, 0*8(%[c1],%[ldc2],2)\n\t"
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
	    c0 = c0 - M + 6*ldc;
	    c1 = c1 - M + 6*ldc;

	  }

	}
/****************************************** N mod 6 = 4 *****************************************************/
	if( N6R & 4 ){ 

	    if( M >> 3 ){
	      size_t m = ( M >> 3 );
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
	            "prefetcht0  0*8(%[c0]          )\n\t"
	            "prefetcht0  4*8(%[c0]          )\n\t"
	            "prefetcht0  0*8(%[c1]          )\n\t"
	            "prefetcht0  4*8(%[c1]          )\n\t"
	            "prefetcht0  0*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0  4*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0  0*8(%[c1],%[ldc2]  )\n\t"
	            "prefetcht0  4*8(%[c1],%[ldc2]  )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );


	        if( K >> 2 ){
	          size_t k = ( K >> 2 );
	          while( k-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128       0*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastf128       2*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm9 \n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm10\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c43,c53,c63,c73]
	                "\n\t" // k=1
	                "vmovapd              8*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd             12*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128       4*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastf128       6*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm9 \n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm10\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c43,c53,c63,c73]
	                "\n\t" // k=2
	                "vmovapd             16*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd             20*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128       8*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastf128      10*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm9 \n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm10\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c43,c53,c63,c73]
	                "\n\t" // k=3
	                "vmovapd             24*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd             28*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128      12*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastf128      14*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm9 \n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm10\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c43,c53,c63,c73]
	                "\n\t"
	                "addq  $32*8, %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128       0*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastf128       2*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm9 \n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm10\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c43,c53,c63,c73]
	                "\n\t" // k=1
	                "vmovapd              8*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd             12*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128       4*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastf128       6*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm9 \n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm10\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c43,c53,c63,c73]
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $8*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128       0*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "vbroadcastf128       2*8(%[b]), %%ymm3 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm8 \n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm9 \n\t" // [c42,c52,c62,c72]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm10\n\t" // [c03,c13,c23,c33]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm11\n\t" // [c43,c53,c63,c73]
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        //  %%ymm4  [c00,c10,c20,c30]
	        //  %%ymm5  [c40,c50,c60,c70]
	        //  %%ymm6  [c01,c11,c21,c31]
	        //  %%ymm7  [c41,c51,c61,c71]
	        //  %%ymm8  [c02,c12,c22,c32]
	        //  %%ymm9  [c42,c52,c62,c72]
	        //  %%ymm10 [c03,c13,c23,c33]
	        //  %%ymm11 [c43,c53,c63,c73]

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%ymm0, %%ymm4 \n\t"
	            "vfmadd213pd 4*8(%[c0]          ), %%ymm0, %%ymm5 \n\t"
	            "vfmadd213pd 0*8(%[c1]          ), %%ymm0, %%ymm6 \n\t"
	            "vfmadd213pd 4*8(%[c1]          ), %%ymm0, %%ymm7 \n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc2]  ), %%ymm0, %%ymm8 \n\t"
	            "vfmadd213pd 4*8(%[c0],%[ldc2]  ), %%ymm0, %%ymm9 \n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2]  ), %%ymm0, %%ymm10\n\t"
	            "vfmadd213pd 4*8(%[c1],%[ldc2]  ), %%ymm0, %%ymm11\n\t"
	            "\n\t"
	            "vmovupd  %%ymm4 , 0*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm5 , 4*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm6 , 0*8(%[c1]          )\n\t"
	            "vmovupd  %%ymm7 , 4*8(%[c1]          )\n\t"
	            "vmovupd  %%ymm8 , 0*8(%[c0],%[ldc2]  )\n\t"
	            "vmovupd  %%ymm9 , 4*8(%[c0],%[ldc2]  )\n\t"
	            "vmovupd  %%ymm10, 0*8(%[c1],%[ldc2]  )\n\t"
	            "vmovupd  %%ymm11, 4*8(%[c1],%[ldc2]  )\n\t"
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
	    if( M & 4 ){

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
	            "prefetcht0  0*8(%[c0]          )\n\t"
	            "prefetcht0  0*8(%[c1]          )\n\t"
	            "prefetcht0  0*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0  0*8(%[c1],%[ldc2]  )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );


	        if( K >> 2 ){
	          size_t k = ( K >> 2 );
	          while( k-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128       0*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "vbroadcastf128       2*8(%[b]), %%ymm7 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm7 , %%ymm7 , %%ymm6 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm7 , %%ymm7 , %%ymm7 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm6 , %%ymm12\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm7 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t" // k=1
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128       4*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm1 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "vbroadcastf128       6*8(%[b]), %%ymm7 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm7 , %%ymm7 , %%ymm6 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm7 , %%ymm7 , %%ymm7 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm1 , %%ymm6 , %%ymm12\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm7 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t" // k=2
	                "vmovapd              8*8(%[a]), %%ymm2 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128       8*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm2 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm2 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "vbroadcastf128      10*8(%[b]), %%ymm7 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm7 , %%ymm7 , %%ymm6 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm7 , %%ymm7 , %%ymm7 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm2 , %%ymm6 , %%ymm12\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm2 , %%ymm7 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t" // k=3
	                "vmovapd             12*8(%[a]), %%ymm3 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128      12*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm3 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm3 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "vbroadcastf128      14*8(%[b]), %%ymm7 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm7 , %%ymm7 , %%ymm6 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm7 , %%ymm7 , %%ymm7 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm3 , %%ymm6 , %%ymm12\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm3 , %%ymm7 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128       0*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "vbroadcastf128       2*8(%[b]), %%ymm7 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm7 , %%ymm7 , %%ymm6 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm7 , %%ymm7 , %%ymm7 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm6 , %%ymm12\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm7 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t" // k=1
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128       4*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm1 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "vbroadcastf128       6*8(%[b]), %%ymm7 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm7 , %%ymm7 , %%ymm6 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm7 , %%ymm7 , %%ymm7 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm1 , %%ymm6 , %%ymm12\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm1 , %%ymm7 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $8*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128       0*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "vbroadcastf128       2*8(%[b]), %%ymm7 \n\t" // [b2k,b3k,b2k,b3k]
	                "vshufpd       $0x00  , %%ymm7 , %%ymm7 , %%ymm6 \n\t" // [b2k,b2k,b2k,b2k]
	                "vshufpd       $0x0f  , %%ymm7 , %%ymm7 , %%ymm7 \n\t" // [b3k,b3k,b3k,b3k]
	                "vfmadd231pd   %%ymm0 , %%ymm6 , %%ymm12\n\t" // [c02,c12,c22,c32]
	                "vfmadd231pd   %%ymm0 , %%ymm7 , %%ymm13\n\t" // [c03,c13,c23,c33]
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        //  %%ymm10 [c00,c10,c20,c30]
	        //  %%ymm11 [c01,c11,c21,c31]
	        //  %%ymm12 [c02,c12,c22,c32]
	        //  %%ymm13 [c03,c13,c23,c33]
	        //  %%ymm14 [c04,c14,c24,c34]
	        //  %%ymm15 [c05,c15,c25,c35]

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%ymm0, %%ymm10\n\t"
	            "vfmadd213pd 0*8(%[c1]          ), %%ymm0, %%ymm11\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc2]  ), %%ymm0, %%ymm12\n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2]  ), %%ymm0, %%ymm13\n\t"
	            "\n\t"
	            "vmovupd  %%ymm10, 0*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm11, 0*8(%[c1]          )\n\t"
	            "vmovupd  %%ymm12, 0*8(%[c0],%[ldc2]  )\n\t"
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
	            "prefetcht0  0*8(%[c0]          )\n\t"
	            "prefetcht0  0*8(%[c1]          )\n\t"
	            "prefetcht0  0*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0  0*8(%[c1],%[ldc2]  )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );


	        if( K >> 2 ){
	          size_t k = ( K >> 2 );
	          while( k-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%xmm0 \n\t" // [a0k,a1k,---,---]
	                "vmovupd              0*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "vmovupd              2*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm7 , %%xmm7 , %%xmm6 \n\t" // [b2k,b2k,---,---]
	                "vshufpd       $0x0f  , %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b3k,b3k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm6 , %%xmm12\n\t" // [c02,c12,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm7 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t" // k=1
	                "vmovapd              2*8(%[a]), %%xmm1 \n\t" // [a0k,a1k,---,---]
	                "vmovupd              4*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "vmovupd              6*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm7 , %%xmm7 , %%xmm6 \n\t" // [b2k,b2k,---,---]
	                "vshufpd       $0x0f  , %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b3k,b3k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm6 , %%xmm12\n\t" // [c02,c12,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm7 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t" // k=2
	                "vmovapd              4*8(%[a]), %%xmm2 \n\t" // [a0k,a1k,---,---]
	                "vmovupd              8*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "vmovupd             10*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm7 , %%xmm7 , %%xmm6 \n\t" // [b2k,b2k,---,---]
	                "vshufpd       $0x0f  , %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b3k,b3k,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm6 , %%xmm12\n\t" // [c02,c12,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm7 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t" // k=3
	                "vmovapd              6*8(%[a]), %%xmm3 \n\t" // [a0k,a1k,---,---]
	                "vmovupd             12*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "vmovupd             14*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm7 , %%xmm7 , %%xmm6 \n\t" // [b2k,b2k,---,---]
	                "vshufpd       $0x0f  , %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b3k,b3k,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm6 , %%xmm12\n\t" // [c02,c12,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm7 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%xmm0 \n\t" // [a0k,a1k,---,---]
	                "vmovupd              0*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "vmovupd              2*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm7 , %%xmm7 , %%xmm6 \n\t" // [b2k,b2k,---,---]
	                "vshufpd       $0x0f  , %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b3k,b3k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm6 , %%xmm12\n\t" // [c02,c12,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm7 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t" // k=1
	                "vmovapd              2*8(%[a]), %%xmm1 \n\t" // [a0k,a1k,---,---]
	                "vmovupd              4*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "vmovupd              6*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm7 , %%xmm7 , %%xmm6 \n\t" // [b2k,b2k,---,---]
	                "vshufpd       $0x0f  , %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b3k,b3k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm6 , %%xmm12\n\t" // [c02,c12,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm7 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $8*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%xmm0 \n\t" // [a0k,a1k,---,---]
	                "vmovupd              0*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "vmovupd              2*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm7 , %%xmm7 , %%xmm6 \n\t" // [b2k,b2k,---,---]
	                "vshufpd       $0x0f  , %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b3k,b3k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm6 , %%xmm12\n\t" // [c02,c12,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm7 , %%xmm13\n\t" // [c03,c13,---,---]
	                "\n\t"
	                "addq  $2*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        //  %%xmm10 [c00,c10,---,---]
	        //  %%xmm11 [c01,c11,---,---]
	        //  %%xmm12 [c02,c12,---,---]
	        //  %%xmm13 [c03,c13,---,---]
	        //  %%xmm14 [c04,c14,---,---]
	        //  %%xmm15 [c05,c15,---,---]

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%xmm0, %%xmm10\n\t"
	            "vfmadd213pd 0*8(%[c1]          ), %%xmm0, %%xmm11\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc2]  ), %%xmm0, %%xmm12\n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2]  ), %%xmm0, %%xmm13\n\t"
	            "\n\t"
	            "vmovupd  %%xmm10, 0*8(%[c0]          )\n\t"
	            "vmovupd  %%xmm11, 0*8(%[c1]          )\n\t"
	            "vmovupd  %%xmm12, 0*8(%[c0],%[ldc2]  )\n\t"
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
	    if( M & 1 ){

	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);

	        __asm__ __volatile__ (
	            "\n\t"
	            "prefetcht0  0*8(%[c0]          )\n\t"
	            "prefetcht0  0*8(%[c1]          )\n\t"
	            "prefetcht0  0*8(%[c0],%[ldc2]  )\n\t"
	            "prefetcht0  0*8(%[c1],%[ldc2]  )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[ldc2]"r"(ldc2)
	        );

	        if( K >> 2 ){
	          size_t k = ( K >> 2 );
	          while( k-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovsd               0*8(%[a]), %%xmm0 \n\t" // [a0k,---,---,---]
	                "vmovupd              0*8(%[b]), %%xmm4 \n\t" // [b0k,b1k,---,---]
	                "vmovupd              2*8(%[b]), %%xmm6 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm0 , %%xmm0 , %%xmm0 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm13\n\t" // [c00,c01,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm6 , %%xmm14\n\t" // [c02,c03,---,---]
	                "\n\t" // k=1
	                "vmovsd               1*8(%[a]), %%xmm1 \n\t" // [a0k,---,---,---]
	                "vmovupd              4*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vmovupd              6*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm1 , %%xmm1 , %%xmm1 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm5 , %%xmm13\n\t" // [c00,c01,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm7 , %%xmm14\n\t" // [c02,c03,---,---]
	                "\n\t" // k=2
	                "vmovsd               2*8(%[a]), %%xmm2 \n\t" // [a0k,---,---,---]
	                "vmovupd              8*8(%[b]), %%xmm4 \n\t" // [b0k,b1k,---,---]
	                "vmovupd             10*8(%[b]), %%xmm6 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm2 , %%xmm2 , %%xmm2 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm4 , %%xmm13\n\t" // [c00,c01,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm6 , %%xmm14\n\t" // [c02,c03,---,---]
	                "\n\t" // k=3
	                "vmovsd               3*8(%[a]), %%xmm3 \n\t" // [a0k,---,---,---]
	                "vmovupd             12*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vmovupd             14*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm3 , %%xmm3 , %%xmm3 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm5 , %%xmm13\n\t" // [c00,c01,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm7 , %%xmm14\n\t" // [c02,c03,---,---]
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovsd               0*8(%[a]), %%xmm0 \n\t" // [a0k,---,---,---]
	                "vmovupd              0*8(%[b]), %%xmm4 \n\t" // [b0k,b1k,---,---]
	                "vmovupd              2*8(%[b]), %%xmm6 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm0 , %%xmm0 , %%xmm0 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm13\n\t" // [c00,c01,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm6 , %%xmm14\n\t" // [c02,c03,---,---]
	                "\n\t" // k=1
	                "vmovsd               1*8(%[a]), %%xmm1 \n\t" // [a0k,---,---,---]
	                "vmovupd              4*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vmovupd              6*8(%[b]), %%xmm7 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm1 , %%xmm1 , %%xmm1 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm5 , %%xmm13\n\t" // [c00,c01,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm7 , %%xmm14\n\t" // [c02,c03,---,---]
	                "\n\t"
	                "addq  $2*8, %[a]\n\t"
	                "addq  $8*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovsd               0*8(%[a]), %%xmm0 \n\t" // [a0k,---,---,---]
	                "vmovupd              0*8(%[b]), %%xmm4 \n\t" // [b0k,b1k,---,---]
	                "vmovupd              2*8(%[b]), %%xmm6 \n\t" // [b2k,b3k,---,---]
	                "vshufpd       $0x00  , %%xmm0 , %%xmm0 , %%xmm0 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm13\n\t" // [c00,c01,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm6 , %%xmm14\n\t" // [c02,c03,---,---]
	                "\n\t"
	                "addq  $1*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        //  %%xmm13 [c00,c01,---,---]
	        //  %%xmm14 [c02,c03,---,---]
	        //  %%xmm15 [c04,c05,---,---]

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vmovlpd  0*8(%[c0]          ), %%xmm10, %%xmm10\n\t"
	            "vmovhpd  0*8(%[c1]          ), %%xmm10, %%xmm10\n\t"
	            "vmovlpd  0*8(%[c0],%[ldc2]  ), %%xmm11, %%xmm11\n\t"
	            "vmovhpd  0*8(%[c1],%[ldc2]  ), %%xmm11, %%xmm11\n\t"
	            "\n\t"
	            "vfmadd213pd  %%xmm10, %%xmm0, %%xmm13\n\t"
	            "vfmadd213pd  %%xmm11, %%xmm0, %%xmm14\n\t"
	            "\n\t"
	            "vmovlpd  %%xmm13, 0*8(%[c0]          )\n\t"
	            "vmovhpd  %%xmm13, 0*8(%[c1]          )\n\t"
	            "vmovlpd  %%xmm14, 0*8(%[c0],%[ldc2]  )\n\t"
	            "vmovhpd  %%xmm14, 0*8(%[c1],%[ldc2]  )\n\t"
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
/****************************************** N mod 6 = 2 *****************************************************/
	if( N6R & 2 ){ 

	    if( M >> 3 ){
	      size_t m = ( M >> 3 );
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
	            "prefetcht0  0*8(%[c0]          )\n\t"
	            "prefetcht0  4*8(%[c0]          )\n\t"
	            "prefetcht0  0*8(%[c1]          )\n\t"
	            "prefetcht0  4*8(%[c1]          )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	        :);


	        if( K >> 2 ){
	          size_t k = ( K >> 2 );
	          while( k-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128       0*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "\n\t" // k=1
	                "vmovapd              8*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd             12*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128       2*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "\n\t" // k=2
	                "vmovapd             16*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd             20*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128       4*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "\n\t" // k=3
	                "vmovapd             24*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd             28*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128       6*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "\n\t"
	                "addq  $32*8, %[a]\n\t"
	                "addq  $8*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128       0*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "\n\t" // k=1
	                "vmovapd              8*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd             12*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128       2*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastf128       0*8(%[b]), %%ymm3 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm3 , %%ymm3 , %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm3 , %%ymm3 , %%ymm3 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "vfmadd231pd   %%ymm0 , %%ymm3 , %%ymm6 \n\t" // [c01,c11,c21,c31]
	                "vfmadd231pd   %%ymm1 , %%ymm3 , %%ymm7 \n\t" // [c41,c51,c61,c71]
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        //  %%ymm4  [c00,c10,c20,c30]
	        //  %%ymm5  [c40,c50,c60,c70]
	        //  %%ymm6  [c01,c11,c21,c31]
	        //  %%ymm7  [c41,c51,c61,c71]
	        //  %%ymm8  [c02,c12,c22,c32]
	        //  %%ymm9  [c42,c52,c62,c72]
	        //  %%ymm10 [c03,c13,c23,c33]
	        //  %%ymm11 [c43,c53,c63,c73]

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%ymm0, %%ymm4 \n\t"
	            "vfmadd213pd 4*8(%[c0]          ), %%ymm0, %%ymm5 \n\t"
	            "vfmadd213pd 0*8(%[c1]          ), %%ymm0, %%ymm6 \n\t"
	            "vfmadd213pd 4*8(%[c1]          ), %%ymm0, %%ymm7 \n\t"
	            "\n\t"
	            "vmovupd  %%ymm4 , 0*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm5 , 4*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm6 , 0*8(%[c1]          )\n\t"
	            "vmovupd  %%ymm7 , 4*8(%[c1]          )\n\t"
	            "\n\t"
	            "addq  $8*8, %[c0]\n\t"
	            "addq  $8*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 2*K;

	      }
	    }
	    if( M & 4 ){

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
	            "prefetcht0  0*8(%[c0]          )\n\t"
	            "prefetcht0  0*8(%[c1]          )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	        :);


	        if( K >> 2 ){
	          size_t k = ( K >> 2 );
	          while( k-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128       0*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "\n\t" // k=1
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128       2*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm1 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "\n\t" // k=2
	                "vmovapd              8*8(%[a]), %%ymm2 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128       4*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm2 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm2 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "\n\t" // k=3
	                "vmovapd             12*8(%[a]), %%ymm3 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128       6*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm3 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm3 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $8*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128       0*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "\n\t" // k=1
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128       2*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm1 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastf128       0*8(%[b]), %%ymm5 \n\t" // [b0k,b1k,b0k,b1k]
	                "vshufpd       $0x00  , %%ymm5 , %%ymm5 , %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vshufpd       $0x0f  , %%ymm5 , %%ymm5 , %%ymm5 \n\t" // [b1k,b1k,b1k,b1k]
	                "vfmadd231pd   %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm0 , %%ymm5 , %%ymm11\n\t" // [c01,c11,c21,c31]
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        //  %%ymm10 [c00,c10,c20,c30]
	        //  %%ymm11 [c01,c11,c21,c31]
	        //  %%ymm12 [c02,c12,c22,c32]
	        //  %%ymm13 [c03,c13,c23,c33]
	        //  %%ymm14 [c04,c14,c24,c34]
	        //  %%ymm15 [c05,c15,c25,c35]

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%ymm0, %%ymm10\n\t"
	            "vfmadd213pd 0*8(%[c1]          ), %%ymm0, %%ymm11\n\t"
	            "\n\t"
	            "vmovupd  %%ymm10, 0*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm11, 0*8(%[c1]          )\n\t"
	            "\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "addq  $4*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 2*K;

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
	            "prefetcht0  0*8(%[c0]          )\n\t"
	            "prefetcht0  0*8(%[c1]          )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	        :);


	        if( K >> 2 ){
	          size_t k = ( K >> 2 );
	          while( k-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%xmm0 \n\t" // [a0k,a1k,---,---]
	                "vmovupd              0*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "\n\t" // k=1
	                "vmovapd              2*8(%[a]), %%xmm1 \n\t" // [a0k,a1k,---,---]
	                "vmovupd              2*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "\n\t" // k=2
	                "vmovapd              4*8(%[a]), %%xmm2 \n\t" // [a0k,a1k,---,---]
	                "vmovupd              4*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "\n\t" // k=3
	                "vmovapd              6*8(%[a]), %%xmm3 \n\t" // [a0k,a1k,---,---]
	                "vmovupd              6*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $8*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%xmm0 \n\t" // [a0k,a1k,---,---]
	                "vmovupd              0*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "\n\t" // k=1
	                "vmovapd              2*8(%[a]), %%xmm1 \n\t" // [a0k,a1k,---,---]
	                "vmovupd              2*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%xmm0 \n\t" // [a0k,a1k,---,---]
	                "vmovupd              0*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vshufpd       $0x0f  , %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b1k,b1k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm5 , %%xmm11\n\t" // [c01,c11,---,---]
	                "\n\t"
	                "addq  $2*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        //  %%xmm10 [c00,c10,---,---]
	        //  %%xmm11 [c01,c11,---,---]
	        //  %%xmm12 [c02,c12,---,---]
	        //  %%xmm13 [c03,c13,---,---]
	        //  %%xmm14 [c04,c14,---,---]
	        //  %%xmm15 [c05,c15,---,---]

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%xmm0, %%xmm10\n\t"
	            "vfmadd213pd 0*8(%[c1]          ), %%xmm0, %%xmm11\n\t"
	            "\n\t"
	            "vmovupd  %%xmm10, 0*8(%[c0]          )\n\t"
	            "vmovupd  %%xmm11, 0*8(%[c1]          )\n\t"
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
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        __asm__ __volatile__ (
	            "\n\t"
	            "prefetcht0  0*8(%[c0]          )\n\t"
	            "prefetcht0  0*8(%[c1]          )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	        :);


	        if( K >> 2 ){
	          size_t k = ( K >> 2 );
	          while( k-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovsd               0*8(%[a]), %%xmm0 \n\t" // [a0k,---,---,---]
	                "vmovupd              0*8(%[b]), %%xmm4 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm0 , %%xmm0 , %%xmm0 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm13\n\t" // [c00,c01,---,---]
	                "\n\t" // k=1
	                "vmovsd               1*8(%[a]), %%xmm1 \n\t" // [a0k,---,---,---]
	                "vmovupd              2*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm1 , %%xmm1 , %%xmm1 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm5 , %%xmm13\n\t" // [c00,c01,---,---]
	                "\n\t" // k=2
	                "vmovsd               2*8(%[a]), %%xmm2 \n\t" // [a0k,---,---,---]
	                "vmovupd              4*8(%[b]), %%xmm4 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm2 , %%xmm2 , %%xmm2 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm4 , %%xmm13\n\t" // [c00,c01,---,---]
	                "\n\t" // k=3
	                "vmovsd               3*8(%[a]), %%xmm3 \n\t" // [a0k,---,---,---]
	                "vmovupd              6*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm3 , %%xmm3 , %%xmm3 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm5 , %%xmm13\n\t" // [c00,c01,---,---]
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $8*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovsd               0*8(%[a]), %%xmm0 \n\t" // [a0k,---,---,---]
	                "vmovupd              0*8(%[b]), %%xmm4 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm0 , %%xmm0 , %%xmm0 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm13\n\t" // [c00,c01,---,---]
	                "\n\t" // k=1
	                "vmovsd               1*8(%[a]), %%xmm1 \n\t" // [a0k,---,---,---]
	                "vmovupd              2*8(%[b]), %%xmm5 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm1 , %%xmm1 , %%xmm1 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm5 , %%xmm13\n\t" // [c00,c01,---,---]
	                "\n\t"
	                "addq  $2*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovsd               0*8(%[a]), %%xmm0 \n\t" // [a0k,---,---,---]
	                "vmovupd              0*8(%[b]), %%xmm4 \n\t" // [b0k,b1k,---,---]
	                "vshufpd       $0x00  , %%xmm0 , %%xmm0 , %%xmm0 \n\t" // [a0k,a0k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm13\n\t" // [c00,c01,---,---]
	                "\n\t"
	                "addq  $1*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        //  %%xmm13 [c00,c01,---,---]
	        //  %%xmm14 [c02,c03,---,---]
	        //  %%xmm15 [c04,c05,---,---]

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vmovlpd  0*8(%[c0]          ), %%xmm10, %%xmm10\n\t"
	            "vmovhpd  0*8(%[c1]          ), %%xmm10, %%xmm10\n\t"
	            "\n\t"
	            "vfmadd213pd  %%xmm10, %%xmm0, %%xmm13\n\t"
	            "\n\t"
	            "vmovlpd  %%xmm13, 0*8(%[c0]          )\n\t"
	            "vmovhpd  %%xmm13, 0*8(%[c1]          )\n\t"
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
	    c0 = c0 - M + 2*ldc;
	    c1 = c1 - M + 2*ldc;

	}
/****************************************** N mod 6 = 1 *****************************************************/
	if( N6R & 1 ){ 

	    if( M >> 3 ){
	      size_t m = ( M >> 3 );
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
	            "prefetcht0  0*8(%[c0]          )\n\t"
	            "prefetcht0  4*8(%[c0]          )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	        :);


	        if( K >> 2 ){
	          size_t k = ( K >> 2 );
	          while( k-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastsd         0*8(%[b]), %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "\n\t" // k=1
	                "vmovapd              8*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd             12*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastsd         1*8(%[b]), %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "\n\t" // k=2
	                "vmovapd             16*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd             20*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastsd         2*8(%[b]), %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "\n\t" // k=3
	                "vmovapd             24*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd             28*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastsd         3*8(%[b]), %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "\n\t"
	                "addq  $32*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastsd         0*8(%[b]), %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "\n\t" // k=1
	                "vmovapd              8*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd             12*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastsd         1*8(%[b]), %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a4k,a5k,a6k,a7k]
	                "vbroadcastsd         0*8(%[b]), %%ymm2 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm2 , %%ymm4 \n\t" // [c00,c10,c20,c30]
	                "vfmadd231pd   %%ymm1 , %%ymm2 , %%ymm5 \n\t" // [c40,c50,c60,c70]
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $1*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        //  %%ymm4  [c00,c10,c20,c30]
	        //  %%ymm5  [c40,c50,c60,c70]
	        //  %%ymm6  [c01,c11,c21,c31]
	        //  %%ymm7  [c41,c51,c61,c71]
	        //  %%ymm8  [c02,c12,c22,c32]
	        //  %%ymm9  [c42,c52,c62,c72]
	        //  %%ymm10 [c03,c13,c23,c33]
	        //  %%ymm11 [c43,c53,c63,c73]

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%ymm0, %%ymm4 \n\t"
	            "vfmadd213pd 4*8(%[c0]          ), %%ymm0, %%ymm5 \n\t"
	            "\n\t"
	            "vmovupd  %%ymm4 , 0*8(%[c0]          )\n\t"
	            "vmovupd  %%ymm5 , 4*8(%[c0]          )\n\t"
	            "\n\t"
	            "addq  $8*8, %[c0]\n\t"
	            "addq  $8*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 1*K;

	      }
	    }
	    if( M & 4 ){

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
	            "prefetcht0  0*8(%[c0]          )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	        :);


	        if( K >> 2 ){
	          size_t k = ( K >> 2 );
	          while( k-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastsd         0*8(%[b]), %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "\n\t" // k=1
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastsd         1*8(%[b]), %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm1 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "\n\t" // k=2
	                "vmovapd              8*8(%[a]), %%ymm2 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastsd         2*8(%[b]), %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm2 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "\n\t" // k=3
	                "vmovapd             12*8(%[a]), %%ymm3 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastsd         3*8(%[b]), %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm3 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastsd         0*8(%[b]), %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "\n\t" // k=1
	                "vmovapd              4*8(%[a]), %%ymm1 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastsd         1*8(%[b]), %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm1 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a1k,a2k,a3k]
	                "vbroadcastsd         0*8(%[b]), %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm4 , %%ymm10\n\t" // [c00,c10,c20,c30]
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $1*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        //  %%ymm10 [c00,c10,c20,c30]
	        //  %%ymm11 [c01,c11,c21,c31]
	        //  %%ymm12 [c02,c12,c22,c32]
	        //  %%ymm13 [c03,c13,c23,c33]
	        //  %%ymm14 [c04,c14,c24,c34]
	        //  %%ymm15 [c05,c15,c25,c35]

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%ymm0, %%ymm10\n\t"
	            "\n\t"
	            "vmovupd  %%ymm10, 0*8(%[c0]          )\n\t"
	            "\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "addq  $4*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 1*K;

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
	            "prefetcht0  0*8(%[c0]          )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	        :);


	        if( K >> 2 ){
	          size_t k = ( K >> 2 );
	          while( k-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%xmm0 \n\t" // [a0k,a1k,---,---]
	                "vmovsd               0*8(%[b]), %%xmm5 \n\t" // [b0k,---,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "\n\t" // k=1
	                "vmovapd              2*8(%[a]), %%xmm1 \n\t" // [a0k,a1k,---,---]
	                "vmovsd               1*8(%[b]), %%xmm5 \n\t" // [b0k,---,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "\n\t" // k=2
	                "vmovapd              4*8(%[a]), %%xmm2 \n\t" // [a0k,a1k,---,---]
	                "vmovsd               2*8(%[b]), %%xmm5 \n\t" // [b0k,---,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vfmadd231pd   %%xmm2 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "\n\t" // k=3
	                "vmovapd              6*8(%[a]), %%xmm3 \n\t" // [a0k,a1k,---,---]
	                "vmovsd               3*8(%[b]), %%xmm5 \n\t" // [b0k,---,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vfmadd231pd   %%xmm3 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%xmm0 \n\t" // [a0k,a1k,---,---]
	                "vmovsd               0*8(%[b]), %%xmm5 \n\t" // [b0k,---,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "\n\t" // k=1
	                "vmovapd              2*8(%[a]), %%xmm1 \n\t" // [a0k,a1k,---,---]
	                "vmovsd               1*8(%[b]), %%xmm5 \n\t" // [b0k,---,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vfmadd231pd   %%xmm1 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovapd              0*8(%[a]), %%xmm0 \n\t" // [a0k,a1k,---,---]
	                "vmovsd               0*8(%[b]), %%xmm5 \n\t" // [b0k,---,---,---]
	                "vshufpd       $0x00  , %%xmm5 , %%xmm5 , %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm10\n\t" // [c00,c10,---,---]
	                "\n\t"
	                "addq  $2*8, %[a]\n\t"
	                "addq  $1*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        //  %%xmm10 [c00,c10,---,---]
	        //  %%xmm11 [c01,c11,---,---]
	        //  %%xmm12 [c02,c12,---,---]
	        //  %%xmm13 [c03,c13,---,---]
	        //  %%xmm14 [c04,c14,---,---]
	        //  %%xmm15 [c05,c15,---,---]

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]          ), %%xmm0, %%xmm10\n\t"
	            "\n\t"
	            "vmovupd  %%xmm10, 0*8(%[c0]          )\n\t"
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
	            "\n\t"
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        __asm__ __volatile__ (
	            "\n\t"
	            "prefetcht0  0*8(%[c0]          )\n\t"
	            "\n\t"
	            :[c0]"+r"(c0)
	        :);


	        if( K >> 2 ){
	          size_t k = ( K >> 2 );
	          while( k-- ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0,1,2,3
	                "vmovapd              0*8(%[a]), %%ymm0 \n\t" // [a0k,a0k,a0k,a0k]
	                "vmovupd              0*8(%[b]), %%ymm4 \n\t" // [b0k,b0k,b0k,b0k]
	                "vfmadd231pd   %%ymm0 , %%ymm4 , %%ymm13\n\t" // [c00,c00,c00,c00]
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	          }
	        }
	        if( K & 2 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0,1
	                "vmovapd              0*8(%[a]), %%xmm0 \n\t" // [a0k,a0k,---,---]
	                "vmovupd              0*8(%[b]), %%xmm4 \n\t" // [b0k,b0k,---,---]
	                "vfmadd231pd   %%xmm0 , %%xmm4 , %%xmm13\n\t" // [c00,c00,---,---]
	                "\n\t"
	                "addq  $2*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){

	            __asm__ __volatile__ (
	                "\n\t" // k=0
	                "vmovsd               0*8(%[a]), %%xmm0 \n\t" // [a0k,---,---,---]
	                "vmovsd               0*8(%[b]), %%xmm4 \n\t" // [b0k,---,---,---]
	                "vfmadd231sd   %%xmm0 , %%xmm4 , %%xmm13\n\t" // [c00,---,---,---]
	                "\n\t"
	                "addq  $1*8, %[a]\n\t"
	                "addq  $1*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }

	        //  %%xmm13 [c00,c01,---,---]
	        //  %%xmm14 [c02,c03,---,---]
	        //  %%xmm15 [c04,c05,---,---]

	        __asm__ __volatile__ (
	            "\n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vperm2f128   $0x01, %%ymm13, %%ymm13, %%ymm12\n\t"
	            "vaddpd              %%xmm12, %%xmm13, %%xmm13\n\t"
	            "vshufpd      $0x01, %%xmm13, %%xmm13, %%xmm12\n\t"
	            "vaddsd              %%xmm12, %%xmm13, %%xmm13\n\t"
	            "\n\t"
	            "vfmadd213sd  0*8(%[c0]          ), %%xmm0, %%xmm13\n\t"
	            "\n\t"
	            "vmovlpd  %%xmm13, 0*8(%[c0]          )\n\t"
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
	    c0 = c0 - M + 1*ldc;
	    c1 = c1 - M + 1*ldc;

	}

	A = A + M*K;
	B = B - K*N;
	c0 = c0- ldc*N + M;
	c1 = c1- ldc*N + M;
	// ---- Kernel

}

